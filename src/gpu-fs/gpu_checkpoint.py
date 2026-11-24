#!/usr/bin/env python3
"""Checkpoint and restore a HIP app with CRIU on one CPX partition of an MI300X.

  gpu_checkpoint.py profile CMD [ARGS...]
  gpu_checkpoint.py checkpoint --kernel NAME[:N] [--continue] [--gem5] DIR CMD [ARGS...]
  gpu_checkpoint.py restore DIR
  gpu_checkpoint.py restore --gem5 [--overlay ODIR] [DIR]      (in a gem5 guest)

profile runs CMD under klHook the same way checkpoint does, without
checkpointing, and then lists every kernel (by base name) with its launch count:
the names and counts --kernel NAME:N selects from.

checkpoint runs CMD under klHook and dumps it into DIR (which must not exist
yet) right before a kernel launch:
  --kernel NAME[:N]  the N-th launch (default 1st) of kernel NAME, by base or
                     mangled name (default: $KL_CKPT_KERNEL)
  --continue         dump with --leave-running and let the app finish; exit
                     with its status (otherwise criu kills the app)
  --gem5             also write DIR.gem5.tar.gz, the package for the gem5
                     guest image: <name>/rootfs/ holds the checkpoint, the
                     app's cwd and every file its process tree had mapped or
                     open at the dump, at their absolute paths, with owners
                     and modes; plus SHA256SUMS and manifest.json (paths,
                     command, kernel, GPU shape)
The app runs in the current directory; its output goes to DIR/app.log.
For profile and checkpoint, KL_HOOK_* variables (e.g. KL_HOOK_HIP_LIB_PATH)
are passed on to klHook.
DIR, the cwd and every file the app had mapped must keep their absolute paths.

restore restores DIR and waits for the app to exit. On the MI300X host it
restores onto a CPX partition again with the stock amdgpu plugin; without
amd-smi (the gem5 guest) it restores directly with amdgpu_plugin_gem5 and
copies the logs out with m5 writefile.
  --gem5          in a gem5 guest, as root: this script is the app gem5
                  runs (--app gpu_checkpoint.py --opts "restore --gem5"). With
                  a package disk (utils/make_pkg_disk.sh; label gem5pkg) it
                  lays the disk's rootfs/ over / (overlayfs) and takes criu,
                  the plugin and DIR (if not given) from the disk; without
                  one DIR must be installed in the image (criu from CRIU and
                  PLUGIN). It logs fatal signals to dmesg, restores with
                  amdgpu_plugin_gem5 and copies out with m5 writefile the
                  app's log, restore.log, the files the app created or
                  changed in DIR, the dmesg tail and a listing of DIR.
                  utils/gem5_verdict.py OUTDIR then says PASS or FAIL.
  --overlay ODIR  with --gem5: in the guest, first put each file
                  ODIR/<path> in place at /<path> (default: $OVERLAY)

On the host the GPU is CPX during the run and SPX afterwards: amdgpu-dkms 7.1.9
cannot restore queues of multi-XCC GPUs. The app and criu share a device cgroup
with only that partition, so the checkpoint holds one GPU.
Env: GPU (PCI address; default: the first GPU), PART (CPX partition 0-7), CRIU, PLUGIN (plugin
directory; default criu/plugins/amdgpu, or criu/plugins/amdgpu/gem5 for a
restore in the gem5 guest), KLHOOK.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import pwd
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.realpath(__file__))
GPU = os.environ.get("GPU")  # PCI address; default: the first GPU KFD reports
PART = int(os.environ.get("PART", "0"))
CRIU = os.environ.get("CRIU", os.path.join(HERE, "criu/criu/criu"))
# CRIU loads every plugin in its -L directory: the stock amdgpu plugin and
# amdgpu_plugin_gem5 (criu-patches/) are built in separate directories.
PLUGIN_STOCK = os.path.join(HERE, "criu/plugins/amdgpu")
PLUGIN_GEM5 = os.path.join(HERE, "criu/plugins/amdgpu/gem5")
PLUGIN = os.environ.get("PLUGIN")
KLHOOK = os.environ.get("KLHOOK", os.path.join(HERE, "klHook/libklhook.so"))
SUDO = [] if os.geteuid() == 0 else ["sudo"]
CRIU_OPTS = ["--link-remap", "--tcp-established", "--file-locks"]

# gem5's CPU is x86-64-v2: hide newer ISA from glibc's ifunc selection.
GLIBC_HWCAPS = ",".join("-" + f for f in (
    "AVX512F AVX512CD AVX512BW AVX512DQ AVX512VL AVX512_VBMI AVX512_VBMI2 "
    "AVX512_IFMA AVX512_VNNI AVX512_BITALG AVX512_VPOPCNTDQ AVX512_BF16 "
    "AVX512_FP16 AVX_VNNI AVX2 BMI1 BMI2 LZCNT MOVBE").split())


class Error(Exception):
    pass


def sudo(*cmd, check=True, **kw):
    return subprocess.run([*SUDO, *cmd], check=check, **kw)


def set_partition(mode):
    sudo("amd-smi", "set", "--gpu", gpu(), "--compute-partition", mode,
         stdout=subprocess.DEVNULL)


def kfd_nodes():
    for path in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties"):
        with open(path) as f:
            yield dict(l.split() for l in f if len(l.split()) == 2)


def gpu():
    """PCI address of the GPU: $GPU, else the first GPU KFD reports (in CPX
    mode its partition 0 has the lowest location_id)."""
    global GPU
    if not GPU:
        gpus = [(int(p["domain"]), int(p["location_id"])) for p in kfd_nodes()
                if int(p.get("simd_count", 0)) > 0]
        if not gpus:
            raise Error("no GPU in the KFD topology (set GPU=<PCI address>)")
        domain, loc = min(gpus)
        GPU = f"{domain:04x}:{loc >> 8:02x}:{(loc >> 3) & 0x1f:02x}.{loc & 7:x}"
    return GPU


def node_props():
    """KFD topology properties of partition PART of the GPU.

    KFD's location_id is bus << 8 | dev << 3 | fn, plus the partition index.
    """
    domain, bus, devfn = gpu().split(":")
    dev, fn = devfn.split(".")
    loc = (int(bus, 16) << 8 | int(dev, 16) << 3 | int(fn, 16)) + PART
    for props in kfd_nodes():
        if (props.get("location_id") == str(loc)
                and props.get("domain") == str(int(domain, 16))):
            return props
    raise Error(f"no KFD node for {gpu()} partition {PART}")


def render_node():
    return "renderD" + node_props()["drm_render_minor"]


def device_cgroup(render):
    """systemd-run properties allowing only /dev/kfd and one render node.

    KFD filters its topology by the caller's device cgroup, so the app and
    criu both see one GPU.
    """
    return ["-p", "DevicePolicy=closed", "-p", "DeviceAllow=/dev/kfd rw",
            "-p", f"DeviceAllow=/dev/dri/{render} rw"]


def unit_prop(unit, prop):
    return subprocess.run(["systemctl", "show", "-P", prop, unit],
                          capture_output=True, text=True).stdout.strip()


def app_command(argv):
    """argv with an absolute path to the program (not resolved: keeps venvs)."""
    if not argv:
        raise Error("no command given")
    exe = shutil.which(argv[0])
    if not exe:
        raise Error(f"not found: {argv[0]}")
    if not os.path.isfile(KLHOOK):
        raise Error(f"no {KLHOOK} (run build.sh first)")
    return [os.path.abspath(exe), *argv[1:]]


def app_unit(dev, env):
    """systemd-run properties to run the app as us, in the cwd, under klHook."""
    env = {"HSA_ENABLE_INTERRUPT": "0",  # gem5 sends no completion IRQs
           # No large BAR: HIP reports isLargeBar=0, so neither HIP nor the
           # app (e.g. PyTorch's tensor.item() on ROCm >= 7.2) reads VRAM
           # directly with the CPU. In gem5 the VRAM BAR is MMIO, KVM must
           # emulate such loads, and it cannot emulate SSE moves like movss:
           # the restored app dies with SIGILL. The choice is made at startup
           # and checkpointed, so it must be set when the app starts.
           "ROC_ENABLE_LARGE_BAR": "0",
           "GLIBC_TUNABLES": f"glibc.cpu.hwcaps={GLIBC_HWCAPS}",
           "LD_PRELOAD": KLHOOK, **env}
    env.update((k, v) for k, v in os.environ.items() if k.startswith("KL_HOOK_"))
    # A service (not --uid) keeps our supplementary groups, e.g. render.
    return [*dev, "-p", f"User={pwd.getpwuid(os.getuid()).pw_name}",
            "-p", f"WorkingDirectory={os.getcwd()}",
            *[f"--setenv={k}={v}" for k, v in env.items()]]


# --- gem5 package -----------------------------------------------------------
# CRIU reopens every file the process tree had mapped or open, by path, and
# checks its size; the guest image must hold them byte for byte. The package
# is <name>/rootfs/<absolute path> for each of them, plus the whole checkpoint
# dir and the app's cwd, with their owners and modes (the restored app runs
# with the checkpoint's uid/gid), SHA256SUMS over rootfs/ and manifest.json.
SKIP_PREFIXES = ("/dev/", "/proc/", "/sys/", "/memfd:", "/dmabuf")
GPU_SHAPE = ("num_xcc", "simd_count", "array_count", "cu_per_simd_array",
             "simd_per_cu", "num_sdma_engines", "num_sdma_xgmi_engines",
             "gfx_target_version", "device_id")


def process_files(pid):
    """(pids, cwds, files) of the process tree of pid: every regular file it
    has mapped or open, its executables and working directories."""
    script = r"""
set -- $1
while [ $# -gt 0 ]; do
    p=$1; shift
    echo "P $p"; echo "C $(readlink /proc/$p/cwd)"; echo "F $(readlink /proc/$p/exe)"
    sed 's/^/M /' /proc/$p/maps
    for f in /proc/$p/fd/*; do echo "F $(readlink $f)"; done
    set -- "$@" $(cat /proc/$p/task/*/children 2>/dev/null)
done
"""
    out = sudo("sh", "-c", script, "sh", str(pid), capture_output=True,
               text=True).stdout
    pids, cwds, files = [], set(), set()
    for line in out.splitlines():
        tag, _, rest = line.partition(" ")
        if tag == "M":
            f = rest.split(None, 5)
            rest = f[5] if len(f) == 6 else ""
        if tag == "P":
            pids.append(int(rest))
        elif tag == "C":
            cwds.add(rest)
        elif (rest.startswith("/") and not rest.endswith(" (deleted)")
              and not rest.startswith(SKIP_PREFIXES) and os.path.isfile(rest)):
            files.add(rest)
    return pids, sorted(cwds), sorted(files)


def make_package(d, listing, cmd, kernel, props):
    """Write <d>.gem5.tar.gz: everything the gem5 guest image needs to
    restore checkpoint d. listing is process_files() of the dumped tree,
    taken right before the dump."""
    name = os.path.basename(d)
    pids, cwds, files = listing
    files = list(files)
    for root, _, fs in os.walk(d):
        files += [os.path.join(root, f) for f in fs if not f.startswith(".kl-")]
    files = sorted(set(files))
    dirs = set(cwds) | {d}
    for p in list(dirs) + files:  # every ancestor, so owners/modes come along
        while p != "/":
            p = os.path.dirname(p)
            dirs.add(p)
    dirs.discard("/")
    manifest = {
        "checkpoint": d, "cwd": cwds, "command": cmd, "kernel": kernel,
        "pids": pids, "gpu": {k: props.get(k) for k in GPU_SHAPE},
        # gem5's guest must show the same GPU shape (make_1xcc_ipt.py)
        "ipt": "1xcc" if props.get("num_xcc") == "1" else "8xcc",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = d + ".gem5.tar.gz"
    sums = []
    with tarfile.open(out, "w:gz", compresslevel=1) as tf:
        def add_bytes(arc, data):
            ti = tarfile.TarInfo(f"{name}/{arc}")
            ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
            tf.addfile(ti, io.BytesIO(data))
        for p in sorted(dirs):
            tf.add(p, f"{name}/rootfs{p}", recursive=False)
        for p in files:
            try:
                tf.add(p, f"{name}/rootfs{p}", recursive=False)
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    while chunk := f.read(1 << 20):
                        h.update(chunk)
                h = h.hexdigest()
            except PermissionError as e:
                raise Error(f"cannot read {p} for the package: {e}")
            sums.append(f"{h}  .{p}\n")
        add_bytes("SHA256SUMS", "".join(sums).encode())
        add_bytes("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(f"gem5 package: {out} ({len(files)} files, "
          f"{os.path.getsize(out) / 2**20:.0f} MiB)", flush=True)
    return out


def profile(args):
    cmd = app_command(args.cmd)
    tmp = tempfile.mkdtemp(prefix="gpu-profile-")
    counts = os.path.join(tmp, "kernels")
    try:
        set_partition("CPX")
        dev = device_cgroup(render_node())
        rc = sudo("systemd-run", "--quiet", "--pipe", "--wait", "--collect",
                  *app_unit(dev, {"KL_HOOK_COUNTS": counts}), *cmd,
                  check=False).returncode
        sys.stdout.flush()
        if not os.path.exists(counts):
            raise Error(f"app exited {rc} without launching a kernel (or before exit())")
        with open(counts) as f:
            rows = [l.rstrip("\n").split("\t", 1) for l in f]
        total = sum(int(n) for n, _ in rows)
        print(f"\n{total} kernel launches, {len(rows)} kernels (app exited {rc}):")
        for n, name in rows:
            print(f"{int(n):8d}  {name}")
        return rc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        set_partition("SPX")


def checkpoint(args):
    cmd = app_command(args.cmd)
    d = os.path.abspath(args.dir)
    os.mkdir(d)
    log = os.path.join(d, "app.log")
    ready, go = os.path.join(d, ".kl-ready"), os.path.join(d, ".kl-go")
    unit = f"gpu-checkpoint-{os.getpid()}"
    try:
        set_partition("CPX")
        props = node_props()
        render = "renderD" + props["drm_render_minor"]
        dev = device_cgroup(render)

        env = {"KL_CKPT_KERNEL": args.kernel, "KL_CKPT_READY": ready, "KL_CKPT_GO": go}
        sudo("systemd-run", "--quiet", f"--unit={unit}", *app_unit(dev, env),
             "-p", "StandardInput=null", "-p", f"StandardOutput=file:{log}",
             "-p", "StandardError=inherit", "-p", "RemainAfterExit=yes", *cmd)
        pid = unit_prop(unit, "MainPID")

        # klHook creates the ready file once the GPU is drained, right before the launch.
        while not os.path.exists(ready):
            if unit_prop(unit, "SubState") != "running":
                with open(log) as f:
                    sys.stdout.write(f.read())
                raise Error(f"app exited before {args.kernel} was launched")
            time.sleep(0.1)

        if args.gem5:  # before the dump: it may kill the process tree
            listing = process_files(pid)
        sudo("systemd-run", "--quiet", "--scope", *dev, CRIU, "dump", "-t", pid,
             "-D", d, "-v4", "-o", "dump.log", *CRIU_OPTS, "--unprivileged",
             "-L", PLUGIN or PLUGIN_STOCK, *(["--leave-running"] if args.cont else []))
        imgs = glob.glob(os.path.join(d, "*.img"))
        sudo("chown", f"{os.getuid()}:{os.getgid()}", *imgs,
             os.path.join(d, "dump.log"), log)
        for img in imgs:
            os.chmod(img, 0o644)
        shutil.copy2(log, log + ".at-dump")  # restore puts it back: criu checks its size
        print(f"checkpoint of {cmd[0]} before {args.kernel} (pid {pid}, {render}) in {d}",
              flush=True)
        if args.gem5:
            make_package(d, listing, cmd, args.kernel, props)

        rc = 0
        if args.cont:
            open(go, "w").close()
            while unit_prop(unit, "SubState") == "running":
                time.sleep(0.25)
            rc = int(unit_prop(unit, "ExecMainStatus") or 0)
            if unit_prop(unit, "ExecMainCode") != "1":  # 1 = CLD_EXITED, else a signal
                rc += 128
            print(f"app exited {rc}")
        return rc
    finally:
        for f in (ready, go):
            if os.path.exists(f):
                os.remove(f)
        sudo("systemctl", "stop", unit, check=False, stderr=subprocess.DEVNULL)
        sudo("systemctl", "reset-failed", unit, check=False, stderr=subprocess.DEVNULL)
        set_partition("SPX")


PKG_LABEL = "gem5pkg"
PKG_MNT = "/mnt/gem5pkg"


def mount_pkg_disk():
    """Mount the package disk (make_pkg_disk.sh), if gem5 attached one, and
    lay its rootfs/ over / with overlayfs, one mount per top-level directory
    (/opt, /usr, /users, ...): its files win over the image's at the same
    paths. Changes go to an upper layer on the root disk, which gem5 keeps
    copy-on-write. Returns the manifest, or None without a package disk."""
    dev = subprocess.run(["blkid", "-L", PKG_LABEL], capture_output=True,
                         text=True).stdout.strip()
    if not dev:
        return None
    os.makedirs(PKG_MNT, exist_ok=True)
    if not os.path.ismount(PKG_MNT):
        subprocess.run(["mount", "-o", "ro", dev, PKG_MNT], check=True)
    subprocess.run(["modprobe", "overlay"], check=False)
    root = os.path.join(PKG_MNT, "rootfs")
    for top in sorted(os.listdir(root)):
        lower, target = os.path.join(root, top), "/" + top
        work = f"/var/lib/gem5pkg/{top}"
        os.makedirs(target, exist_ok=True)
        os.makedirs(work + "/upper", exist_ok=True)
        os.makedirs(work + "/work", exist_ok=True)
        subprocess.run(["mount", "-t", "overlay", "overlay", "-o",
                        f"lowerdir={lower}:{target},upperdir={work}/upper,"
                        f"workdir={work}/work", target], check=True)
        print(f"package disk {dev}: {target}", flush=True)
    with open(os.path.join(PKG_MNT, "manifest.json")) as f:
        return json.load(f)


def overlay(src):
    """Put the files under src (at their absolute paths below it) in place.

    Files the checkpoint needs that differ from the image's are installed
    under an overlay dir instead, so other checkpoints keep theirs. gem5's
    disk is copy-on-write, so this lasts one run."""
    for root, _, fs in os.walk(src):
        for f in fs:
            s = os.path.join(root, f)
            dst = "/" + os.path.relpath(s, src)
            tmp = dst + ".overlay.tmp"
            shutil.copy2(s, tmp)
            os.chown(tmp, os.stat(s).st_uid, os.stat(s).st_gid)
            os.replace(tmp, dst)  # a new inode: running processes keep theirs
            print(f"overlay: {dst}", flush=True)


def writefile(path, name):
    subprocess.run(["m5", "writefile", path, name], stdout=subprocess.DEVNULL)


def dir_state(d):
    """name -> (size, mtime) of the regular files in d. Compared before and
    after the restore rather than against the time: files installed into the
    image can carry host timestamps ahead of the guest's clock."""
    st = {}
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            s = os.stat(p)
            st[f] = (s.st_size, s.st_mtime_ns)
    return st


def gem5_report(d, before):
    """Copy the results of a restore in the gem5 guest out to the host's
    outdir: the files the app created or changed in DIR (e.g. a result
    file), the kernel log (crashes, GPU faults) and a listing of DIR."""
    for f, st in sorted(dir_state(d).items()):
        if before.get(f) != st and not f.endswith((".img", ".log")):
            writefile(os.path.join(d, f), f)
    dmesg = subprocess.run(["dmesg"], capture_output=True, text=True).stdout
    tail = "".join(dmesg.splitlines(True)[-60:])
    with open("/tmp/dmesg-after-restore.txt", "w") as f:
        f.write(tail)
    writefile("/tmp/dmesg-after-restore.txt", "dmesg-after-restore.txt")
    with open("/tmp/ls-after-restore.txt", "w") as f:
        f.write(time.ctime() + "\n")
        f.write(subprocess.run(["ls", "-la", "--full-time", d],
                               capture_output=True, text=True).stdout)
    writefile("/tmp/ls-after-restore.txt", "ls-after-restore.txt")
    bad = [l for l in tail.splitlines()
           if any(w in l.lower() for w in ("segfault", "fatal", "signal", "fault", "traps"))]
    if bad:
        print("dmesg after restore:\n" + "\n".join(bad[-20:]), flush=True)


def restore(args):
    global CRIU
    plugin = PLUGIN
    if args.gem5:
        manifest = mount_pkg_disk()
        if manifest:
            args.dir = args.dir or manifest["checkpoint"]
            if "CRIU" not in os.environ:
                CRIU = os.path.join(PKG_MNT, "criu/criu/criu")
            plugin = plugin or os.path.join(PKG_MNT, "criu/plugins/amdgpu/gem5")
    if not args.dir:
        raise Error("no checkpoint dir given (and no package disk)")
    d = os.path.abspath(args.dir)
    # <log>.at-dump: the app's log as it was at the dump (app.log for
    # checkpoints made by this script). The log is an open file of the app;
    # criu checks its size, so it is put back before and after each restore.
    logs = [f[:-len(".at-dump")] for f in glob.glob(os.path.join(d, "*.at-dump"))]
    if not logs:
        raise Error(f"no checkpoint in {d} (no *.at-dump)")
    if args.gem5 and args.overlay:
        overlay(args.overlay)
    os.chdir(d)
    for log in logs:
        shutil.copy2(log + ".at-dump", log)

    native = (not args.gem5 and shutil.which("amd-smi")
              and os.path.exists(f"/sys/bus/pci/devices/{gpu()}"))
    plugin = plugin or (PLUGIN_STOCK if native else PLUGIN_GEM5)
    before = dir_state(d)
    try:
        prefix = []
        if native:
            set_partition("CPX")
            prefix = ["systemd-run", "--quiet", "--scope", *device_cgroup(render_node())]
        if args.gem5:
            # criu restore returns 0 once the restore succeeded, whatever the
            # app does afterwards: log fatal signals so a crash shows in dmesg.
            with open("/proc/sys/kernel/print-fatal-signals", "w") as f:
                f.write("1")
        # Skip the checks where gem5's MI300X differs from the real one: MEC
        # and SDMA firmware versions, RAS capability bits, VRAM size, caches.
        checks = ["KFD_FW_VER_CHECK=0", "KFD_SDMA_FW_VER_CHECK=0",
                  "KFD_CAPABILITY_CHECK=0", "KFD_VRAM_SIZE_CHECK=0",
                  "KFD_CACHES_COUNT_CHECK=0"]
        rc = sudo(*prefix, "env", *checks, CRIU, "restore", "-D", d, "-v4",
                  "-o", "restore.log", *CRIU_OPTS, "--skip-file-rwx-check",
                  "-L", plugin, check=False).returncode
        print(f"criu restore exit {rc}", flush=True)
        for log in logs:
            with open(log) as f:
                sys.stdout.write(f.read())
        sys.stdout.flush()
        if args.gem5 or shutil.which("m5"):
            for log in logs:
                writefile(log, os.path.basename(log) + ".after-restore")
            writefile("restore.log", "restore.log")
        if args.gem5:
            gem5_report(d, before)
        for log in logs:
            shutil.copy2(log + ".at-dump", log)  # keep the checkpoint restorable
        return rc
    finally:
        if native:
            set_partition("SPX")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("profile", help="run an app and list its kernels and launch counts")
    pr.add_argument("cmd", nargs=argparse.REMAINDER)
    c = sub.add_parser("checkpoint", help="run an app and checkpoint it")
    kernel = os.environ.get("KL_CKPT_KERNEL")
    c.add_argument("--kernel", metavar="NAME[:N]", default=kernel, required=kernel is None,
                   help="checkpoint before this kernel launch (default: $KL_CKPT_KERNEL)")
    c.add_argument("--continue", dest="cont", action="store_true",
                   help="let the app finish after the dump")
    c.add_argument("--gem5", action="store_true",
                   help="also write DIR.gem5.tar.gz, the package to install "
                        "into the gem5 guest image")
    c.add_argument("dir")
    c.add_argument("cmd", nargs=argparse.REMAINDER)
    r = sub.add_parser("restore", help="restore a checkpoint")
    r.add_argument("--gem5", action="store_true",
                   help="in a gem5 guest (gem5 --app gpu_checkpoint.py --opts "
                        "'restore --gem5'): restore from the package disk, copy "
                        "the results out")
    r.add_argument("--overlay", metavar="DIR", default=os.environ.get("OVERLAY"),
                   help="with --gem5: first put the files under DIR in place "
                        "(default: $OVERLAY)")
    r.add_argument("dir", nargs="?",
                   help="checkpoint dir (with --gem5 and a package disk: "
                        "default from its manifest)")
    args = p.parse_args()
    try:
        sys.exit({"profile": profile, "checkpoint": checkpoint,
                  "restore": restore}[args.command](args))
    except (Error, OSError) as e:
        sys.exit(f"{os.path.basename(sys.argv[0])}: {e}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"{os.path.basename(sys.argv[0])}: failed ({e.returncode}): {' '.join(e.cmd)}")


if __name__ == "__main__":
    main()
