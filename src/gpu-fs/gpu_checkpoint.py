#!/usr/bin/env python3
"""Checkpoint and restore a HIP app with CRIU on one CPX partition of an MI300X.

  gpu_checkpoint.py profile CMD [ARGS...]
  gpu_checkpoint.py checkpoint --kernel NAME[:N] [--continue] DIR CMD [ARGS...]
  gpu_checkpoint.py restore DIR

profile runs CMD under klHook the same way checkpoint does, without
checkpointing, and then lists every kernel (by base name) with its launch count:
the names and counts --kernel NAME:N selects from.

checkpoint runs CMD under klHook and dumps it into DIR (which must not exist
yet) right before a kernel launch:
  --kernel NAME[:N]  the N-th launch (default 1st) of kernel NAME, by base or
                     mangled name (default: $KL_CKPT_KERNEL)
  --continue         dump with --leave-running and let the app finish; exit
                     with its status (otherwise criu kills the app)
The app runs in the current directory; its output goes to DIR/app.log.
For profile and checkpoint, KL_HOOK_* variables (e.g. KL_HOOK_HIP_LIB_PATH)
are passed on to klHook.
DIR, the cwd and every file the app had mapped must keep their absolute paths.

restore restores DIR and waits for the app to exit. On the MI300X host it
restores onto a CPX partition again with the stock amdgpu plugin; without
amd-smi (the gem5 guest) it restores directly with amdgpu_plugin_gem5 and
copies the logs out with m5 writefile.

On the host the GPU is CPX during the run and SPX afterwards: amdgpu-dkms 7.1.9
cannot restore queues of multi-XCC GPUs. The app and criu share a device cgroup
with only that partition, so the checkpoint holds one GPU.
Env: GPU (PCI address), PART (CPX partition 0-7), CRIU, PLUGIN (plugin
directory; default criu/plugins/amdgpu, or criu/plugins/amdgpu/gem5 for a
restore in the gem5 guest), KLHOOK.
"""
import argparse
import glob
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.realpath(__file__))
GPU = os.environ.get("GPU", "0000:f6:00.0")
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
    sudo("amd-smi", "set", "--gpu", GPU, "--compute-partition", mode,
         stdout=subprocess.DEVNULL)


def render_node():
    """Render node of partition PART of GPU.

    KFD's location_id is bus << 8 | dev << 3 | fn, plus the partition index.
    """
    domain, bus, devfn = GPU.split(":")
    dev, fn = devfn.split(".")
    loc = (int(bus, 16) << 8 | int(dev, 16) << 3 | int(fn, 16)) + PART
    for path in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties"):
        with open(path) as f:
            props = dict(l.split() for l in f if len(l.split()) == 2)
        if (props.get("location_id") == str(loc)
                and props.get("domain") == str(int(domain, 16))):
            return "renderD" + props["drm_render_minor"]
    raise Error(f"no render node for {GPU} partition {PART}")


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
        render = render_node()
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


def restore(args):
    d = os.path.abspath(args.dir)
    if not os.path.isfile(os.path.join(d, "app.log.at-dump")):
        raise Error(f"no checkpoint in {d}")
    os.chdir(d)
    shutil.copy2("app.log.at-dump", "app.log")  # criu checks the size of app.log

    native = shutil.which("amd-smi") and os.path.exists(f"/sys/bus/pci/devices/{GPU}")
    plugin = PLUGIN or (PLUGIN_STOCK if native else PLUGIN_GEM5)
    try:
        prefix = []
        if native:
            set_partition("CPX")
            prefix = ["systemd-run", "--quiet", "--scope", *device_cgroup(render_node())]
        # Skip the checks where gem5's MI300X differs from the real one: MEC
        # and SDMA firmware versions, RAS capability bits, VRAM size, caches.
        checks = ["KFD_FW_VER_CHECK=0", "KFD_SDMA_FW_VER_CHECK=0",
                  "KFD_CAPABILITY_CHECK=0", "KFD_VRAM_SIZE_CHECK=0",
                  "KFD_CACHES_COUNT_CHECK=0"]
        rc = sudo(*prefix, "env", *checks, CRIU, "restore", "-D", d, "-v4",
                  "-o", "restore.log", *CRIU_OPTS, "--skip-file-rwx-check",
                  "-L", plugin, check=False).returncode
        print(f"criu restore exit {rc}", flush=True)
        with open("app.log") as f:
            sys.stdout.write(f.read())
        sys.stdout.flush()
        if shutil.which("m5"):
            subprocess.run(["m5", "writefile", "app.log", "app.log.after-restore"])
            subprocess.run(["m5", "writefile", "restore.log", "restore.log"])
        shutil.copy2("app.log.at-dump", "app.log")  # keep the checkpoint restorable
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
    c.add_argument("dir")
    c.add_argument("cmd", nargs=argparse.REMAINDER)
    r = sub.add_parser("restore", help="restore a checkpoint")
    r.add_argument("dir")
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
