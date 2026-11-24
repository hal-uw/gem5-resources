# GPU checkpoints with CRIU

Checkpoint a HIP application on a real AMD MI300X with
[CRIU](https://criu.org) and its amdgpu plugin, then restore it natively or
inside a gem5 GPU full-system guest. With [klHook](klHook), the checkpoint is
taken right before a chosen kernel launch. The application needs no changes.

| Path | What it is |
|---|---|
| `criu/` | Submodule: upstream CRIU, pinned to `v4.0` |
| `criu-patches/` | Patches applied to `criu/` by `build.sh`, in name order (see [CRIU patches](#criu-patches)) |
| `klHook/` | Submodule: upstream klHook (LD_PRELOAD hook for HIP launches), pinned to its `main` |
| `klhook-patches/` | Patches applied to `klHook/` by `build.sh`, in name order (see [klHook patches](#klhook-patches)) |
| `build.sh` | Applies the patches and builds criu, both amdgpu plugins and `libklhook.so` |
| `gpu_checkpoint.py` | `profile`: lists an app's kernels and launch counts; `checkpoint`: runs an app and checkpoints it; `restore`: restores it, on the host or (`--gem5`) in the gem5 guest |
| `utils/make_pkg_disk.sh` | Makes the package disk of a gem5 package |
| `utils/pkg_disk_config.py` | gem5 config wrapper: runs an unmodified board config with the package disk as second disk |
| `utils/gem5_verdict.py` | PASS/FAIL of a gem5 restore, from gem5's output directory |
| `utils/make_1xcc_ipt.py` | Makes the single-XCC IP-discovery binary the gem5 guest needs for a CPX checkpoint |

## Setup

```sh
git submodule update --init src/gpu-fs/criu src/gpu-fs/klHook
sudo apt install libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
    python3-protobuf libnet1-dev libcap-dev libbsd-dev libgnutls28-dev libnftables-dev \
    uuid-dev libnl-3-dev libdrm-dev pkg-config
src/gpu-fs/build.sh
```

ROCm comes from `/opt/rocm` (override it with `ROCM_PATH`). On the gem5 host
`build.sh` needs only criu and the plugins; it builds klHook (loaded into the
app at checkpoint time) only when the installed HIP has the launch APIs it
hooks (`hipLaunchKernelExC`, ROCm 7), and says why otherwise. klHook has no
device code, so it is compiled host-only, with the newest GCC installation
whose C++ headers are installed (clang otherwise takes the newest GCC, which
may lack them). The gem5 package disk also needs `fakeroot`, `e2fsprogs`
and `sfdisk` (util-linux).

## Checkpoint and restore

```sh
src/gpu-fs/gpu_checkpoint.py profile <app> [args...]
src/gpu-fs/gpu_checkpoint.py checkpoint --kernel <kernel>[:<n>] [--continue] [--gem5] <checkpoint-dir> <app> [args...]
src/gpu-fs/gpu_checkpoint.py restore [--gem5 [--overlay <dir>]] <checkpoint-dir>
```

- `profile` runs the app the same way `checkpoint` does (same CPX partition,
  cgroup and environment) without checkpointing, then prints every kernel's
  base name and launch count: what `--kernel <kernel>:<n>` picks from. Library
  kernels (e.g. hipBLASLt GEMMs) can differ between CPX and the full GPU.
- `--kernel <kernel>[:<n>]` (or `KL_CKPT_KERNEL`) checkpoints right before
  the n-th launch (default 1st) of the kernel, matched by its base name or its
  mangled name. `KL_HOOK_VERBOSE=1` prints every launched kernel's name.
- At the selected launch klHook waits for the GPU to finish all earlier work
  (via `hipDeviceSynchronize`), then hands the process over to be dumped. By
  default the dump kills the app. With `--continue` the app finishes natively,
  and the script returns the app's exit status.
- `--gem5` (checkpoint) also writes `<checkpoint-dir>.gem5.tar.gz`, the
  package for the gem5 guest image (see [gem5](#gem5)).
- `restore` waits for the restored app to exit and prints its output.
- After a restore, klHook notices on its own that it was restored and launches
  the kernel.
- The app runs in the current directory. Its output goes to `<checkpoint-dir>/app.log`. The checkpoint must stay at
  the same absolute path (it holds the app's cwd and its open `app.log`), and
  so must every file the app had mapped (binary, ROCm libraries,
  `libklhook.so`).
- Environment variables: `GPU` (PCI address, default: the first GPU), `PART`
  (CPX partition 0-7, default 0), `CRIU`, `PLUGIN` (plugin directory: by
  default `criu/plugins/amdgpu` for checkpoints and native restores,
  `criu/plugins/amdgpu/gem5` for restores in the gem5 guest), `KLHOOK`.
  `KL_HOOK_*` variables are passed on to klHook; an app that brings its own
  ROCm (e.g. a PyTorch wheel) needs `KL_HOOK_HIP_LIB_PATH` and
  `KL_HOOK_HSA_LIB_PATH` pointing at its `libamdhip64.so` and
  `libhsa-runtime64.so`.

### Why CPX

With amdgpu-dkms 7.1.9 the kernel rejects the restore of any queue on a GPU
with more than one XCC (`amdgpu: Failed to restore queue (-22)`).
`kfd_criu_restore_queue()` compares the saved MQD size (per-XCC size × XCCs)
with the size for a single XCC. An MI300X in SPX mode has 8 XCCs. The scripts
therefore switch the GPU to CPX (8 partitions of 1 XCC) for the run and back
to SPX afterwards. The GPU must be idle for that switch.

The app and criu run in a device cgroup that allows only `/dev/kfd` and one
partition's render node. KFD filters its topology by the caller's device
cgroup, so both see a single GPU, and the checkpoint holds exactly one GPU.

## CRIU patches

`criu-patches/0001-plugins-amdgpu-add-amdgpu_plugin_gem5.patch` makes
`make amdgpu_plugin` build two plugins from the same sources:

| Plugin | `-L` directory | Use |
|---|---|---|
| `amdgpu_plugin.so` | `criu/plugins/amdgpu` | Stock upstream behaviour. Checkpoints, and restores on the MI300X |
| `amdgpu_plugin_gem5.so` | `criu/plugins/amdgpu/gem5` | Restores in the gem5 guest |

CRIU loads every plugin in its `-L` directory, so the two are built in
separate directories; never put both in one. `amdgpu_plugin_gem5` differs
from the stock plugin in four ways:

- It restores onto the subset of the checkpointed GPUs that hold state (on by
  default; `KFD_RESTORE_GPU_SUBSET=0` turns it off). A process on one GPU of
  an 8-GPU node is checkpointed with all 8 GPUs, because ROCm opens every
  visible GPU, and the stock plugin then needs all 8 at restore. GPUs that
  hold nothing but their MMIO-remap page are left out. The image files are
  not changed. A checkpoint made by `gpu_checkpoint.py` holds one GPU, so
  nothing is left out.
- It counts only the GPU links it restores in the topology match. In a
  device cgroup the dump stores the raw link count of the partition (15 on an
  MI300X in CPX) but only the one link it can see; gem5's GPU has one link.
- It gives the restored process back its transparent huge pages. CRIU
  restores memory page by page and re-applies `madvise()` only at the end,
  so `MADV_HUGEPAGE` buffers (e.g. `hipHostMalloc`) come back as 4 KiB
  pages and KFD maps them for the GPU with 4 KiB PTEs: a restored kernel
  then has no 2 MiB TLB hits and many more page walks than the original.
  The plugin collapses those regions (`process_madvise(MADV_COLLAPSE)`,
  Linux 6.1+) before KFD resumes the process.
- It writes the SDMA `COPY_LINEAR` count as bytes - 1 on MI300X, as the
  kernel driver does. Upstream writes the byte count, which copies one byte
  past each buffer; gem5's SDMA model executes exactly that and corrupts the
  restored GPU memory.

## klHook patches

Upstream klHook only prints hooked calls. `build.sh` applies:

| Patch | Adds |
|---|---|
| `0001-Checkpoint-support-…` | `KL_CKPT_KERNEL`/`KL_CKPT_READY`/`KL_CKPT_GO`: drain the GPU and wait right before the selected launch; continue when told to or when restored. Hooks `hipLaunchKernelExC`, `hipLaunchCooperativeKernel` and `hipExtLaunchKernel` too |
| `0002-Support-PyTorch-kernels-…` | Base names for kernels in anonymous namespaces or with lambdas; `KL_HOOK_COUNTS` (used by `profile`); builds with `--as-needed` so an app's own HIP runtime is the only one loaded |

## gem5

`checkpoint --gem5` packages everything the guest needs into
`<checkpoint-dir>.gem5.tar.gz`. Right before the dump it lists every regular
file the app's process tree has mapped or open, its executables and working
directories. The package holds `<name>/rootfs/` with those files, the whole
checkpoint directory and the working directory, at their absolute paths and
with their owners and modes; `SHA256SUMS` over `rootfs/`; and
`manifest.json` (checkpoint path, cwd, command, kernel, and the GPU shape
from KFD: XCC, SIMD, array and SDMA counts, so `ipt` says whether the guest
needs the single-XCC IP-discovery binary).

On the gem5 host, nothing is installed into the guest image and gem5 is not
changed. Three steps:

```sh
# 1. the package disk (next to the tarball: <checkpoint-dir>.gem5.img)
BASE=<guest disk image> src/gpu-fs/utils/make_pkg_disk.sh <checkpoint-dir>.gem5.tar.gz
# 2. gem5, with the disk attached and this script as the app
gem5.opt -d m5out-app [gem5 options] \
    src/gpu-fs/utils/pkg_disk_config.py <checkpoint-dir>.gem5.img \
    configs/example/gem5_library/x86-mi300x-gpu.py [config options] \
    --image <guest disk image> --kernel <guest kernel> \
    --ipt-binary <IP-discovery binary> \
    --app src/gpu-fs/gpu_checkpoint.py --opts "restore --gem5"
# 3. PASS or FAIL
src/gpu-fs/utils/gem5_verdict.py m5out-app
```

Where each part happens:

| Step | Where | When |
|---|---|---|
| Package (`checkpoint --gem5`) | MI300X host | with the checkpoint |
| Package disk (`utils/make_pkg_disk.sh`) | gem5 host | once per checkpoint, before gem5 |
| Single-XCC IP-discovery binary (`utils/make_1xcc_ipt.py`) | gem5 host | once per gem5 setup; pass it with `--ipt-binary` when the manifest's `ipt` is `1xcc` (one CPX partition), the 8-XCC one for `8xcc` |
| Attaching the disk (`utils/pkg_disk_config.py`) | gem5 command line | every run, in front of the config |
| Overlaying the disk on `/` (overlayfs) | guest, `gpu_checkpoint.py restore --gem5` | every run, automatically, before criu |
| Restore, copying results out | guest, `gpu_checkpoint.py restore --gem5` | every run |
| PASS/FAIL (`utils/gem5_verdict.py`) | gem5 host | after the run |

`--overlay <dir>` (below) is a different thing: copying files into place for
a checkpoint installed in the image itself, not needed with a package disk.

1. `utils/make_pkg_disk.sh` turns the package into a small ext4 disk image
   (label `gem5pkg`). It holds the package's files except those identical to
   the guest image's (`BASE`, read only; unset keeps every file), criu and
   `amdgpu_plugin_gem5` (from `CRIU_TREE`, default `criu/` built by
   `build.sh`) and the manifest. It needs `fakeroot`,
   e2fsprogs and `sfdisk`; each checkpoint costs only its own disk, and the
   guest image is never written.
2. `utils/pkg_disk_config.py <disk> <config> [options]` runs any X86 board
   config unchanged and attaches the disk as the guest's second disk,
   copy-on-write like the root disk. The IP-discovery binary must match the
   manifest's `ipt` (printed by `make_pkg_disk.sh`): `1xcc` for a checkpoint
   of one CPX partition, made with
   `utils/make_1xcc_ipt.py <8-XCC binary> mi300x_1xcc_discovery.bin`.
3. gem5 copies `gpu_checkpoint.py` (`--app`) into the guest and runs it as
   root with `restore --gem5` (`--opts`). It mounts the disk (label
   `gem5pkg`) and lays its files
   over `/` with overlayfs, one mount per top-level directory: the package's
   files win over the image's at the same paths, so packages never
   conflict. It takes criu, the plugin and the checkpoint path from the
   disk, logs fatal signals to `dmesg`, restores, and copies out with
   `m5 writefile` the app's log (`app.log.after-restore`), `restore.log`,
   every file the app created or changed in the checkpoint directory (e.g. a
   result file), the tail of `dmesg` and a listing of the directory.
4. `utils/gem5_verdict.py <outdir>` prints `PASS` or `FAIL`: criu restored,
   the app's log came out, the GPU was active (`shaderActiveTicks` in
   `stats.txt`) and `dmesg` shows no crash or fault. Whether the app's own
   result is right is for you to check in its log (its tail is printed).

`restore --gem5 [DIR]` also restores a checkpoint already installed in the
guest image, without a package disk (criu from `CRIU` and `PLUGIN`), with
`--overlay <dir>` (or `OVERLAY`) for files that differ from the image's:
`<dir>/<path>` is copied to `/<path>` first.

- **Files.** The guest needs the checkpoint and every file the app had
  mapped at the same paths, byte for byte: binary, inputs it still has open,
  ROCm libraries, `libklhook.so`, with their owners and modes (the restored
  app runs with the checkpoint's uid and gid; it must be able to write in its
  cwd, for example a result file). The package carries exactly these.
- **GPU shape.** CRIU restores only onto a GPU with the same SIMD, shader
  array and SDMA counts. A CPX checkpoint holds one XCC (38 CUs, 4 arrays,
  2 SDMA engines), so the guest must see a single-XCC MI300X:
  ```sh
  src/gpu-fs/utils/make_1xcc_ipt.py <8-XCC ipt binary> mi300x_1xcc_discovery.bin
  ```
  and give it to gem5 with `--ipt-binary`.
- **Property checks.** The restore skips the checks where gem5's MI300X
  differs from the real one: MEC firmware version (`KFD_FW_VER_CHECK=0`; the
  host's may be newer than gem5's), SDMA firmware version, RAS capability
  bits, VRAM size and cache count.
- **No large BAR.** `gpu_checkpoint.py` (checkpoint and profile) runs the app with
  `ROC_ENABLE_LARGE_BAR=0`, so HIP reports `isLargeBar = 0` and nothing reads
  VRAM directly with the CPU. In gem5 the VRAM BAR is MMIO, and KVM cannot
  emulate SSE loads from it (`movss`): PyTorch's `tensor.item()` (ROCm 7.2+
  reads the scalar straight from VRAM on large-BAR GPUs) killed a restored
  app with SIGILL. Checkpoints taken without it have the setting baked in.
- **Exit status.** `criu restore` returns 0 once the restore succeeded,
  whatever the app does afterwards; hence the `PASS`/`FAIL` checks above.
