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
| `gpu_checkpoint.py` | `profile`: lists an app's kernels and launch counts; `checkpoint`: runs an app and checkpoints it; `restore`: restores it, on the host or in the gem5 guest |
| `run_gem5_restore.sh` | Boots the gem5 guest and runs `gpu_checkpoint.py restore` in it |
| `make_1xcc_ipt.py` | Makes the single-XCC IP-discovery binary the gem5 guest needs for a CPX checkpoint |

## Setup

```sh
git submodule update --init src/gpu-fs/criu src/gpu-fs/klHook
sudo apt install libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
    python3-protobuf libnet1-dev libcap-dev libbsd-dev libgnutls28-dev libnftables-dev \
    uuid-dev libnl-3-dev libdrm-dev pkg-config
src/gpu-fs/build.sh
```

ROCm comes from `/opt/rocm` (override it with `ROCM_PATH`).

## Checkpoint and restore

```sh
src/gpu-fs/gpu_checkpoint.py profile <app> [args...]
src/gpu-fs/gpu_checkpoint.py checkpoint --kernel <kernel>[:<n>] [--continue] <checkpoint-dir> <app> [args...]
src/gpu-fs/gpu_checkpoint.py restore <checkpoint-dir>
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
- `restore` waits for the restored app to exit and prints its output.
- After a restore, klHook notices on its own that it was restored and launches
  the kernel.
- The app runs in the current directory. Its output goes to `<checkpoint-dir>/app.log`. The checkpoint must stay at
  the same absolute path (it holds the app's cwd and its open `app.log`), and
  so must every file the app had mapped (binary, ROCm libraries,
  `libklhook.so`).
- Environment variables: `GPU` (PCI address, default `0000:f6:00.0`), `PART`
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
from the stock plugin in three ways:

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

`gpu_checkpoint.py restore` restores directly when `amd-smi` is not present, as in
the gem5 guest, with `amdgpu_plugin_gem5`, and copies `app.log` and
`restore.log` out with `m5 writefile`.
`run_gem5_restore.sh <outdir> <checkpoint-dir-in-guest>` boots the guest and
runs it there (the guest needs `python3`).

- **Files.** The guest image needs the checkpoint and every file the app had
  mapped at the same paths, byte for byte: binary, inputs it still has open,
  ROCm libraries, `libklhook.so`. Keep their owners and modes (the restored
  app runs with the checkpoint's uid and gid; it must be able to write in its
  cwd, for example a result file).
- **GPU shape.** CRIU restores only onto a GPU with the same SIMD, shader
  array and SDMA counts. A CPX checkpoint holds one XCC (38 CUs, 4 arrays,
  2 SDMA engines), so the guest must see a single-XCC MI300X:
  ```sh
  src/gpu-fs/make_1xcc_ipt.py <8-XCC ipt binary> mi300x_1xcc_discovery.bin
  ```
  `run_gem5_restore.sh` passes it to gem5 with `--ipt-binary`; set `IPT` to
  use another binary.
- **Property checks.** The restore skips the checks where gem5's MI300X
  differs from the real one: MEC firmware version (`KFD_FW_VER_CHECK=0`; the
  host's may be newer than gem5's), SDMA firmware version, RAS capability
  bits, VRAM size and cache count.
- **Exit status.** `criu restore` returns 0 once the restore succeeded,
  whatever the app does afterwards. Check the app's output, and `dmesg` in
  the guest for a crash.
