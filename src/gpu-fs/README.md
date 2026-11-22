# GPU checkpoints with CRIU

Checkpoint a HIP application on a real AMD MI300X with
[CRIU](https://criu.org) and its amdgpu plugin, then restore it natively or
inside a gem5 GPU full-system guest. With [klHook](klHook), the checkpoint is
taken right before a chosen kernel launch. The application needs no changes.

| Path | What it is |
|---|---|
| `criu/` | Submodule: upstream CRIU, pinned to `v4.0` |
| `criu-patches/` | Patches applied to `criu/` by `build.sh`, in name order |
| `klHook/` | Submodule: LD_PRELOAD hook for HIP launches; pauses before the selected kernel |
| `build.sh` | Applies the patches and builds criu, the amdgpu plugin and `libklhook.so` |
| `kl-ckpt.sh` | Runs an app and checkpoints it before the selected kernel launch |
| `ckpt.sh` | Runs an app and checkpoints it once it sits in `sleep()` (no klHook) |
| `restore.sh` | Restores a checkpoint, on the host or in the gem5 guest |
| `run_gem5_restore.sh` | Boots the gem5 guest and runs `restore.sh` in it |

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
KL_CKPT_KERNEL=vector_square:2 src/gpu-fs/kl-ckpt.sh [--continue] <ckpt-dir> <app> [args...]
src/gpu-fs/restore.sh <ckpt-dir>
```

- `KL_CKPT_KERNEL=<kernel>[:<n>]` selects the n-th launch (default 1st) of the
  kernel, matched by its base name or its mangled name. `KL_HOOK_VERBOSE=1`
  prints every launched kernel's name.
- At that launch klHook waits for the GPU to finish all earlier work (via
  `hipDeviceSynchronize`), then hands the process to `kl-ckpt.sh`, which dumps
  it. By default the dump kills the app. With `--continue` the app finishes
  natively, and the script returns the app's exit status.
- After a restore, klHook notices on its own that it was restored and launches
  the kernel.
- The app's output goes to `<ckpt-dir>/app.log`. The checkpoint must stay at
  the same absolute path (it holds the app's cwd and its open `app.log`), and
  so must every file the app had mapped (binary, ROCm libraries,
  `libklhook.so`).
- Environment variables: `GPU` (PCI address, default `0000:f6:00.0`), `PART`
  (CPX partition 0-7, default 0), `CRIU`, `PLUGIN`, `KLHOOK`.

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

### gem5

`restore.sh` restores directly when `amd-smi` is not present, as in the gem5
guest, and copies `app.log` and `restore.log` out with `m5 writefile`.
`run_gem5_restore.sh <outdir> <ckpt-dir-in-guest>` boots the guest and runs
it. The guest image needs the checkpoint and every file the app had mapped at
the same paths.
