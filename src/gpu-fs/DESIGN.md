# gpu-fs: how it works

Background for [README.md](README.md), which has the steps to run. Nothing
here is needed to use the tools.

## Checkpointing at a kernel launch

klHook is an `LD_PRELOAD` library that wraps HIP's launch functions
(`hipLaunchKernel`, `hipModuleLaunchKernel`, `hipExtModuleLaunchKernel`,
`hipLaunchKernelExC`, `hipLaunchCooperativeKernel`, `hipExtLaunchKernel`).
`gpu_checkpoint.py checkpoint` starts the app with `KL_CKPT_KERNEL=NAME[:N]`,
`KL_CKPT_READY` and `KL_CKPT_GO` set. At the N-th launch of `NAME`, klHook:

1. calls `hipDeviceSynchronize()`, so the GPU has no work in flight;
2. creates the `KL_CKPT_READY` file and waits;
3. the script runs `criu dump` with the stock amdgpu plugin, which saves the
   process and its GPU state (buffers, queues, events);
4. without `--continue` the dump kills the app; with it, the script creates
   `KL_CKPT_GO` and the app carries on natively.

After a restore klHook sees that its process was restored (a restored
process is a new process, so its start time in `/proc/self/stat` changed),
prints `kl-hook: restored` and makes the launch it
was stopped at. Kernels are matched by base name (the demangled name without
template arguments or parameters) or by mangled name; `KL_HOOK_COUNTS`
makes klHook write each kernel's launch count at exit (`profile`).

klHook loads the HIP and HSA runtimes it hooks with `dlopen` at startup, by
path: `/opt/rocm/lib` unless `KL_HOOK_HIP_LIB_PATH`/`KL_HOOK_HSA_LIB_PATH`
say otherwise. It must be the same runtime the app uses; a second one sees
no GPU state of the app. PyTorch's pip wheels bring their own ROCm in
`torch/lib`, which `gpu_checkpoint.py` finds on the Python interpreter's
`sys.path` (without importing torch).

## Why CPX

With amdgpu-dkms 7.1.9 the kernel rejects the restore of any queue on a GPU
with more than one XCC (`amdgpu: Failed to restore queue (-22)`).
`kfd_criu_restore_queue()` compares the saved MQD size (per-XCC size × XCCs)
with the size for a single XCC. An MI300X in SPX mode has 8 XCCs. The script
therefore switches the GPU to CPX (8 partitions of 1 XCC each, 38 CUs) for
the run and back to SPX afterwards. The GPU must be idle for that switch.

The app and criu run in a device cgroup that allows only `/dev/kfd` and one
partition's render node. KFD filters its topology by the caller's device
cgroup, so both see a single GPU, and the checkpoint holds exactly one GPU.

Consequences:
- Library kernels can differ from those on the full GPU: hipBLASLt picks
  GEMM solutions for 38 CUs. Profile on CPX (`profile` does).
- hipBLASLt's solution search for each new GEMM shape is much slower on CPX.
- The gem5 guest must show a single-XCC GPU (below).

## Host environment for the app

`checkpoint` and `profile` run the app as the calling user, in the current
directory, in a systemd service (in the device cgroup), with:

| Variable | Why |
|---|---|
| `LD_PRELOAD=libklhook.so` | the hook |
| `HSA_ENABLE_INTERRUPT=0` | gem5 sends no completion interrupts; the restored runtime must poll |
| `ROC_ENABLE_LARGE_BAR=0` | HIP reports `isLargeBar = 0`, so neither HIP nor the app reads VRAM with the CPU. In gem5 the VRAM BAR is MMIO, KVM cannot emulate SSE loads from it (`movss`), and PyTorch's `tensor.item()` (ROCm 7.2+) died with SIGILL. Fixed at app start, so it is in the checkpoint |
| `GLIBC_TUNABLES=glibc.cpu.hwcaps=...` | glibc picks its string/memory functions by CPU features at startup, and the choice is in the checkpoint; gem5's CPU is x86-64-v2, so newer ISA extensions are hidden from that choice |
| `KL_HOOK_*` | passed through from the caller |

The app's stdout and stderr go to `DIR/app.log`, a file the app keeps open.
CRIU records its size and write offset and refuses to restore if the size
changed, so the script saves `app.log.at-dump` and puts it back before and
after every restore. That is why a restored run's log starts with the
output from before the checkpoint.

## gem5 packages

CRIU reopens, by path, every file the process tree had mapped or open, and
checks its size and build-ID; the guest must have them byte for byte.
`checkpoint --gem5` lists them right before the dump (from
`/proc/<pid>/maps`, `/proc/<pid>/fd`, `exe` and `cwd` of every process in
the tree) and writes `DIR.gem5.tar.gz`:

| Entry | Content |
|---|---|
| `<name>/rootfs/<abs path>` | those files, the whole checkpoint directory and the app's working directory, with owners and modes |
| `<name>/SHA256SUMS` | over `rootfs/` |
| `<name>/manifest.json` | checkpoint path, cwd, command, kernel; GPU shape from KFD (`num_xcc`, `simd_count`, `array_count`, `cu_per_simd_array`, SDMA counts); `ipt`: `1xcc` or `8xcc` |

The restored app runs with the checkpoint's uid and gid, so the owners
matter: the app must be able to write in its working directory (BC writes
`result.txt` there).

`utils/make_pkg_disk.sh` turns a package into an ext4 image with label
`gem5pkg`: `/rootfs/` (minus files identical to the guest image's, given as
`BASE`), `/criu/` (criu and `amdgpu_plugin_gem5`) and `/manifest.json`.
`utils/pkg_disk_config.py` wraps any X86 board config: it patches
`X86Board` so that the package disk is added next to the root disk,
copy-on-write, and then runs the config unchanged.

In the guest, `gpu_checkpoint.py restore --gem5`:
1. mounts the disk and lays `/rootfs/<top>` over `/<top>` with overlayfs,
   one mount per top-level directory, so the package's files win over the
   image's at the same paths;
2. sets `/proc/sys/kernel/print-fatal-signals`, since `criu restore` exits 0
   once the restore succeeded, whatever the app does afterwards;
3. restores with criu and `amdgpu_plugin_gem5` from the disk, with the
   property checks where gem5's MI300X differs from the real one turned off:
   `KFD_FW_VER_CHECK=0` (the host's MEC firmware may be newer than gem5's),
   `KFD_SDMA_FW_VER_CHECK=0`, `KFD_CAPABILITY_CHECK=0` (RAS bits),
   `KFD_VRAM_SIZE_CHECK=0`, `KFD_CACHES_COUNT_CHECK=0`;
4. waits for the app and copies out with `m5 writefile`: `app.log.after-restore`,
   `restore.log`, every file created or changed in the checkpoint directory,
   `dmesg-after-restore.txt` and `ls-after-restore.txt`.

### GPU shape

CRIU's amdgpu plugin restores only onto a GPU with the same `simd_count`,
`array_count`, `cu_per_simd_array` and SDMA engine counts (the `KFD_*_CHECK`
variables do not cover these). A CPX checkpoint has 1 XCC: 38 CUs (152
SIMDs), 4 shader arrays, 2 SDMA engines, no XGMI SDMA engines. The guest
driver takes the XCC and SDMA counts from the IP-discovery table gem5 hands
it (`--ipt-binary`). `utils/make_1xcc_ipt.py` keeps GC instance 0 and SDMA
instances 0 and 1 of the 8-XCC table and drops the other GC and SDMA
entries, recomputing sizes and checksums.

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

## build.sh

- Applies each patch series when its submodule is clean, reports "already
  applied" when the last patch reverse-applies, and stops when the
  submodule has other local changes.
- Builds criu and both plugins; builds klHook only when the installed HIP
  has `hipLaunchKernelExC` (ROCm 7). klHook has no device code, so it is
  compiled host-only, with the newest GCC whose C++ headers are installed.
- `ROCM_PATH` sets the ROCm directory (default `/opt/rocm`).
