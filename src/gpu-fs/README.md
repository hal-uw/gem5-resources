# GPU checkpoints with CRIU, restored in gem5

Checkpoint a HIP or PyTorch application on a real AMD MI300X right before a
chosen kernel launch, then restore it natively or in gem5's GPU full-system
model, so gem5 simulates only the kernels you care about. The app needs no
changes. How it works: [DESIGN.md](DESIGN.md).

Two machines are involved; they can be the same one:
- **MI300X host:** builds the tools, checkpoints the app, packages it.
- **gem5 host:** builds gem5 and the guest image, restores the package in gem5.

Each step below has a command, what it should print, how long it takes,
and what to read if it fails. Commands run from the gem5-resources root.
Every script, `gpu_checkpoint.py` included, reads its settings from
`src/gpu-fs/env.sh` itself, so no shell state is needed between steps; a
variable set in the environment overrides `env.sh`. In commands that `cd`
elsewhere, `$R` stands for the gem5-resources root (`R=$PWD` before the `cd`).

## Rules that save hours

- **Do not upgrade packages on the MI300X host while you have checkpoints**
  (`apt upgrade`, unattended-upgrades). A restore needs every file the app
  had mapped (glibc, ROCm, Python, torch) to be byte-identical to the one at
  checkpoint time.
- **Do not move or rename a checkpoint directory.** It must stay at the
  absolute path where it was taken, and so must the app's binary, inputs
  and working directory.
- **Use an idle GPU.** The scripts switch it to CPX mode and back; that fails
  if anything else has it open, and on a shared machine it would disrupt the
  GPU's user. Set `GPU` in `env.sh` to one `preflight.sh host` shows as idle.
- **Never put `amdgpu_plugin.so` and `amdgpu_plugin_gem5.so` in one
  directory.** CRIU loads every plugin in its plugin directory.
- **The host and the guest image should have the same ROCm and PyTorch, at
  the same paths.** Then the package disk is small (step 8).

## Quick start (smoke test)

On the MI300X host, after [step 1](#1-build-the-tools-mi300x-host) and
[step 2](#2-configure-and-check-mi300x-host) (square does not need the
PyTorch install):
```sh
src/gpu-fs/examples/host.sh square
```
Expected: last line `PASS square all (gem5 package: <CK>/square.gem5.tar.gz)`,
in about 6 s. On the gem5 host, after [steps 6–8](#6-build-gem5-gem5-host),
with that package copied to `CK/square.gem5.tar.gz` there (the gem5 host's
`CK`; any path works if you pass it as a second argument):
```sh
src/gpu-fs/examples/gem5.sh square
```
Expected: last line `PASS square gem5 restore (<CK>/m5out-square)`.

## Examples

`examples/host.sh APP` and `examples/gem5.sh APP` run every step for these:

| APP | What | Kernel launches | Checkpoint before | Kernels after restore | Success line |
|---|---|---|---|---|---|
| `square` | `src/gpu/square` | 51 | `vector_square:50` | 2 | `PASSED!` |
| `bc` | Pannotia BC (`src/gpu/pannotia/bc`), input `1k_128k.gr` (downloaded) | 1051 | `backtrack_kernel:448` | 3 | `Completed iteration 149` |
| `mnist` | PyTorch FashionMNIST, one batch: `test_1batch/pytorch_qs_mnist.py` from [modori-yh/Gem5](https://github.com/modori-yh/Gem5/tree/main/materials/04-GPU-model/pytorch/MNIST) (downloaded at a pinned commit; dataset downloaded) | 11 | `nll_loss_forward_reduce_cuda_kernel_2d` | 5 | `Done!` |

Measured on the MI300X host (`examples/host.sh APP`, all steps):

| APP | Time | Checkpoint | Package |
|---|---|---|---|
| `square` | 6 s | 258 MB | 85 MB |
| `bc` | 6 s | 275 MB | 84 MB |
| `mnist` | 1 min 35 s | 2.0 GB | 3.6 GB (the package disk leaves out torch if the guest image matches) |

To add an app, copy `examples/apps/square.sh` and set its kernel, launch
count and success line.

---

## MI300X host

### 1. Build the tools (MI300X host)

Requirements:
- an MI300X with the amdgpu driver, ROCm 7 in `/opt/rocm` (klHook needs
  ROCm 7), Ubuntu 24.04 ([Tested with](#tested-with));
- sudo without a password prompt (the scripts call it for the partition
  switch and criu), `systemd-run`, `amd-smi`, `curl`;
- `python3` on `PATH` (3.8 or newer, any installation, e.g. conda) runs the
  tools; the app's own Python is separate (`PYTHON`, step 2);
- network access: submodules, and the bc and mnist inputs.

```sh
git submodule update --init src/gpu-fs/criu src/gpu-fs/klHook
sudo apt install libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
    python3-protobuf libnet1-dev libcap-dev libbsd-dev libgnutls28-dev libnftables-dev \
    uuid-dev libnl-3-dev libdrm-dev pkg-config
src/gpu-fs/build.sh
```
- **Expect:** it ends without an error, and `src/gpu-fs/criu/criu/criu`,
  `src/gpu-fs/criu/plugins/amdgpu/gem5/amdgpu_plugin_gem5.so` and
  `src/gpu-fs/klHook/libklhook.so` exist.
- **Time:** seconds to 2 min.
- **Notes:** "Warn: Externally managed python environment ...
  PIP_BREAK_SYSTEM_PACKAGES" during the criu build is harmless; install
  nothing for it. `build.sh` applies `criu-patches/` and `klhook-patches/` to the
  submodules and can be rerun at any time. `ROCM_PATH` overrides
  `/opt/rocm`. On a host without ROCm 7 it builds criu and the plugins only.
- **If it fails:** "has local changes that are not ...patch": the submodule
  was edited; `git -C src/gpu-fs/<sub> stash` or reset it.

### 2. Configure and check (MI300X host)

```sh
cp src/gpu-fs/env.example.sh src/gpu-fs/env.sh
src/gpu-fs/utils/preflight.sh host     # lists the GPUs: pick an idle one
# edit src/gpu-fs/env.sh: GPU (and CK, PYTHON if not the defaults), then again:
src/gpu-fs/utils/preflight.sh host
```
- **Expect:** `PASS` or `WARN` on every line and `preflight host: ready`.
  Under `MI300X GPUs` it prints each GPU's PCI address, partition mode and
  `idle` or the pids using it; `GPU` should be `idle` and `SPX`.
- **If a line says FAIL:** its text says what to do. The scripts create `CK`.

Settings in `env.sh` for this host:

| Variable | Default | Meaning |
|---|---|---|
| `GPU` | first GPU | PCI address of an idle MI300X |
| `CK` | `$HOME/gpu-ckpt` | where examples build apps and write checkpoints and packages |
| `PYTHON` | `/usr/bin/python3` | interpreter for the PyTorch example |

For PyTorch (the `mnist` example), install the pinned build for the system
Python, the same one the guest image gets (step 7):
```sh
sudo /usr/bin/python3 -m pip install --break-system-packages -r src/gpu-fs/examples/mnist/requirements.txt
```
It needs about 10 GB on `/`.

### 3. Profile the app (MI300X host)

Run the app once under klHook to see its kernels:
```sh
src/gpu-fs/examples/host.sh APP profile                  # an example
R=$PWD; cd <app's working dir> && $R/src/gpu-fs/gpu_checkpoint.py profile <app> [args...]   # any app
```
- **Expect:** the app's output, then the launch counts, most launched first:
  ```
  51 kernel launches, 1 kernels (app exited 0):
        51  vector_square
  ```
  `host.sh` also prints `profile OK: ...` and saves this to `CK/APP.profile.txt`.
  These counts are klHook's and include every launch. An app's own kernel
  timer may count fewer: square's skips its untimed warm-up launch (50 vs 51).
- **Launch order:** set `KL_HOOK_VERBOSE=1` and klHook prints a
  `kl-hook: kernel <name>` line for each launch. Library kernel names
  (hipBLASLt) can be hundreds of characters long; `--kernel` takes them whole.
- **If it fails:** see [FAQ](#faq): "No CUDA GPUs are available", "the
  partition switch fails".

Choosing the checkpoint kernel (`--kernel NAME[:N]` = right before the N-th
launch of `NAME`, matched by base or mangled name):
- To run the last K launches after the restore, stop at launch
  total − K + 1. Find its name in the `KL_HOOK_VERBOSE=1` output and count
  which occurrence of that name it is.
- Profile the way you checkpoint. `profile` runs on CPX, where library
  kernels (hipBLASLt GEMMs) differ from the full GPU's.
- PyTorch: stop after the last new GEMM shape. hipBLASLt searches for a
  solution on the CPU the first time it sees a shape; before that point gem5
  would simulate the search.
- gem5 is slow: a few kernels are plenty.

### 4. Checkpoint and restore natively (MI300X host)

```sh
src/gpu-fs/examples/host.sh APP                     # every step: prepare, profile, checkpoint, restore
src/gpu-fs/examples/host.sh APP checkpoint          # one step
src/gpu-fs/examples/host.sh APP restore
```
Any app:
```sh
R=$PWD; cd <app's working dir>
$R/src/gpu-fs/gpu_checkpoint.py checkpoint --gem5 --kernel NAME[:N] DIR <app> [args...]
$R/src/gpu-fs/gpu_checkpoint.py restore DIR
```
- **Expect (checkpoint):** `checkpoint of <app> before NAME[:N] (pid ...)`,
  then `gem5 package: DIR.gem5.tar.gz (N files, M MiB)`.
- **Expect (restore):** `criu restore exit 0`, then the app's whole log:
  the output from before the checkpoint, `kl-hook: restored`, and the
  output of the rest of the run. That order holds for C/HIP apps; Python
  buffers its output, so for mnist the `kl-hook:` lines come first and all
  of the app's output after them. `host.sh` checks for the success line and
  saves the output to `CK/APP.restore.txt`. A timer the app started before
  the checkpoint shows a meaningless time ([FAQ](#faq)).
- **Options:**
  - `--gem5` also writes `DIR.gem5.tar.gz`, the package for gem5 (step 9).
  - Without `--continue` the dump kills the app. With it, the app finishes
    natively and the script returns its exit status.
  - `DIR` must not exist; the script creates it. `host.sh` refuses to
    overwrite unless `FORCE=1`.
  - A restore does not change the checkpoint; restore as often as you like.
  - The app's output goes to `DIR/app.log`.
- **If it fails:** "app exited before NAME was launched": wrong kernel or
  count (step 3). Restore errors: `DIR/restore.log`, and
  [FAQ](#faq): "bad build-ID".

Other settings `gpu_checkpoint.py` reads: `PART` (CPX partition, 0–7,
default 0), `CRIU`, `PLUGIN`, `KLHOOK` (built paths by default),
`KL_HOOK_VERBOSE=1`, and any `KL_HOOK_*`, passed on to klHook.
`KL_HOOK_HIP_LIB_PATH`/`KL_HOOK_HSA_LIB_PATH` point klHook at the app's HIP
runtime; for PyTorch wheels the script finds `torch/lib` itself.

### 5. Copy the package to the gem5 host

Copy `CK/APP.gem5.tar.gz` (or `DIR.gem5.tar.gz`) to the gem5 host, to any
path: the package carries its own absolute paths. `examples/gem5.sh APP`
looks for it at the gem5 host's `CK/APP.gem5.tar.gz` unless you pass the
path. Keep `DIR` on the MI300X host for native restores.

---

## gem5 host

### 6. Build gem5 (gem5 host)

```sh
git submodule update --init src/gpu-fs/criu src/gpu-fs/klHook
sudo apt install <the packages from step 1> fakeroot e2fsprogs fdisk
src/gpu-fs/build.sh                 # criu and amdgpu_plugin_gem5 for the package disk
git clone https://github.com/gem5/gem5 <gem5>      # upstream gem5, outside this repo
cd <gem5> && scons build/VEGA_X86/gem5.opt -j$(nproc)
```
Use a gem5 version that has the MI300X full-system board config,
`configs/example/gem5_library/x86-mi300x-gpu.py`; `preflight.sh gem5`
checks for it.
- **Expect:** `src/gpu-fs/criu/criu/criu`,
  `src/gpu-fs/criu/plugins/amdgpu/gem5/amdgpu_plugin_gem5.so` and
  `<gem5>/build/VEGA_X86/gem5.opt` exist. Without ROCm 7, `build.sh` says
  it skips klHook, which is fine here.
- **Time:** gem5 takes 30–60 min.

### 7. Build the guest disk image (gem5 host)

```sh
src/gpu-fs/guest-image/build.sh WORKDIR       # outside the repo, ~100 GB free
```
It runs the stock [`src/x86-ubuntu-gpu-ml`](../x86-ubuntu-gpu-ml) packer
build with [`guest-image/rocm-install.sh`](guest-image/rocm-install.sh),
which installs:
- the MI300X host's ROCm (7.2.4) and amdgpu driver repo (31.60);
- the runtime libraries of criu and `amdgpu_plugin_gem5`;
- PyTorch pinned to `examples/mnist/requirements.txt`, for the system
  Python, as on the host.

- **Expect:** it ends by printing the `GUEST_IMG`, `GUEST_KERNEL` and `IPT8`
  lines for `env.sh`.
- **Time:** about 1 h. Needs `qemu-system-x86_64`, `unzip`, network, about
  100 GB in `WORKDIR`, and KVM: `/dev/kvm` usable by your user (group `kvm`;
  log in again after adding yourself). gem5 needs KVM too.
- **Keep the image and kernel together:** the image's driver is built for
  that kernel.
- **If your MI300X host differs:** change `ROCM_VERSION`, `AMDGPU_REPO` and
  the PyTorch pins in `rocm-install.sh` (and `requirements.txt`) to match
  what `preflight.sh host` prints there (ROCm, and torch/torchvision for
  `PYTHON`) before building. `build.sh` refuses if the pins disagree.

### 8. Configure and check (gem5 host)

Set these in `src/gpu-fs/env.sh`, then check:

| Variable | Meaning |
|---|---|
| `GEM5_DIR` | gem5 tree; or set `GEM5` (binary) and `GEM5_CONFIG` (default `configs/example/gem5_library/x86-mi300x-gpu.py`) |
| `GUEST_IMG`, `GUEST_KERNEL` | from step 7 |
| `IPT8` | 8-XCC MI300X IP-discovery binary (step 7 prints it) |
| `CK` | where packages, package disks and gem5 output go |
| `GEM5_OPTS`, `CONFIG_OPTS` | extra options for gem5 and for the board config |

```sh
src/gpu-fs/utils/preflight.sh gem5
```
- **Expect:** `preflight gem5: ready`.

### 9. Restore in gem5 (gem5 host)

```sh
src/gpu-fs/examples/gem5.sh APP [PACKAGE]          # PACKAGE default: CK/APP.gem5.tar.gz
```
It runs these steps, each skipped if already done:
1. `utils/make_1xcc_ipt.py IPT8 IPT1`: the single-XCC IP-discovery binary,
   since a CPX checkpoint holds a 1-XCC GPU (the manifest's `ipt` is `1xcc`).
2. `BASE=GUEST_IMG utils/make_pkg_disk.sh PACKAGE`: the package disk, with
   the package's files minus those the image already has.
3. gem5, output in `CK/m5out-APP`:
   ```sh
   gem5.opt -d CK/m5out-APP utils/pkg_disk_config.py PACKAGE.img GEM5_CONFIG \
       --image GUEST_IMG --kernel GUEST_KERNEL --ipt-binary IPT1 \
       --app src/gpu-fs/gpu_checkpoint.py --opts "restore --gem5"
   ```
   In the guest, `gpu_checkpoint.py` lays the package disk over `/`,
   restores with criu, and copies results out.
4. `utils/gem5_verdict.py CK/m5out-APP`, then a check for the app's success
   line.

- **Expect:** `PASS: criu restore exit 0, ...`, the tail of the app's log,
  and the last line `PASS APP gem5 restore (...)`.
- **Results in `CK/m5out-APP`:**
  - `app.log.after-restore`: the app's output;
  - `restore.log`: criu's log;
  - `stats.txt`: gem5's statistics, including kernel timings;
  - `dmesg-after-restore.txt`;
  - `gem5.log`: gem5's own output;
  - every file the app created or changed in its checkpoint directory.
- **Time:** not measured yet. Expect minutes for `square` and hours for
  `mnist`, where Python's shutdown also runs on the simulated CPU.
- **If it fails:**
  - `restore.log` for criu errors;
  - `dmesg-after-restore.txt` for crashes and GPU faults;
  - [FAQ](#faq): "topology", "SIGILL".

---

## FAQ

Each entry: symptom → cause → fix.

**`RuntimeError: No CUDA GPUs are available` (or any app failing only under klHook)**
→ klHook hooks a different HIP runtime than the app's, usually `/opt/rocm`'s
next to the one bundled in a PyTorch wheel.
→ The script finds torch's for Python apps. For any other app that brings
its own ROCm, set `KL_HOOK_HIP_LIB_PATH` and `KL_HOOK_HSA_LIB_PATH` to its
`libamdhip64.so` and `libhsa-runtime64.so`.

**`gpu_checkpoint.py: app exited before NAME was launched`**
→ The kernel name or count does not match this run.
→ Run `profile` (step 3). Library kernels differ between CPX and the full
GPU, and between library versions.

**The partition switch fails, or `amd-smi set ... --compute-partition` errors**
→ Another process has the GPU open.
→ Choose an idle GPU (`GPU=` in `env.sh`; `preflight.sh host` lists them).

**`amdgpu: Failed to restore queue (-22)` in `dmesg`**
→ A restore on a GPU with more than one XCC (SPX mode); amdgpu-dkms 7.1.9
rejects it.
→ Use `gpu_checkpoint.py`, which runs on one CPX partition. See
[DESIGN.md](DESIGN.md#why-cpx).

**A native restore fails with `bad build-ID`, `Can't open locale-archive` or a size mismatch**
→ A file the app had mapped changed after the checkpoint, usually a package
upgrade (glibc, locales, ROCm).
→ Check `grep -E ' (upgrade|install) ' /var/log/dpkg.log | tail`, put the old
versions back, or take the checkpoint again.

**PyTorch is very slow under `profile`/`checkpoint`**
→ hipBLASLt searches for a GEMM solution on the CPU for each new GEMM
shape, and on a 38-CU CPX partition that is much slower (nanoGPT: ~25 s per
iteration).
→ Expected. Checkpoint after the last new shape, so the restored app does
no search.

**The restored run's log shows output from before the checkpoint**
→ The app's log is a file it had open at the dump; the script puts it back
as it was so CRIU can reopen it, and the restored app appends.
→ Expected. Everything after `kl-hook: restored` is from the restored run.

**The app reports a huge or negative time**
→ Its timer started before the checkpoint and stopped after the restore;
the GPU clock base differs.
→ Ignore that timer. Kernels that ran wholly after the restore are timed
correctly, and gem5's `stats.txt` has its own timings.

**Python output is out of order relative to `kl-hook:` lines**
→ Python block-buffers stdout when writing to a file.
→ Run Python with `-u` (or `PYTHONUNBUFFERED=1`) if the order matters.

**`criu restore exit 0`, but the app failed**
→ criu returns once the restore succeeded, whatever the app does afterwards.
→ Check the app's output. In gem5, `gem5_verdict.py` also checks `dmesg`
for crashes and GPU faults.

**gem5: the restore fails on the GPU topology (simd_count, array_count, SDMA)**
→ The guest's GPU shape differs from the checkpoint's.
→ Pass the single-XCC IP-discovery binary (`make_1xcc_ipt.py`) for a `1xcc`
package. `examples/gem5.sh` does this.

**gem5: the restored app dies with SIGILL**
→ It read GPU memory directly with the CPU (large BAR), e.g. PyTorch's
`tensor.item()`. KVM cannot emulate SSE loads from gem5's VRAM BAR.
→ `gpu_checkpoint.py` runs apps with `ROC_ENABLE_LARGE_BAR=0`. Retake
checkpoints made without it.

**gem5: the restored app cannot write its result file**
→ The restored app runs with the checkpoint's uid and gid; its working
directory must be writable by that user.
→ Keep the owners the package records (make_pkg_disk.sh does); don't
`chown` in the image.

**The package or package disk is huge**
→ It holds every file the app had mapped, e.g. all of torch.
→ Make the host and the guest image match (step 2, step 7). The package disk
then leaves out every file the image already has. The `.gem5.tar.gz`
itself always holds everything.

**gem5 keeps running after the last kernel**
→ The app's remaining CPU work (printing, Python and library shutdown) runs
on gem5's CPU model.
→ Expected; with PyTorch it can take longer than the kernels.

## Tested with

| | MI300X host | gem5 guest image |
|---|---|---|
| OS | Ubuntu 24.04.5, kernel 6.8.0-147 | Ubuntu 24.04.2, kernel 6.8.0-124 |
| Driver | amdgpu-dkms 1:7.1.9.31600000 (repo `amdgpu/31.60`) | same repo |
| ROCm | 7.2.4 | 7.2.4 |
| glibc | 2.39-0ubuntu8.9 | |
| PyTorch | 2.12.1+rocm7.2, torchvision 0.27.1+rocm7.2 | same |
| CRIU | v4.0 + `criu-patches/` | |
| gem5 | | not recorded yet |

## Files

| Path | What it is |
|---|---|
| `gpu_checkpoint.py` | `profile`, `checkpoint`, `restore` (host) and `restore --gem5` (guest) |
| `build.sh` | applies the patches; builds criu, both amdgpu plugins and klHook |
| `env.example.sh` | settings template; copy to `env.sh` |
| `examples/host.sh`, `examples/gem5.sh` | every step for an example app, with checks |
| `examples/apps/*.sh` | example definitions: build, command, kernel, success line |
| `examples/mnist/requirements.txt` | the PyTorch build for the host and the guest |
| `guest-image/` | `build.sh` and `rocm-install.sh` for the gem5 guest image |
| `utils/preflight.sh` | checks a host is ready (`host` or `gem5`) |
| `utils/make_pkg_disk.sh` | package → package disk |
| `utils/pkg_disk_config.py` | runs a gem5 board config with the package disk attached |
| `utils/make_1xcc_ipt.py` | single-XCC IP-discovery binary |
| `utils/gem5_verdict.py` | PASS/FAIL of a gem5 run |
| `criu/`, `criu-patches/` | CRIU v4.0 submodule and its patches |
| `klHook/`, `klhook-patches/` | klHook submodule and its patches |
