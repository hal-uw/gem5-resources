#!/bin/bash
# Check that a machine is ready for the gpu-fs flow. Prints one line per check:
#   PASS <check>: <detail>   ready
#   WARN <check>: <detail>   works, but read the detail
#   FAIL <check>: <detail>   fix this first (the detail says how)
# and exits 1 if any check failed.
#   preflight.sh host    the MI300X host: checkpoint and native restore
#   preflight.sh gem5    the gem5 host: package disk and gem5 restore
# Reads src/gpu-fs/env.sh if it exists (see env.example.sh).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$HERE/env.sh" ] && . "$HERE/env.sh"
MODE=${1:-}
case $MODE in host|gem5) ;; *) echo "usage: $0 host|gem5" >&2; exit 2;; esac

fails=0
pass() { echo "PASS $1: $2"; }
warn() { echo "WARN $1: $2"; }
fail() { echo "FAIL $1: $2"; fails=$((fails + 1)); }
have() { command -v "$1" > /dev/null || [ -x "/usr/sbin/$1" ] || [ -x "/sbin/$1" ]; }
free_gb() { local d=$1; while [ ! -e "$d" ]; do d=$(dirname "$d"); done
            df -P --block-size=1G "$d" | awk 'NR==2 {print $4}'; }

CK=${CK:-$HOME/gpu-ckpt}

check_build() {  # $1: also klHook
    local f
    for f in criu/criu plugins/amdgpu/amdgpu_plugin.so plugins/amdgpu/gem5/amdgpu_plugin_gem5.so; do
        if [ -f "$HERE/criu/$f" ]; then pass "built criu/$f" "yes"
        else fail "built criu/$f" "missing: run src/gpu-fs/build.sh"; fi
    done
    if [ "$1" = klhook ]; then
        if [ -f "$HERE/klHook/libklhook.so" ]; then pass "built klHook/libklhook.so" "yes"
        else fail "built klHook/libklhook.so" "missing: run src/gpu-fs/build.sh (needs ROCm 7)"; fi
    fi
}

check_disk() {  # $1: GB wanted
    local gb; gb=$(free_gb "$CK")
    if [ "$gb" -ge "$1" ]; then pass "free disk at CK=$CK" "${gb} GB"
    else warn "free disk at CK=$CK" "${gb} GB; the examples need up to ~$1 GB (mnist: 2 GB checkpoint + 3.6 GB package)"; fi
}

if [ "$MODE" = host ]; then
    # GPUs in the KFD topology: MI300X is gfx942 (gfx_target_version 90402).
    gpus=()
    for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -f "$p" ] || continue
        simd=$(awk '$1=="simd_count"{print $2}' "$p")
        [ "${simd:-0}" -gt 0 ] || continue
        gfx=$(awk '$1=="gfx_target_version"{print $2}' "$p")
        loc=$(awk '$1=="location_id"{print $2}' "$p")
        dom=$(awk '$1=="domain"{print $2}' "$p")
        bdf=$(printf '%04x:%02x:%02x.%x' "$dom" $((loc >> 8)) $(((loc >> 3) & 31)) $((loc & 7)))
        gpus+=("$bdf gfx$gfx $(cat "$(dirname "$p")/gpu_id")")
    done
    mi300=$(printf '%s\n' "${gpus[@]}" | grep -c ' gfx90402 ' || true)
    if [ "$mi300" -gt 0 ]; then
        pass "MI300X GPUs" "$(printf '%s\n' "${gpus[@]}" | awk '{print $1}' | sort -u | wc -l) (PCI address, partition mode, users:)"
        # Per GPU: partition mode, and KFD processes that have its memory
        # mapped (ROCm opens every visible GPU, so any ROCm app counts).
        for bdf in $(printf '%s\n' "${gpus[@]}" | awk '{print $1}' | sort -u); do
            mode=$(cat "/sys/bus/pci/devices/$bdf/current_compute_partition" 2>/dev/null || echo "?")
            users=""
            for id in $(printf '%s\n' "${gpus[@]}" | awk -v b="$bdf" '$1==b {print $3}'); do
                for f in /sys/class/kfd/kfd/proc/*/vram_"$id"; do
                    [ -e "$f" ] && users="$users $(basename "$(dirname "$f")")"
                done
            done
            users=$(echo $users | tr ' ' '\n' | sort -u | tr '\n' ' ')
            if [ -z "${users// }" ]; then echo "     $bdf  $mode  idle"
            else echo "     $bdf  $mode  in use by pid ${users}"; fi
        done
    else
        fail "MI300X GPUs" "none in /sys/class/kfd (is amdgpu loaded? found: ${gpus[*]:-nothing})"
    fi
    if [ -n "${GPU:-}" ]; then
        if [ -e "/sys/bus/pci/devices/$GPU" ]; then pass "GPU=$GPU" "exists (check above that it is idle and in SPX)"
        else fail "GPU=$GPU" "no such PCI device; pick one listed above"; fi
    else
        warn "GPU" "unset: the scripts use the first GPU; set GPU in src/gpu-fs/env.sh to an idle one from the list above"
    fi

    if [ -f /opt/rocm/.info/version ]; then
        v=$(cat /opt/rocm/.info/version)
        if [ "${v%%.*}" -ge 7 ]; then pass "ROCm" "$v (tested: 7.2.4)"
        else fail "ROCm" "$v: klHook needs ROCm 7 (tested: 7.2.4)"; fi
    else
        fail "ROCm" "no /opt/rocm/.info/version: install ROCm 7 (tested: 7.2.4)"
    fi
    dkms=$(dpkg-query -W -f '${Version}' amdgpu-dkms 2>/dev/null || true)
    if [ -n "$dkms" ]; then pass "amdgpu-dkms" "$dkms (tested: 1:7.1.9.31600000)"
    else warn "amdgpu-dkms" "not installed as a package; the CPX workaround was tested with 7.1.9"; fi
    kmod=$(cat /sys/module/amdgpu/version 2>/dev/null || true)
    [ -n "$kmod" ] && pass "amdgpu module" "$kmod"

    for t in amd-smi systemd-run python3 sudo curl; do
        if have "$t"; then pass "tool $t" "$(command -v "$t")"
        else fail "tool $t" "not found"; fi
    done
    if sudo -n true 2> /dev/null; then pass "sudo" "works without a password prompt"
    else warn "sudo" "asks for a password: the scripts call sudo (partition switch, criu); unattended runs need it without one"; fi
    pass "glibc" "libc6 $(dpkg-query -W -f '${Version}' libc6 2>/dev/null || ldd --version | head -1 | awk '{print $NF}') (keep it unchanged while you have checkpoints)"

    # PyTorch for the mnist example: the build examples/mnist/requirements.txt pins.
    PY=${PYTHON:-/usr/bin/python3}
    req=$HERE/examples/mnist/requirements.txt
    for pkg in torch torchvision; do
        want=$(awk -F'==' -v p="$pkg" '$1==p {print $2}' "$req")
        have=$("$PY" -c "import importlib.metadata as m; print(m.version('$pkg'))" 2>/dev/null || true)
        if [ "$have" = "$want" ]; then pass "$pkg for PYTHON=$PY" "$have (as pinned)"
        elif [ -z "$have" ]; then warn "$pkg for PYTHON=$PY" "not installed: only the mnist example needs it (README step 2, ~10 GB)"
        else warn "$pkg for PYTHON=$PY" "$have, requirements.txt pins $want: the guest image must get the same build"; fi
    done

    check_build klhook
    check_disk 7
else
    GEM5=${GEM5:-${GEM5_DIR:+$GEM5_DIR/build/VEGA_X86/gem5.opt}}
    GEM5_CONFIG=${GEM5_CONFIG:-${GEM5_DIR:+$GEM5_DIR/configs/example/gem5_library/x86-mi300x-gpu.py}}
    chk() {  # name value hint
        if [ -z "$2" ]; then fail "$1" "unset: set it in src/gpu-fs/env.sh ($3)"
        elif [ -e "$2" ]; then pass "$1" "$2"
        else fail "$1" "$2 does not exist ($3)"; fi
    }
    if [ -z "${GEM5:-}" ] && [ -z "${GEM5_CONFIG:-}" ]; then
        fail GEM5_DIR "unset: set it in src/gpu-fs/env.sh to your gem5 tree (README step 6)"
    else
        chk GEM5 "${GEM5:-}" "build it: cd GEM5_DIR && scons build/VEGA_X86/gem5.opt"
        chk GEM5_CONFIG "${GEM5_CONFIG:-}" "use a gem5 that has configs/example/gem5_library/x86-mi300x-gpu.py"
    fi
    chk GUEST_IMG "${GUEST_IMG:-}" "guest-image/build.sh makes it"
    chk GUEST_KERNEL "${GUEST_KERNEL:-}" "the vmlinux built with the image"
    if [ -n "${IPT1:-}" ] && [ -f "$IPT1" ]; then pass IPT1 "$IPT1"
    else chk IPT8 "${IPT8:-}" "8-XCC MI300X IP-discovery binary; examples/gem5.sh makes the 1-XCC one from it"; fi

    if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then pass "/dev/kvm" "usable"
    elif [ -e /dev/kvm ]; then fail "/dev/kvm" "no access: sudo usermod -aG $(stat -c %G /dev/kvm) $USER, then log in again"
    else fail "/dev/kvm" "missing: gem5's GPU full-system runs need KVM"; fi
    for t in fakeroot debugfs mke2fs e2fsck sfdisk python3; do
        if have "$t"; then pass "tool $t" "found"
        else fail "tool $t" "not found (apt install fakeroot e2fsprogs fdisk)"; fi
    done
    if have qemu-system-x86_64; then pass "tool qemu-system-x86_64" "found (guest-image/build.sh)"
    else warn "tool qemu-system-x86_64" "not found: only guest-image/build.sh needs it (apt install qemu-system-x86 unzip)"; fi
    check_build none
    check_disk 10
fi

echo
if [ "$fails" -gt 0 ]; then echo "preflight $MODE: $fails FAIL"; exit 1; fi
echo "preflight $MODE: ready"
