#!/bin/bash
# Build the gem5 guest disk image and kernel for restoring gpu-fs checkpoints:
# the stock src/x86-ubuntu-gpu-ml packer build with guest-image/rocm-install.sh
# (ROCm 7.2.4, criu's runtime libraries, pinned PyTorch) in place of its own.
#   build.sh WORKDIR [packer build options...]
# WORKDIR (outside the repo: it gets ~80 GB) gets a copy of
# src/x86-ubuntu-gpu-ml; the stock directory is not changed. Outputs:
#   WORKDIR/disk-image/x86-ubuntu-gpu-rocm72   GUEST_IMG
#   WORKDIR/vmlinux-rocm72                     GUEST_KERNEL
#   WORKDIR/files/mi300_discovery              IPT8 (8-XCC IP discovery)
# Needs qemu-system-x86_64, unzip, KVM, network, ~100 GB free; takes ~1 h.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STOCK=$(cd "$HERE/../../x86-ubuntu-gpu-ml" && pwd)
[ $# -ge 1 ] || { echo "usage: $0 WORKDIR [packer build options...]  (WORKDIR: ~100 GB free, outside the repo)" >&2; exit 2; }
WORK=$(realpath -m "$1")
shift
NAME=x86-ubuntu-gpu-rocm72

# The guest's PyTorch must be the build the MI300X host checkpoints with.
req=$HERE/../examples/mnist/requirements.txt
for pin in $(grep -E '^(torch|torchvision)==' "$req"); do
    grep -q "=$pin\$" "$HERE/rocm-install.sh" ||
        { echo "rocm-install.sh does not install $pin from $req" >&2; exit 1; }
done

mkdir -p "$WORK"
rsync -a --exclude disk-image --exclude packer "$STOCK/" "$WORK/"
cp "$HERE/rocm-install.sh" "$WORK/scripts/rocm-install.sh"
# Room for ROCm 7.2 and PyTorch; the kernel under its own name.
sed -i -e 's/disk_size *= *"[0-9]*"/disk_size        = "80000"/' \
       -e "s/vmlinux-rocm70/vmlinux-rocm72/" "$WORK/x86-ubuntu-gpu-ml.pkr.hcl"
cd "$WORK"
./build.sh -var "image_name=$NAME" "$@"

cat <<EOF

Done. In src/gpu-fs/env.sh:
GUEST_IMG=$WORK/disk-image/$NAME
GUEST_KERNEL=$WORK/vmlinux-rocm72
IPT8=$WORK/files/mi300_discovery
EOF
