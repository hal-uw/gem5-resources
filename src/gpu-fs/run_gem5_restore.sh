#!/bin/bash
# Boot the gem5 guest and run 'gpu_checkpoint.py restore' on a checkpoint inside the image.
#   run_gem5_restore.sh <outdir> <checkpoint-dir-in-guest>
# GUEST_CRIU: CRIU tree inside the guest image (criu/criu, plugins/amdgpu).
OUT=$(realpath -m "${1:?usage: $0 <outdir> <checkpoint-dir-in-guest>}")
CKPT=${2:?checkpoint dir inside the guest image}
G=${GEM5:-/nobackup2/vishnu/gpu-fs/gap-gem5}               # branch criu-restore
GB=/nobackup2/vishnu/gpu-fs/mgpu/gem5-resources/src/x86-ubuntu-gpu-ml
GC=${GUEST_CRIU:-/users/vramadas/square/criu}
HERE=$(dirname "$(realpath "$0")")

mkdir -p "$OUT"; APP=$OUT/guest_restore.sh
{ echo '#!/bin/bash'
  echo "cat > /tmp/gpu_checkpoint.py <<'GPU_CHECKPOINT_EOF'"; cat "$HERE/gpu_checkpoint.py"; echo GPU_CHECKPOINT_EOF
  echo "CRIU=$GC/criu/criu PLUGIN=$GC/plugins/amdgpu python3 /tmp/gpu_checkpoint.py restore $CKPT"
} > "$APP"

cd "$G" && build/VEGA_X86/gem5.opt -d "$OUT" \
    --debug-flags=GPUKernelInfo --debug-file=trace.txt \
    configs/example/gem5_library/x86-mi300x-gpu.py \
    --image $GB/disk-image-k138/x86-ubuntu-rocm72-k138 \
    --kernel $GB/vmlinux-rocm72-k138 \
    --ipt-binary $GB/disk-image-rocm72/mi300x_hybrid_discovery.bin \
    --app "$APP"
# Pass = the app's output after restore, and the GPU ran kernels.
cat "$OUT/app.log.after-restore"
grep -c "Completed kernel" "$OUT/trace.txt"
