#!/bin/bash
# Run an app under klHook on one CPX partition of an MI300X and checkpoint it
# right before the kernel launch that klHook selects.
#   KL_CKPT_KERNEL=<kernel>[:<n>] kl-ckpt.sh [--continue] <dir> <cmd> [args...]
# --continue: dump with --leave-running and let the app finish (exit status is
# the app's). Otherwise criu kills the app after the dump. The GPU is CPX
# during the run and SPX afterwards. Restore with restore.sh <dir>.
# Env: KLHOOK (libklhook.so), GPU (PCI address), PART (0-7), CRIU, PLUGIN.
set -euo pipefail
CONT=; [ "${1:-}" = --continue ] && { CONT=1; shift; }
[ $# -ge 2 ] && [ -n "${KL_CKPT_KERNEL:-}" ] || {
    echo "usage: KL_CKPT_KERNEL=<kernel>[:<n>] $0 [--continue] <dir> <cmd> [args...]" >&2; exit 2; }
HERE=$(dirname "$(realpath "$0")")
DIR=$(realpath -m "$1"); shift
CMD=$(realpath "$(type -P "$1")" 2>/dev/null) || { echo "not found: $1" >&2; exit 2; }; shift
KLHOOK=$(realpath "${KLHOOK:-$HERE/klHook/libklhook.so}")
GPU=${GPU:-0000:f6:00.0}
PART=${PART:-0}
CRIU=${CRIU:-$HERE/criu/criu/criu}
PLUGIN=${PLUGIN:-$HERE/criu/plugins/amdgpu}
UNIT=kl-ckpt-$$
[ -f "$KLHOOK" ] || { echo "no $KLHOOK (run build.sh first)" >&2; exit 2; }

# Render node of partition PART: KFD location_id = bus<<8 | dev<<3 | fn, + partition.
render_node() {
    local bdf=${GPU#*:}
    local loc=$(( (16#${bdf%%:*} << 8 | 16#${bdf:3:2} << 3 | 16#${bdf##*.}) + PART ))
    awk -v l=$loc -v d=$((16#${GPU%%:*})) '
        FNR == 1 { a = b = 0 }
        $1 == "location_id" && $2 == l { a = 1 }
        $1 == "domain" && $2 == d { b = 1 }
        $1 == "drm_render_minor" && a && b { print "renderD" $2; exit }
    ' /sys/class/kfd/kfd/topology/nodes/*/properties
}
running() { [ "$(systemctl show -P SubState "$UNIT")" = running ]; }

cleanup() {
    sudo systemctl stop "$UNIT" 2>/dev/null || true
    sudo systemctl reset-failed "$UNIT" 2>/dev/null || true
    sudo amd-smi set --gpu "$GPU" --compute-partition SPX > /dev/null
}

mkdir "$DIR" && cd "$DIR"
trap cleanup EXIT
sudo amd-smi set --gpu "$GPU" --compute-partition CPX > /dev/null
R=$(render_node)
[ -n "$R" ] || { echo "no render node for $GPU partition $PART" >&2; exit 1; }

# App and criu share a device cgroup with only /dev/kfd and $R, so KFD shows
# them one GPU (it filters its topology by the reader's cgroup).
DEV=(-p DevicePolicy=closed -p "DeviceAllow=/dev/kfd rw" -p "DeviceAllow=/dev/dri/$R rw")

# HSA_ENABLE_INTERRUPT=0: gem5 sends no completion IRQs, so ROCr must poll.
# GLIBC_TUNABLES: gem5's CPU is x86-64-v2; hide newer ISA from glibc's ifuncs.
M=-AVX512F,-AVX512CD,-AVX512BW,-AVX512DQ,-AVX512VL,-AVX512_VBMI,-AVX512_VBMI2
M=$M,-AVX512_IFMA,-AVX512_VNNI,-AVX512_BITALG,-AVX512_VPOPCNTDQ,-AVX512_BF16
M=$M,-AVX512_FP16,-AVX_VNNI,-AVX2,-BMI1,-BMI2,-LZCNT,-MOVBE
sudo systemd-run --quiet --unit="$UNIT" "${DEV[@]}" -p User="$(id -un)" \
    -p WorkingDirectory="$DIR" -p StandardInput=null -p RemainAfterExit=yes \
    -p StandardOutput="file:$DIR/app.log" -p StandardError=inherit \
    --setenv=HSA_ENABLE_INTERRUPT=0 --setenv=GLIBC_TUNABLES=glibc.cpu.hwcaps=$M \
    --setenv=LD_PRELOAD="$KLHOOK" --setenv=KL_CKPT_KERNEL="$KL_CKPT_KERNEL" \
    --setenv=KL_CKPT_READY="$DIR/.kl-ready" --setenv=KL_CKPT_GO="$DIR/.kl-go" \
    "$CMD" "$@"
PID=$(systemctl show -P MainPID "$UNIT")

# klHook creates .kl-ready once the GPU is drained, right before the launch.
until [ -e .kl-ready ]; do
    running || { cat app.log; echo "app exited before $KL_CKPT_KERNEL was launched" >&2; exit 1; }
    sleep 0.1
done

sudo systemd-run --quiet --scope "${DEV[@]}" "$CRIU" dump -t "$PID" -D . -v4 -o dump.log \
    --link-remap --tcp-established --file-locks --unprivileged -L "$PLUGIN" \
    ${CONT:+--leave-running}
sudo chown "$(id -u):$(id -g)" ./*.img dump.log app.log && chmod 644 ./*.img
cp -p app.log app.log.at-dump    # restore.sh puts this back: criu checks its size
echo "checkpoint of $CMD before $KL_CKPT_KERNEL (pid $PID, $R) in $DIR"

rc=0
if [ -n "$CONT" ]; then
    touch .kl-go
    while running; do sleep 0.1; done
    rc=$(systemctl show -P ExecMainStatus "$UNIT")
    echo "app exited $rc"
fi
rm -f .kl-ready .kl-go
exit "$rc"
