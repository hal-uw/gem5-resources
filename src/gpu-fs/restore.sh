#!/bin/bash
# Restore a checkpoint made by ckpt.sh and wait for the app to exit.
#   restore.sh <dir>
# On the MI300X host it restores onto one CPX partition (CPX during the run,
# SPX afterwards). Without amd-smi (the gem5 guest) it restores directly.
# Env: GPU, PART, CRIU, PLUGIN (as in ckpt.sh).
set -uo pipefail
HERE=$(dirname "$(realpath "$0")")
DIR=$(realpath "${1:?usage: $0 <dir>}")
GPU=${GPU:-0000:f6:00.0}
PART=${PART:-0}
CRIU=${CRIU:-$HERE/criu/criu/criu}
PLUGIN=${PLUGIN:-$HERE/criu/plugins/amdgpu}
SUDO=; [ "$EUID" = 0 ] || SUDO=sudo

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

cd "$DIR" || exit 2
cp -p app.log.at-dump app.log

DEV=()
if command -v amd-smi > /dev/null && [ -e "/sys/bus/pci/devices/$GPU" ]; then
    trap '$SUDO amd-smi set --gpu "$GPU" --compute-partition SPX > /dev/null' EXIT
    $SUDO amd-smi set --gpu "$GPU" --compute-partition CPX > /dev/null || exit 1
    R=$(render_node)
    DEV=(systemd-run --quiet --scope -p DevicePolicy=closed
         -p "DeviceAllow=/dev/kfd rw" -p "DeviceAllow=/dev/dri/$R rw")
fi

# Skip the checks where gem5's MI300X differs from the real one.
$SUDO "${DEV[@]}" env KFD_SDMA_FW_VER_CHECK=0 KFD_CAPABILITY_CHECK=0 \
    KFD_VRAM_SIZE_CHECK=0 KFD_CACHES_COUNT_CHECK=0 \
    "$CRIU" restore -D . -v4 -o restore.log \
    --link-remap --tcp-established --file-locks --skip-file-rwx-check -L "$PLUGIN"
rc=$?
echo "criu restore exit $rc"
cat app.log
if command -v m5 > /dev/null; then
    m5 writefile app.log app.log.after-restore
    m5 writefile restore.log restore.log
fi
cp -p app.log.at-dump app.log    # keep the checkpoint restorable
exit $rc
