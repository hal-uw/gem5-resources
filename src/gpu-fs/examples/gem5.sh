#!/bin/bash
# On the gem5 host: restore an example's checkpoint package in gem5 and check
# the result.
#   gem5.sh APP [PACKAGE]
# APP:     square, bc, mnist (what to expect in the app's output)
# PACKAGE: the APP.gem5.tar.gz made by host.sh (default: $CK/APP.gem5.tar.gz)
# Needs, in src/gpu-fs/env.sh: GEM5_DIR (or GEM5 and GEM5_CONFIG), GUEST_IMG,
# GUEST_KERNEL, and IPT8 or IPT1 (utils/preflight.sh gem5 checks them).
# Steps, each skipped when its output is newer than its input:
#   1. IPT1, the single-XCC IP-discovery binary, from IPT8
#   2. the package disk PACKAGE%.tar.gz.img (utils/make_pkg_disk.sh)
#   3. gem5, output in $CK/m5out-APP (gem5's own log: m5out-APP/gem5.log)
#   4. utils/gem5_verdict.py, then the app's own output
# The last line is "PASS ..." or "FAIL ..."; the exit status is 0 on PASS.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
[ $# -ge 1 ] || die "usage: $0 square|bc|mnist [package.gem5.tar.gz]"
load_app "$1"
PKG=$(realpath "${2:-$PKG}")
[ -f "$PKG" ] || die "no package $PKG (copy it from the MI300X host)"
GEM5=${GEM5:-${GEM5_DIR:+$GEM5_DIR/build/VEGA_X86/gem5.opt}}
GEM5_CONFIG=${GEM5_CONFIG:-${GEM5_DIR:+$GEM5_DIR/configs/example/gem5_library/x86-mi300x-gpu.py}}
IPT1=${IPT1:-$CK/mi300x_1xcc_discovery.bin}
for v in GEM5 GEM5_CONFIG GUEST_IMG GUEST_KERNEL; do
    [ -n "${!v:-}" ] && [ -e "${!v}" ] || die "$v='${!v:-}' not found: set it in src/gpu-fs/env.sh (utils/preflight.sh gem5)"
done
OUT=$CK/m5out-$APP
mkdir -p "$CK"
echo "$APP_DESC"

ipt=$(tar xzOf "$PKG" --wildcards '*/manifest.json' | python3 -c 'import json,sys; print(json.load(sys.stdin)["ipt"])')
step "IP-discovery binary for ipt=$ipt"
if [ "$ipt" = 1xcc ]; then
    if [ ! -s "$IPT1" ]; then
        [ -n "${IPT8:-}" ] && [ -f "$IPT8" ] || die "IPT8 not found: set it in src/gpu-fs/env.sh"
        "$GPUFS/utils/make_1xcc_ipt.py" "$IPT8" "$IPT1"
    fi
    IPT=$IPT1
else
    [ -n "${IPT8:-}" ] && [ -f "$IPT8" ] || die "IPT8 not found: set it in src/gpu-fs/env.sh"
    IPT=$IPT8
fi
echo "using $IPT"

DISK=${PKG%.tar.gz}.img
step "package disk $DISK"
if [ ! -s "$DISK" ] || [ "$PKG" -nt "$DISK" ]; then
    BASE=$GUEST_IMG "$GPUFS/utils/make_pkg_disk.sh" "$PKG" "$DISK"
fi
ls -lh "$DISK"

step "gem5 restore into $OUT (slow: minutes for square, hours for mnist)"
rm -rf "$OUT"; mkdir -p "$OUT"
# shellcheck disable=SC2086  # GEM5_OPTS and CONFIG_OPTS are word lists
"$GEM5" -d "$OUT" ${GEM5_OPTS:-} \
    "$GPUFS/utils/pkg_disk_config.py" "$DISK" "$GEM5_CONFIG" ${CONFIG_OPTS:-} \
    --image "$GUEST_IMG" --kernel "$GUEST_KERNEL" --ipt-binary "$IPT" \
    --app "$GC" --opts "restore --gem5" > "$OUT/gem5.log" 2>&1 ||
    echo "WARN: gem5 exited $? (see $OUT/gem5.log)"

step "verdict"
"$GPUFS/utils/gem5_verdict.py" "$OUT" || die "gem5_verdict.py (see $OUT/restore.log, $OUT/dmesg-after-restore.txt)"
grep -qF "$APP_EXPECT" "$OUT/app.log.after-restore" ||
    die "restored app did not print '$APP_EXPECT' (see $OUT/app.log.after-restore)"
echo
echo "PASS $APP gem5 restore ($OUT)"
