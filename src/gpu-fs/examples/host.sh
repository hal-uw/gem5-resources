#!/bin/bash
# On the MI300X host: build an example app, profile it, checkpoint it for gem5
# and restore it natively, checking each step.
#   host.sh APP [STEP]
# APP:  square (smoke test), bc, mnist (examples/apps/*.sh)
# STEP: all (default) | prepare | profile | checkpoint | restore
# Writes, under CK (default $HOME/gpu-ckpt; set it in src/gpu-fs/env.sh):
#   work/APP/            the app's build and working directory
#   APP.profile.txt      kernel launch counts
#   APP/                 the checkpoint (must stay at this path)
#   APP.gem5.tar.gz      the gem5 package: copy it to the gem5 host
#   APP.restore.txt      output of the native restore
# The last line is "PASS ..." or "FAIL ..."; the exit status is 0 on PASS.
# FORCE=1 replaces an existing checkpoint and package of APP.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
[ $# -ge 1 ] || die "usage: $0 square|bc|mnist [all|prepare|profile|checkpoint|restore]"
load_app "$1"
STEP=${2:-all}
mkdir -p "$CK"
echo "$APP_DESC"
echo "GPU=${GPU:-<first GPU>} CK=$CK"

do_prepare() {
    step "prepare: build or fetch $APP in $W"
    app_prepare
    echo "cwd: $APP_CWD"
    echo "command: ${APP_CMD[*]}"
}

do_profile() {
    step "profile: run $APP once under klHook, count kernel launches"
    (cd "$APP_CWD" && "$GC" profile "${APP_CMD[@]}") 2>&1 | tee "$CK/$APP.profile.txt"
    local n name=${APP_KERNEL%%:*}
    n=$(awk '/kernel launches,/ {print $1}' "$CK/$APP.profile.txt")
    [ -n "$n" ] || die "profile printed no launch counts (see $CK/$APP.profile.txt)"
    [ "$n" = "$APP_LAUNCHES" ] ||
        echo "WARN: $n launches, expected $APP_LAUNCHES: the checkpoint kernel ($APP_KERNEL) may need changing"
    grep -qE "^ +[0-9]+  $name\$" "$CK/$APP.profile.txt" ||
        die "kernel $name not launched (see $CK/$APP.profile.txt)"
    echo "profile OK: $n launches, $name among them"
}

do_checkpoint() {
    step "checkpoint: $APP before $APP_KERNEL into $DIR, package $PKG"
    if [ -e "$DIR" ] || [ -e "$PKG" ]; then
        [ "${FORCE:-0}" = 1 ] || die "$DIR or $PKG exists: remove them or set FORCE=1"
        rm -rf "$DIR" "$PKG"
    fi
    (cd "$APP_CWD" && "$GC" checkpoint --gem5 --kernel "$APP_KERNEL" "$DIR" "${APP_CMD[@]}") ||
        die "checkpoint (see $DIR/app.log and $DIR/dump.log)"
    [ -s "$PKG" ] || die "no package $PKG"
    echo "checkpoint OK: $(du -sh "$DIR" | cut -f1) in $DIR, package $(du -sh "$PKG" | cut -f1)"
}

do_restore() {
    step "restore: $DIR natively on this host"
    [ -d "$DIR" ] || die "no checkpoint $DIR: run the checkpoint step first"
    "$GC" restore "$DIR" 2>&1 | tee "$CK/$APP.restore.txt" || true
    grep -q 'kl-hook: restored' "$CK/$APP.restore.txt" ||
        die "the app did not resume (see $CK/$APP.restore.txt and $DIR/restore.log)"
    grep -qF "$APP_EXPECT" "$CK/$APP.restore.txt" ||
        die "restored app did not print '$APP_EXPECT' (see $CK/$APP.restore.txt)"
    echo "restore OK: '$APP_EXPECT' after 'kl-hook: restored'"
}

case $STEP in
prepare) do_prepare ;;
profile) do_prepare; do_profile ;;
checkpoint) do_prepare; do_checkpoint ;;
restore) do_restore ;;
all) do_prepare; do_profile; do_checkpoint; do_restore ;;
*) die "unknown step $STEP" ;;
esac
echo
case $STEP in all|checkpoint) echo "PASS $APP $STEP (gem5 package: $PKG)" ;; *) echo "PASS $APP $STEP" ;; esac
