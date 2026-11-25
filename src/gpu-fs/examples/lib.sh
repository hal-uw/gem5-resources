# Shared by examples/host.sh and examples/gem5.sh. Not run directly.
GPUFS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GR=$(cd "$GPUFS/../.." && pwd)              # gem5-resources root
[ -f "$GPUFS/env.sh" ] && . "$GPUFS/env.sh"
CK=${CK:-$HOME/gpu-ckpt}
GC=$GPUFS/gpu_checkpoint.py
export GPU                                   # read by gpu_checkpoint.py

die() { echo "FAIL: $*" >&2; exit 1; }
step() { echo; echo "=== $*"; }

# load_app NAME: sets APP_KERNEL, APP_EXPECT, APP_DESC and defines app_prepare
# (builds or downloads the app into $W, sets APP_CWD and the APP_CMD array).
load_app() {
    APP=$1
    [ -f "$GPUFS/examples/apps/$APP.sh" ] ||
        die "unknown app '$APP' (one of: $(cd "$GPUFS/examples/apps" && ls | sed 's/\.sh$//' | tr '\n' ' '))"
    W=$CK/work/$APP                          # the app's build and working dir
    DIR=$CK/$APP                             # the checkpoint
    PKG=$CK/$APP.gem5.tar.gz                 # the gem5 package
    . "$GPUFS/examples/apps/$APP.sh"
}
