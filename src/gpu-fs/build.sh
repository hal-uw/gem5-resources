#!/bin/bash
# Build CRIU (with criu-patches/*.patch applied), its amdgpu plugins and klHook
# (with klhook-patches/*.patch applied).
#   build.sh
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
ROCM=${ROCM_PATH:-/opt/rocm}
cd "$HERE"

git submodule update --init -- criu klHook

# Apply the patch series in <dir> to submodule <sub> once, in name order, as
# working-tree changes on top of the pinned commit.
apply_patches() {
    local sub=$1 dir=$2 p
    local patches=("$HERE/$dir"/*.patch)
    [ -e "${patches[0]}" ] || return 0
    if git -C "$sub" diff --quiet HEAD; then
        for p in "${patches[@]}"; do
            git -C "$sub" apply "$p"
            echo "applied: $dir/$(basename "$p")"
        done
    elif git -C "$sub" apply --reverse --check "${patches[-1]}" 2> /dev/null; then
        echo "already applied: $dir/*.patch"
    else
        echo "$sub has local changes that are not $dir/*.patch" >&2
        return 1
    fi
}
apply_patches criu criu-patches
apply_patches klHook klhook-patches

# /usr/bin first: a protoc from conda etc. clashes with the system protobuf.
PATH=/usr/bin:$PATH make -C criu -j"$(nproc)" criu
PATH=/usr/bin:$PATH make -C criu amdgpu_plugin

# --as-needed: klHook dlopens HIP/HSA itself (KL_HOOK_HIP_LIB_PATH); a link-time
# libamdhip64 would load a second runtime next to e.g. the one PyTorch bundles.
"$ROCM/bin/hipcc" -std=c++17 -shared -fPIC -Wl,--as-needed -o klHook/libklhook.so klHook/klhook.cc -ldl

echo "built: criu/criu/criu criu/plugins/amdgpu/amdgpu_plugin.so" \
     "criu/plugins/amdgpu/gem5/amdgpu_plugin_gem5.so klHook/libklhook.so"
