#!/bin/bash
# Build CRIU (with criu-patches/*.patch applied), its amdgpu plugin and klHook.
#   build.sh
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
ROCM=${ROCM_PATH:-/opt/rocm}
cd "$HERE"

git submodule update --init -- criu klHook

# Apply each patch once; skip patches that are already applied.
for p in criu-patches/*.patch; do
    [ -e "$p" ] || continue
    if git -C criu apply --reverse --check "$HERE/$p" 2> /dev/null; then
        echo "already applied: $p"
    else
        git -C criu apply "$HERE/$p"
        echo "applied: $p"
    fi
done

# /usr/bin first: a protoc from conda etc. clashes with the system protobuf.
PATH=/usr/bin:$PATH make -C criu -j"$(nproc)" criu
PATH=/usr/bin:$PATH make -C criu amdgpu_plugin

"$ROCM/bin/hipcc" -std=c++17 -shared -fPIC -o klHook/libklhook.so klHook/klhook.cc -ldl

echo "built: criu/criu/criu criu/plugins/amdgpu/amdgpu_plugin.so" \
     "criu/plugins/amdgpu/gem5/amdgpu_plugin_gem5.so klHook/libklhook.so"
