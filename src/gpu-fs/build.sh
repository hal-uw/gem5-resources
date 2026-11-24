#!/bin/bash
# Build CRIU (with criu-patches/*.patch applied), its amdgpu plugins and klHook
# (with klhook-patches/*.patch applied).
#   build.sh
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
ROCM=${ROCM_PATH:-/opt/rocm}
cd "$HERE"

git submodule update --init -- criu klHook

# Bring submodule <sub> to its pinned commit + the patch series in <dir> (name
# order), as working-tree changes. Compares whole trees, untracked files
# included, so a rerun works whatever is left over: already patched -> nothing
# to do; partly patched (e.g. tracked files reverted by a checkout while the
# files the patches create stay) -> those files are completed. Any other local
# change stops the build, nothing is overwritten.
apply_patches() {
    local sub=$1 dir=$2 p path want have base bad=()
    local patches=("$HERE/$dir"/*.patch)
    [ -e "${patches[0]}" ] || return 0
    local tmp; tmp=$(mktemp -d)
    # want: HEAD + patches; have: the working tree (git add -A on a scratch
    # index: untracked files count, ignored build outputs do not)
    GIT_INDEX_FILE=$tmp/want git -C "$sub" read-tree HEAD
    for p in "${patches[@]}"; do
        GIT_INDEX_FILE=$tmp/want git -C "$sub" apply --cached "$p"
    done
    want=$(GIT_INDEX_FILE=$tmp/want git -C "$sub" write-tree)
    GIT_INDEX_FILE=$tmp/have git -C "$sub" read-tree HEAD
    GIT_INDEX_FILE=$tmp/have git -C "$sub" add -A
    have=$(GIT_INDEX_FILE=$tmp/have git -C "$sub" write-tree)
    base=$(git -C "$sub" rev-parse "HEAD^{tree}")
    if [ "$have" = "$want" ]; then
        echo "already applied: $dir/*.patch"
    else
        # every file that differs must be as in HEAD or as patched
        while read -r path; do
            have_blob=$(git -C "$sub" rev-parse -q --verify "$have:$path" || true)
            if [ "$have_blob" != "$(git -C "$sub" rev-parse -q --verify "$want:$path" || true)" ] &&
               [ "$have_blob" != "$(git -C "$sub" rev-parse -q --verify "$base:$path" || true)" ]; then
                bad+=("$path")
            fi
        done < <(git -C "$sub" diff-tree -r --name-only "$have" "$want")
        if [ ${#bad[@]} -gt 0 ]; then
            echo "$sub has local changes that are not $dir/*.patch:" >&2
            printf '  %s\n' "${bad[@]}" >&2
            rm -rf "$tmp"
            return 1
        fi
        while read -r path; do
            if git -C "$sub" rev-parse -q --verify "$want:$path" > /dev/null; then
                GIT_INDEX_FILE=$tmp/want git -C "$sub" checkout-index -f -- "$path"
            else
                rm -f "$sub/$path"
            fi
        done < <(git -C "$sub" diff-tree -r --name-only "$have" "$want")
        echo "applied: $dir/*.patch"
    fi
    rm -rf "$tmp"
}
apply_patches criu criu-patches
apply_patches klHook klhook-patches

# /usr/bin first: a protoc from conda etc. clashes with the system protobuf.
PATH=/usr/bin:$PATH make -C criu -j"$(nproc)" criu
PATH=/usr/bin:$PATH make -C criu amdgpu_plugin

echo "built: criu/criu/criu criu/plugins/amdgpu/amdgpu_plugin.so" \
     "criu/plugins/amdgpu/gem5/amdgpu_plugin_gem5.so"

# klHook is needed only where apps are checkpointed: it is loaded into the app,
# and a gem5 package carries the copy the app had mapped. Skip it where it
# cannot be built.
HIPCC=$ROCM/bin/hipcc
skip_klhook() {
    echo "klHook not built: $* (it is needed only on the checkpoint host)"
    exit 0
}
[ -x "$HIPCC" ] || skip_klhook "no $HIPCC"
# It hooks hipLaunchKernelExC, which older HIP (e.g. ROCm 6.2) does not have.
grep -qs hipLaunchKernelExC "$ROCM"/include/hip/*.h ||
    skip_klhook "the HIP in $ROCM has no hipLaunchKernelExC (ROCm too old)"

# klHook has no device code: --offload-host-only, so no GPU target is needed.
# clang takes the newest GCC installation it finds, which may lack its C++
# headers (e.g. gcc-14 without libstdc++-14-dev). If <cmath> does not compile,
# try the GCC installations from newest to oldest.
cxx_ok() {
    echo '#include <cmath>' |
        "$HIPCC" --offload-host-only "$@" -x hip -fsyntax-only - 2> /dev/null
}
gcc=()
if ! cxx_ok; then
    # GCC installations for this machine's target, newest version first
    for d in $(printf '%s\n' /usr/lib/gcc/"$(uname -m)"-*/* /usr/lib64/gcc/"$(uname -m)"-*/* |
               awk -F/ '{ print $NF "\t" $0 }' | sort -rV | cut -f2); do
        [ -d "$d" ] || continue
        if cxx_ok --gcc-install-dir="$d"; then gcc=(--gcc-install-dir="$d"); break; fi
    done
    [ ${#gcc[@]} -gt 0 ] || skip_klhook "$HIPCC finds no GCC with C++ headers"
    echo "klHook: using ${gcc[0]}"
fi
# --as-needed: klHook dlopens HIP/HSA itself (KL_HOOK_HIP_LIB_PATH); a link-time
# libamdhip64 would load a second runtime next to e.g. the one PyTorch bundles.
"$HIPCC" --offload-host-only "${gcc[@]}" -std=c++17 -shared -fPIC -Wl,--as-needed \
    -o klHook/libklhook.so klHook/klhook.cc -ldl
echo "built: klHook/libklhook.so"
