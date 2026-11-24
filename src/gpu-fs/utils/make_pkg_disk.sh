#!/bin/bash
# Make the package disk of a gem5 checkpoint package (gpu_checkpoint.py
# checkpoint --gem5): a small ext4 image that run_gem5_restore.sh attaches as
# the guest's second disk. The guest's root image is never changed.
#   make_pkg_disk.sh <package.gem5.tar.gz> [<output.img>]
# The disk (label gem5pkg) holds
#   /rootfs/<abs path>   the package's files, with their owners and modes,
#                        except those identical to the root image's (BASE)
#   /criu/               criu and amdgpu_plugin_gem5 (CRIU_TREE)
#   /manifest.json       from the package
# restore --gem5 mounts it and lays /rootfs over the guest's / (overlayfs),
# so the package's files win over the image's at the same paths.
# Env:
#   BASE       root disk image to leave identical files out against
#              (default: the k138 image; BASE= keeps every file)
#   CRIU_TREE  built CRIU tree (default: ../criu, after
#              build.sh)
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
TAR=$(realpath "${1:?usage: $0 <package.gem5.tar.gz> [<output.img>]}")
OUT=$(realpath -m "${2:-${TAR%.tar.gz}.img}")
GB=/nobackup2/vishnu/gpu-fs/mgpu/gem5-resources/src/x86-ubuntu-gpu-ml
BASE=${BASE-$GB/disk-image-k138/x86-ubuntu-rocm72-k138}
CRIU_TREE=${CRIU_TREE:-$HERE/../criu}
for f in criu/criu plugins/amdgpu/gem5/amdgpu_plugin_gem5.so; do
    [ -f "$CRIU_TREE/$f" ] || { echo "no $CRIU_TREE/$f (run build.sh or set CRIU_TREE)" >&2; exit 1; }
done

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
st=$tmp/stage
mkdir -p "$st"
# fakeroot keeps the tarball's owners for mke2fs -d, in one session.
fakeroot -s "$tmp/fr" tar xzf "$TAR" -C "$st" --same-owner --numeric-owner
name=$(ls "$st"); pkg=$st/$name
[ -d "$pkg/rootfs" ] && [ -f "$pkg/manifest.json" ] || { echo "not a gem5 package: $TAR" >&2; exit 1; }
(cd "$pkg/rootfs" && sha256sum --quiet -c ../SHA256SUMS) && echo "SHA256SUMS ok"

if [ -n "$BASE" ]; then
    # One read-only debugfs session dumps every package file the root image
    # has; identical ones need not be on the package disk.
    (cd "$pkg/rootfs" && find . -type f | sed 's|^\.||' | sort) > "$tmp/files"
    mkdir "$tmp/d"
    awk -v d="$tmp/d" '{ printf "dump %s %s/%d\n", $0, d, NR }' "$tmp/files" > "$tmp/dumpcmds"
    /sbin/debugfs -f "$tmp/dumpcmds" "$BASE?offset=1048576" > /dev/null 2>&1 || true
    n=0; same=0; differ=0
    while read -r p; do
        n=$((n + 1))
        if [ -e "$tmp/d/$n" ]; then
            if cmp -s "$pkg/rootfs$p" "$tmp/d/$n"; then
                rm -f "$pkg/rootfs$p"; same=$((same + 1))
            else
                echo "differs from the root image: $p"; differ=$((differ + 1))
            fi
            rm -f "$tmp/d/$n"
        fi
    done < "$tmp/files"
    echo "$n files: $same identical to the root image (left out), $differ differ"
fi

fakeroot -i "$tmp/fr" -s "$tmp/fr" sh -c '
    mkdir -p "$1/criu/criu" "$1/criu/plugins/amdgpu/gem5"
    cp "$2/criu/criu" "$1/criu/criu/"
    cp "$2/plugins/amdgpu/gem5/amdgpu_plugin_gem5.so" "$1/criu/plugins/amdgpu/gem5/"
    chown -R 0:0 "$1/criu"; chmod 755 "$1/criu/criu/criu"; chmod 644 "$1/criu/plugins/amdgpu/gem5/"*.so
    chown 0:0 "$1/manifest.json"; rm -f "$1/SHA256SUMS"' \
    sh "$pkg" "$CRIU_TREE"

kb=$(du -sk --apparent-size "$pkg" | cut -f1)
size=$(( kb + kb / 5 + 65536 ))k   # +20% + 64 MiB for ext4 metadata
rm -f "$OUT"
fakeroot -i "$tmp/fr" /sbin/mke2fs -q -t ext4 -L gem5pkg -E root_owner=0:0 \
    -d "$pkg" "$OUT" "$size"
/sbin/e2fsck -fn "$OUT" > /dev/null
echo "package disk: $OUT ($(du -h --apparent-size "$OUT" | cut -f1)), $(python3 -c "
import json; m = json.load(open('$pkg/manifest.json'))
print('checkpoint', m['checkpoint'], '| ipt', m.get('ipt'))")"
