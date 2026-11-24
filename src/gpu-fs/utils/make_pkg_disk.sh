#!/bin/bash
# Make the package disk of a gem5 checkpoint package (gpu_checkpoint.py
# checkpoint --gem5): a small ext4 image that gem5 attaches as the guest's
# second disk (utils/pkg_disk_config.py). The root image is never changed.
#   make_pkg_disk.sh <package.gem5.tar.gz> [<output.img>]
# The disk (label gem5pkg) holds
#   /rootfs/<abs path>   the package's files, with their owners and modes,
#                        except those identical to the root image's (BASE)
#   /criu/               criu and amdgpu_plugin_gem5 (CRIU_TREE)
#   /manifest.json       from the package
# In the guest, gpu_checkpoint.py restore --gem5 (gem5's --app, with --opts
# "restore --gem5") lays /rootfs over / (overlayfs): the package's files win over the image's
# at the same paths. Attach it with utils/pkg_disk_config.py; the guest's
# IP-discovery binary must match the manifest's "ipt" (1xcc: one CPX
# partition, utils/make_1xcc_ipt.py).
# Env:
#   BASE       the guest's root disk image: files identical to the ones in
#              it are left out (unset: keep every file). Read only.
#   CRIU_TREE  built CRIU tree (default: ../criu, after
#              build.sh)
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
TAR=$(realpath "${1:?usage: $0 <package.gem5.tar.gz> [<output.img>]}")
OUT=$(realpath -m "${2:-${TAR%.tar.gz}.img}")
BASE=${BASE:-}
# e2fsprogs live in /sbin on some systems, which may not be in PATH
tool() { command -v "$1" || command -v "/usr/sbin/$1" || command -v "/sbin/$1" ||
         { echo "$1 not found (install e2fsprogs)" >&2; exit 1; }; }
DEBUGFS=$(tool debugfs) MKE2FS=$(tool mke2fs) E2FSCK=$(tool e2fsck)
command -v fakeroot > /dev/null || { echo "fakeroot not found" >&2; exit 1; }
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
    # the root file system: the image's first partition (gem5's
    # root_partition 1), or the whole image without a partition table
    off=$(python3 -I -c '
import json, shutil, subprocess, sys
sf = shutil.which("sfdisk") or shutil.which("sfdisk", path="/usr/sbin:/sbin")
r = subprocess.run([sf, "-J", sys.argv[1]], capture_output=True, text=True)
t = json.loads(r.stdout)["partitiontable"] if r.returncode == 0 else None
print(t["partitions"][0]["start"] * t.get("sectorsize", 512) if t else 0)' "$BASE")
    "$DEBUGFS" -f "$tmp/dumpcmds" "$BASE?offset=$off" > /dev/null 2>&1 || true
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
fakeroot -i "$tmp/fr" "$MKE2FS" -q -t ext4 -L gem5pkg -E root_owner=0:0 \
    -d "$pkg" "$OUT" "$size"
"$E2FSCK" -fn "$OUT" > /dev/null
echo "package disk: $OUT ($(du -h --apparent-size "$OUT" | cut -f1)), $(python3 -I -c "
import json; m = json.load(open('$pkg/manifest.json'))
print('checkpoint', m['checkpoint'], '| ipt', m.get('ipt'))")"
