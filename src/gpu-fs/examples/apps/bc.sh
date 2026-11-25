# Pannotia BC (src/gpu/pannotia/bc) on 1k_128k.gr: 150 iterations, 1051
# launches. Checkpoint before launch 448 of backtrack_kernel: the last
# iteration's last 3 kernels run after the restore.
APP_DESC="bc: 1051 launches; checkpoint before backtrack_kernel launch 448, 3 run after restore"
APP_KERNEL=backtrack_kernel:448
APP_LAUNCHES=1051
APP_EXPECT='Completed iteration 149'
BC_INPUT_URL=http://dist.gem5.org/dist/develop/datasets/pannotia/bc/1k_128k.gr

app_prepare() {
    mkdir -p "$W"
    local src=$GR/src/gpu/pannotia/bc
    if [ ! -x "$W/bc_hip.default" ] || [ "$src/BC.cpp" -nt "$W/bc_hip.default" ]; then
        # -O1: BC has a runtime error in fgets with -O2 and -O3 (Makefile.default)
        "${ROCM_PATH:-/opt/rocm}/bin/hipcc" -O1 -Wno-unused-result -Wno-unused-value --offload-arch=gfx942 \
            -I"$src/../graph_parser" "$src/BC.cpp" -o "$W/bc_hip.default" || die "building bc"
    fi
    if [ ! -s "$W/1k_128k.gr" ]; then
        curl -fsSL -o "$W/1k_128k.gr" "$BC_INPUT_URL" || die "downloading $BC_INPUT_URL"
    fi
    APP_CWD=$W
    APP_CMD=("$W/bc_hip.default" "$W/1k_128k.gr")
}
