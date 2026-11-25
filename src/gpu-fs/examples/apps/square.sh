# square (src/gpu/square): 51 launches of vector_square (1 warm-up + 50).
# Checkpoint before launch 50: 2 kernels run after the restore. The smoke test.
APP_DESC="square: 51 vector_square launches; checkpoint before launch 50, 2 run after restore"
APP_KERNEL=vector_square:50
APP_LAUNCHES=51
APP_EXPECT='PASSED!'

app_prepare() {
    mkdir -p "$W"
    local src=$GR/src/gpu/square
    if [ ! -x "$W/square" ] || [ "$src/square.cpp" -nt "$W/square" ]; then
        "${ROCM_PATH:-/opt/rocm}/bin/hipcc" -O3 -Wno-unused-result -Wno-unused-value --offload-arch=gfx942 -I"$src" \
            "$src/square.cpp" -o "$W/square" || die "building square"
    fi
    APP_CWD=$W
    APP_CMD=("$W/square")
}
