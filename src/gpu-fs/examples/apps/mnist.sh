# PyTorch FashionMNIST inference on one batch: test_1batch/pytorch_qs_mnist.py
# from https://github.com/modori-yh/Gem5/tree/main/materials/04-GPU-model/pytorch/MNIST
# (downloaded at a pinned commit and checked):
# 11 launches. Checkpoint before the only nll_loss_forward_reduce_cuda_kernel_2d
# (launch 7): 5 kernels run after the restore, all after the last new GEMM
# shape, so gem5 does not simulate hipBLASLt's solution search.
APP_DESC="mnist: 11 launches; checkpoint before launch 7 (nll_loss_forward), 5 run after restore"
APP_KERNEL=nll_loss_forward_reduce_cuda_kernel_2d
APP_LAUNCHES=11
APP_EXPECT='Done!'
PYTHON=${PYTHON:-/usr/bin/python3}
MNIST_COMMIT=3ea9b7e3598cd15e8c0ea688fa850249eab0fad5
MNIST_URL=https://raw.githubusercontent.com/modori-yh/Gem5/$MNIST_COMMIT/materials/04-GPU-model/pytorch/MNIST/test_1batch/pytorch_qs_mnist.py
MNIST_SHA256=51d75ff0903fc8ab473c44c8e405c64b259417656f9d6e917e733bd894952b74

app_prepare() {
    local script=$W/pytorch_qs_mnist.py
    mkdir -p "$W"
    if [ ! -s "$script" ]; then
        curl -fsSL -o "$script" "$MNIST_URL" || die "downloading $MNIST_URL"
    fi
    echo "$MNIST_SHA256  $script" | sha256sum -c --quiet - || die "$script differs from the pinned version"
    "$PYTHON" - "$GPUFS/examples/mnist/requirements.txt" <<'EOF' || die "PyTorch for $PYTHON (see examples/mnist/requirements.txt)"
import sys, importlib.metadata as md
want = {}
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if "==" in line and not line.startswith("-"):
        name, ver = line.split("==")
        want[name] = ver
for name, ver in want.items():
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        sys.exit(f"{name} is not installed for {sys.executable}; install {sys.argv[1]}")
    if have != ver:
        print(f"WARN: {name} {have} for {sys.executable}, requirements.txt pins {ver}: "
              "the package will carry this torch instead of using the guest image's",
              file=sys.stderr)
EOF
    APP_CWD=$W                               # the dataset downloads to ./data
    APP_CMD=("$PYTHON" "$script")
}
