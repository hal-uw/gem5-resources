# Settings for the gpu-fs scripts. Copy to env.sh (same directory) and edit:
#   cp src/gpu-fs/env.example.sh src/gpu-fs/env.sh
# utils/preflight.sh and examples/*.sh read env.sh by themselves, so you do
# not need to source it. Variables already set in the environment win.
# Leave a value empty to use the default shown.

# --- MI300X host --------------------------------------------------------------

# PCI address of the GPU to use (see: utils/preflight.sh host). It must be
# idle: the scripts switch it to CPX and back. Default: the first GPU.
: "${GPU:=}"

# Where the examples build their apps and write checkpoints and packages.
# A checkpoint must stay at the absolute path where it was taken.
# Default: $HOME/gpu-ckpt
: "${CK:=}"

# Python for the PyTorch example (MNIST). Use the interpreter whose torch
# matches the gem5 guest image's, at the same path (examples/mnist/
# requirements.txt). Default: /usr/bin/python3
: "${PYTHON:=}"

# --- gem5 host ----------------------------------------------------------------

# gem5 source tree, built with: scons build/VEGA_X86/gem5.opt
: "${GEM5_DIR:=}"
# gem5 binary and board config. Defaults: $GEM5_DIR/build/VEGA_X86/gem5.opt,
# $GEM5_DIR/configs/example/gem5_library/x86-mi300x-gpu.py
: "${GEM5:=}"
: "${GEM5_CONFIG:=}"

# Guest disk image and its kernel (guest-image/build.sh makes both).
: "${GUEST_IMG:=}"
: "${GUEST_KERNEL:=}"

# 8-XCC MI300X IP-discovery binary (e.g. the guest image's
# /usr/lib/firmware/amdgpu/mi300_discovery, or
# src/x86-ubuntu-gpu-ml/files/mi300_discovery). examples/gem5.sh makes the
# single-XCC one from it. Default IPT1: $CK/mi300x_1xcc_discovery.bin
: "${IPT8:=}"
: "${IPT1:=}"

# Extra options: GEM5_OPTS go before the config (gem5's own), CONFIG_OPTS
# after it (the board config's).
: "${GEM5_OPTS:=}"
: "${CONFIG_OPTS:=}"
