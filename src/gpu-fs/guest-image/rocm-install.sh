#!/bin/bash

# Copyright (c) 2024 The Regents of the University of California.
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD 3-Clause

# gpu-fs variant of src/x86-ubuntu-gpu-ml/scripts/rocm-install.sh, for
# restoring CRIU checkpoints taken on an MI300X host (src/gpu-fs). It differs
# from the stock script in four ways, marked "gpu-fs:" below:
#   - ROCm 7.2.4 and the amdgpu 31.60 driver repo, as on the MI300X host
#   - the runtime libraries of criu and amdgpu_plugin_gem5 (the package disk
#     brings criu itself)
#   - PyTorch pinned to the build in src/gpu-fs/examples/mnist/requirements.txt,
#     installed with the system Python like on the host, so the package disk
#     can leave torch out
#   - python3-pip instead of the nonexistent package pip3
# guest-image/build.sh runs the stock packer build with this script.

# Installing the packages in this script instead of the user-data
# file dueing ubuntu autoinstall. The reason is that sometimes
# the package install failes. This method is more reliable.
echo 'installing packages'
apt-get update
apt-get install -y scons
apt-get install -y git
apt-get install -y vim
apt-get install -y build-essential

# Remove the motd
rm /etc/update-motd.d/*

# Build the m5 util
git clone https://github.com/gem5/gem5.git --depth=1 --filter=blob:none --no-checkout --sparse --single-branch --branch=stable
pushd gem5
# Checkout just the files we need
git sparse-checkout add util/m5
git sparse-checkout add include
git checkout
# Build the library and binary
pushd util/m5
scons build/x86/out/m5
cp build/x86/out/m5 /sbin/m5
popd
popd
rm -rf gem5

# Occasionally connecting to github fails. Bail now instead of making a disk
# image that is not usable.
if [ ! -f /sbin/m5 ]; then
    echo "m5 util did not appear to build correctly. This disk will not be usable."
    echo "Try to build the disk again if there was a temporary error (e.g., not able to connect to github)."
    echo "For other problems create an issue at https://github.com/gem5/gem5/issues."
    exit 1
fi

# Make the script to load amdgpu module with the above module executable
chmod a+x /home/gem5/load_amdgpu.sh

# gpu-fs: the MI300X host's versions (preflight.sh host prints them).
ROCM_VERSION=7.2.4
AMDGPU_REPO=31.60

# The following instructions were obtained from the ROCm installation guide:
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/
#     native-install/ubuntu.html

# Make the directory if it doesn't exist yet.
# This location is recommended by the distribution maintainers.
sudo mkdir --parents --mode=0755 /etc/apt/keyrings

# Download the key, convert the signing-key to a full
# keyring required by apt and store in the keyring directory
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/${AMDGPU_REPO} noble main" \
        | sudo tee /etc/apt/sources.list.d/amdgpu.list

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
        | sudo tee --append /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
        | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt update

sudo apt -y install amdgpu-dkms
sudo apt -y install rocm
sudo apt -y install cmake

# gpu-fs: runtime libraries of criu and amdgpu_plugin_gem5 (ldd of both).
sudo apt -y install libprotobuf-c1 libnl-3-200 libnet1 libbsd0 libgnutls30t64 \
    libnftables1 libdrm2 libdrm-amdgpu1 python3

# Make directory for GPU BIOS. These are placed in /root for compatibility with
# the legacy GPUFS configs.
sudo mkdir -p /root/roms
sudo chmod 777 /root
sudo chmod 777 /root/roms

# File describing the hardware topology of the GPU. Used starting with MI300X.
# Change permission so packer can copy here after disk build.
sudo touch /usr/lib/firmware/amdgpu/ip_discovery.bin
sudo chmod 777 /usr/lib/firmware/amdgpu/ip_discovery.bin


# Install a known-working version of Linux as this might change after stable
# release. Install this after DKMS so they are rebuilt.
KERNEL=6.8.0-124-generic

sudo apt -y install "linux-image-${KERNEL}"
sudo apt -y install "linux-headers-${KERNEL}" "linux-modules-extra-${KERNEL}"

echo "Extracting linux kernel"
sudo bash -c "/usr/src/linux-headers-${KERNEL}/scripts/extract-vmlinux /boot/vmlinuz-${KERNEL} > /home/gem5/vmlinux-gpu-ml"

# Build the gem5 Linux module containing symbols missing due to outdated ACPI support in gem5
pushd /home/gem5
make -C /lib/modules/${KERNEL}/build M=${PWD}
if [ ! -f ./gem5_wmi.ko ]; then
    echo "gem5_wmi module did not appear to build correctly. This disk will not be usable."
    echo "Please post an issue at https://github.com/gem5/gem5/issues with this output log."
    exit 1
fi
popd

# Make the discovery files writeable by packer
touch /usr/lib/firmware/amdgpu/mi300_discovery
touch /usr/lib/firmware/amdgpu/mi350_discovery

chmod 777 /usr/lib/firmware/amdgpu/mi300_discovery
chmod 777 /usr/lib/firmware/amdgpu/mi350_discovery

# Note about pip: This disk is created for the express purpose of being run in
# gem5 and is therefore effectively sandboxed enough that we can use the pip
# option --break-system-packages. If you plan to modify this disk image with
# pip packages that might conflict, it is up to you to resolve the conflicts.

# gpu-fs: the PyTorch build of src/gpu-fs/examples/mnist/requirements.txt
# (guest-image/build.sh checks that the two agree), for the system Python, as
# on the MI300X host: /usr/local/lib/python3.12/dist-packages.
TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
TORCH=torch==2.12.1+rocm7.2
TORCHVISION=torchvision==0.27.1+rocm7.2
sudo apt -y install python3-pip
sudo /usr/bin/python3 -m pip install --break-system-packages --no-cache-dir \
    --index-url "$TORCH_INDEX" "$TORCH" "$TORCHVISION"

# Setup gem5 auto login.
mv /home/gem5/serial-getty@.service /lib/systemd/system/

echo -e "\n/home/gem5/run_gem5_app.sh\n" >> /root/.bashrc
