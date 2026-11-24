# gem5 config wrapper: run an unmodified X86/Viper board config with a CRIU
# package disk (make_pkg_disk.sh) attached as the guest's second disk.
#   gem5.opt [gem5 options] pkg_disk_config.py <package-disk.img> \
#       <config.py> [config options...]
# The disk is added next to the root disk when the config calls
# set_kernel_disk_workload(), copy-on-write like the root disk, so the image
# file is never written. Nothing in gem5 or in the config changes.
import os
import runpy
import sys

from m5.objects import CowDiskImage, IdeDisk, RawDiskImage

from gem5.components.boards.x86_board import X86Board

if len(sys.argv) < 3:
    sys.exit(f"usage: {sys.argv[0]} <package-disk.img> <config.py> [options]")
PKG_DISK = os.path.realpath(sys.argv[1])
CONFIG = os.path.realpath(sys.argv[2])
if not os.path.isfile(PKG_DISK):
    sys.exit(f"no package disk {PKG_DISK}")

_add_disk_to_board = X86Board._add_disk_to_board


def _add_disks(self, disk_image):
    _add_disk_to_board(self, disk_image)
    disk = IdeDisk(driveID="device1")
    disk.image = CowDiskImage(child=RawDiskImage(read_only=True), read_only=False)
    disk.image.child.image_file = PKG_DISK
    ide = self.pc.south_bridge.ide
    ide.pkg_disk = disk  # a child of the controller, not an orphan
    ide.disks = [*ide.disks, disk]


X86Board._add_disk_to_board = _add_disks

sys.argv = [CONFIG, *sys.argv[3:]]
sys.path.insert(0, os.path.dirname(CONFIG))
runpy.run_path(CONFIG, run_name="__m5_main__")
