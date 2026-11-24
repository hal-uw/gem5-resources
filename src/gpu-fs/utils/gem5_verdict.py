#!/usr/bin/env python3
"""PASS/FAIL of a gem5 restore (gpu_checkpoint.py restore --gem5), from
gem5's outdir.

    gem5_verdict.py <outdir>

PASS when criu restored (the guest console says "criu restore exit 0"), the
app's log came out (*.after-restore), the GPU was active (shaderActiveTicks
in stats.txt) and the guest kernel logged no crash or GPU fault
(dmesg-after-restore.txt). Whether the app's own result is right is for you
to check in its log; the tail is printed. "kernels completed" counts
"Completed kernel" lines in trace.txt (debug flag GPUKernelInfo).
"""
import glob
import os
import sys


def verdict(out):
    """PASS when criu restored, the app's log came out, the GPU was active
    and the guest kernel logged no crash or GPU fault."""
    def read(f):
        try:
            with open(os.path.join(out, f), errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""
    console = read("board.pc.com_1.device")
    crit = [l for l in console.splitlines() if l.startswith("criu restore exit")]
    criu_rc = crit[-1].split()[-1] if crit else "?"
    logs = glob.glob(os.path.join(out, "*.after-restore"))
    kernels = read("trace.txt").count("Completed kernel")
    ticks = 0
    with open(os.path.join(out, "stats.txt"), errors="replace") as f:
        for l in f:
            if "shaderActiveTicks" in l:
                ticks += int(l.split()[1])
    faults = [l for l in read("dmesg-after-restore.txt").splitlines()
              if any(w in l.lower() for w in ("segfault", "fatal", "signal", "fault", "traps"))]
    for log in logs:
        print(f"--- {os.path.basename(log)} (tail)")
        print("".join(read(log).splitlines(True)[-5:]), end="")
    ok = criu_rc == "0" and logs and ticks > 0 and not faults
    print(f"{'PASS' if ok else 'FAIL'}: criu restore exit {criu_rc}, "
          f"{len(logs)} app log(s), {kernels} kernels completed, "
          f"shaderActiveTicks {ticks}, {len(faults)} fault lines in dmesg "
          f"({out})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(verdict(os.path.abspath(sys.argv[1])))
