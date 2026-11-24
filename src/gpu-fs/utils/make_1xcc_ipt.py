#!/usr/bin/env python3
"""Make a single-XCC MI300X IP-discovery binary for the gem5 guest.

A checkpoint of one CPX partition holds a GPU with 1 XCC: 38 CUs (152 SIMDs),
4 shader arrays, 2 SDMA engines and no XGMI SDMA engines. CRIU's amdgpu plugin
restores only onto a GPU with the same simd_count, array_count,
num_sdma_xgmi_engines, ... (the KFD_*_CHECK variables do not cover these).
The gem5 guest's driver takes the XCC and SDMA counts from the IP_DISCOVERY
table that gem5 hands it (--ipt-binary), and gem5 models one XCD anyway.

This keeps the input's GC instance 0 (XCC 0) and SDMA instances 0 and 1 (on
AID 0, like XCC 0) and drops the other GC and SDMA entries. All other tables
are copied unchanged; the IP_DISCOVERY size and checksum, the table offsets
and the binary size and checksum are recomputed and verified.

Usage:
    make_1xcc_ipt.py <8-XCC MI300X ipt binary> <output>

The input is the 8-XCC table the guest otherwise gets (for example
disk-image-rocm72/mi300x_hybrid_discovery.bin). With the output, the guest
KFD reports simd_count 152, array_count 4, cu_per_simd_array 10,
num_sdma_engines 2, num_sdma_xgmi_engines 0 and num_xcc 1.
"""

import struct
import sys

HDR_FMT = "<IHHHH"  # signature, version major, minor, checksum, size
TABLE_FMT = "<HHHH"  # offset, checksum, size, padding
TABLE_BASE = struct.calcsize(HDR_FMT)
NAMES = ["IP_DISCOVERY", "GC", "HARVEST_INFO", "VCN_INFO", "MALL_INFO", "NPS"]
IPT_SIZE = 0xA00 * 4  # DISCOVERY_TMR_SIZE, the size gem5 reads
CHECKSUM_START = 10  # binary checksum covers bytes from here to the end

GC_HWID, SDMA0_HWID = 11, 42
KEEP = {GC_HWID: {0}, SDMA0_HWID: {0, 1}}
IPDS_SIGNATURE = 0x53445049  # "IPDS"
IPDS_HDR_LEN = 80  # struct ip_discovery_header (v4)


def tables(blob):
    return [struct.unpack_from(TABLE_FMT, blob, TABLE_BASE + 8 * t)[:3]
            for t in range(len(NAMES))]


def table_ok(blob, off, ck, size):
    return sum(blob[off:off + size]) & 0xFFFF == ck


def shrink_ip_discovery(t):
    """Drop the GC/SDMA entries not in KEEP from an IP_DISCOVERY table whose
    die offset has been made table-relative."""
    sig, ver, _, _, ndies = struct.unpack_from("<IHHIH", t, 0)
    assert sig == IPDS_SIGNATURE and ver == 4 and ndies == 1, (hex(sig), ver, ndies)
    b64 = t[78] & 1  # base_addr_64_bit
    _, die_off = struct.unpack_from("<HH", t, 14)
    assert die_off == IPDS_HDR_LEN, die_off
    p = IPDS_HDR_LEN
    die_id, nips = struct.unpack_from("<HH", t, p)
    p += 4
    kept = []
    for _ in range(nips):
        hw, inst, nb = struct.unpack_from("<HBB", t, p)
        n = 8 + nb * (8 if b64 else 4)
        if not (hw in KEEP and inst not in KEEP[hw]):
            kept.append(t[p:p + n])
        p += n
    out = bytearray(t[:IPDS_HDR_LEN]) + struct.pack("<HH", die_id, len(kept))
    out += b"".join(kept) + t[p:]
    struct.pack_into("<H", out, 6, len(out))  # ip_discovery_header.size
    print(f"IP_DISCOVERY: {nips} -> {len(kept)} IPs, {len(t)} -> {len(out)} bytes")
    return bytes(out)


def main(src_path, out_path):
    g = open(src_path, "rb").read()
    gt = tables(g)
    first = gt[0][0]
    out = bytearray(g[:first])
    for t in range(len(NAMES)):
        struct.pack_into(TABLE_FMT, out, TABLE_BASE + 8 * t, 0, 0, 0, 0)
    for t, (off, ck, size) in enumerate(gt):
        if not size:
            continue
        assert table_ok(g, off, ck, size), f"input {NAMES[t]} checksum"
        data = g[off:off + size]
        if NAMES[t] == "IP_DISCOVERY":
            # Die offsets are relative to the binary start; edit the table with
            # a table-relative offset, then put the original back (the table
            # stays at the same place).
            assert off == first
            d = bytearray(data)
            die_id, die_off = struct.unpack_from("<HH", d, 14)
            struct.pack_into("<HH", d, 14, die_id, die_off - off)
            d = bytearray(shrink_ip_discovery(bytes(d)))
            struct.pack_into("<HH", d, 14, die_id, die_off)
            data = bytes(d)
            ck = sum(data) & 0xFFFF
        struct.pack_into(TABLE_FMT, out, TABLE_BASE + 8 * t, len(out), ck, len(data), 0)
        out += data
    size = len(out)
    sig, vmaj, vmin, _, _ = struct.unpack_from(HDR_FMT, g, 0)
    struct.pack_into(HDR_FMT, out, 0, sig, vmaj, vmin, 0, size)
    ck = sum(out[CHECKSUM_START:size]) & 0xFFFF
    struct.pack_into(HDR_FMT, out, 0, sig, vmaj, vmin, ck, size)
    assert size <= IPT_SIZE
    out += bytes(IPT_SIZE - size)
    open(out_path, "wb").write(out)
    for t, (off, ck2, sz) in enumerate(tables(out)):
        if sz:
            assert table_ok(out, off, ck2, sz), NAMES[t]
            print(f"{NAMES[t]:13} off {off:5} size {sz:5} checksum ok")
    print(f"binary size {size}, checksum {ck:#06x}, file {len(out)} bytes")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
