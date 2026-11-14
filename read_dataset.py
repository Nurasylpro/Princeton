"""Reader for the AuraCare CSI dataset (numpy only). Ships inside the dataset zip.

    python read_dataset.py raw/p1_moving_1.bin          # prints a summary
    from read_dataset import read_raw, iq64
    for pkt in read_raw("raw/p1_moving_1.bin"): ...

raw/<chunk>.bin records:  <f64 ts_unix><u16 length><datagram>, little-endian.
Datagram with magic 0xC5110001 = RuView ADR-018 raw CSI:
  off size field
    0   4  magic (u32)        4   1  node_id          5  1  n_antennas
    6   2  n_subcarriers      8   4  freq_mhz         12 4  seq (u32, +1 per packet per node)
   16   1  rssi (i8, dBm)    17   1  noise_floor (i8) 18 1  ppdu_type   19 1 flags (bit5 = first 4 bytes zeroed)
   20   n_antennas*n_subcarriers*2 int8: ESP32 order, [imag, real] per subcarrier.
        HT packet (raw_len 256): [0:128] L-LTF, subcarriers 0..31 then -32..-1;
                                 [128:256] HT-LTF, same order.
        non-HT packet (raw_len 128): L-LTF only.
Other magics (0xC5110002 vitals, 0xC5110006 feature state, ...) are firmware side-channels; skip them.
"""
from __future__ import annotations

import struct
import sys
from collections import Counter
from typing import Iterator

import numpy as np

CSI_MAGIC = 0xC5110001
_HDR = struct.Struct("<IBBHIIbbBB")


def read_raw(path: str) -> Iterator[dict]:
    data = open(path, "rb").read()
    off = 0
    while off + 10 <= len(data):
        ts, n = struct.unpack_from("<dH", data, off)
        off += 10
        d = data[off:off + n]
        off += n
        if len(d) < _HDR.size or struct.unpack_from("<I", d)[0] != CSI_MAGIC:
            continue
        _, node, n_ant, n_sc, freq, seq, rssi, nf, ppdu, flags = _HDR.unpack_from(d)
        raw = np.frombuffer(d[20:20 + n_ant * n_sc * 2], dtype=np.int8)
        yield {"ts_unix": ts, "node_id": node, "seq": seq, "rssi_dbm": rssi, "noise_floor_dbm": nf,
               "freq_mhz": freq, "n_subcarriers": n_sc, "raw_len": len(raw),
               "first_word_sanitized": bool(flags & 0x20), "raw": raw}


def iq64(raw: np.ndarray) -> np.ndarray:
    """L-LTF part (first 128 int8) -> 64 complex values in buffer order (0..31, -32..-1)."""
    v = raw[:128].astype(np.float32)
    return v[1::2] + 1j * v[0::2]  # real is the 2nd byte, imag the 1st


if __name__ == "__main__":
    pkts = list(read_raw(sys.argv[1]))
    if not pkts:
        sys.exit("no CSI packets")
    by_node = Counter(p["node_id"] for p in pkts)
    span = pkts[-1]["ts_unix"] - pkts[0]["ts_unix"]
    print(f"{len(pkts)} CSI packets over {span:.1f} s; per node: {dict(sorted(by_node.items()))}")
    for node in sorted(by_node):
        seqs = [p["seq"] for p in pkts if p["node_id"] == node]
        lost = sum(max(0, b - a - 1) for a, b in zip(seqs, seqs[1:]))
        print(f"  node {node}: {by_node[node] / span:.1f} packets/s, seq gaps (lost packets): {lost}")
    print("first packet amplitude (64, buffer order):", np.round(np.abs(iq64(pkts[0]['raw'])), 1).tolist())
