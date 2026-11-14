"""Build the zip for the teammates from one recorded session.

    python package_dataset.py S20261114_2010   ->  auracare_dataset_S20261114_2010.zip
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "data" / "sessions"

README = """# AuraCare CSI dataset: session {session}

One 2.5 x 2.5 x 2.5 m room. The transmitter is an iPhone 15 Pro hotspot at the room
centre, 0.76 m high (2.4 GHz, channel 6, 20 MHz). The receivers are ESP32-WROOM-32
DevKit V1 boards (original ESP32) in room corners, 0.40 m high: node_id 2 and 3.
`session.json` lists their exact positions and their distances to the phone. Nothing
was moved between chunks.

## Files
- `session.json`: one entry per 60 s chunk. Each entry has `state`, `people`,
  `start_unix`/`end_unix` (laptop clock), packets per node, and for prompted chunks
  `actions` (each `sit_down`/`stand_up`/`jump` command with the time it was spoken;
  people react about 0.5–1 s later).
- `raw/<chunk>.bin`: every UDP datagram exactly as received, each as
  `<f64 ts_unix><u16 len><datagram>`. See `read_dataset.py` for the byte layout.
- `decoded/<chunk>.jsonl.gz`: one JSON object per CSI packet, with the same fields as the
  live `/v1/csi` stream: ts_unix, node_id, seq, rssi_dbm, noise_floor_dbm, channel,
  freq_mhz, bandwidth_mhz, raw_len, sig_mode, first_word_sanitized and raw (int8).
- `read_dataset.py`: reader. `python read_dataset.py raw/p1_moving_1.bin` prints a summary.

## CSI layout
Each packet carries the int8 values exactly as the ESP32 reported them, as
**[imag, real]** pairs. The iPhone's packets are 802.11n (HT, 20 MHz), so most packets
have `raw_len` = 256:
- `raw[0:128]` is the L-LTF: 64 subcarriers in the order 0..31, -32..-1. This is the
  same legacy layout as a non-HT capture.
- `raw[128:256]` is the HT-LTF: 64 slots, same order.

Non-HT packets, if any, have 128 values (L-LTF only). No 64 -> 52 mapping has been
applied. When `first_word_sanitized` is true, the first 4 bytes (subcarriers 0 and 1)
were flagged invalid by the ESP32 and zeroed by the firmware.

## Sampling
About 45 packets/s per node; the firmware caps it at 50/s. Arrival is uneven, so use
`ts_unix` and `seq` (+1 per packet per node; a gap means a lost packet). There is no
on-chip timestamp in this packet format.

## Privacy
No MAC addresses are recorded. Nodes are identified only by `node_id` (2 and 3).

## Chunks
{chunks}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    args = ap.parse_args()
    sdir = SESSIONS_DIR / args.session
    meta = json.loads((sdir / "session.json").read_text())
    lines = ["| chunk | state | people | start (UTC) | seconds | packets per node | actions |",
             "|---|---|---|---|---|---|---|"]
    for c in meta["chunks"]:
        lines.append(f"| {c['id']} | {c['state']} | {c['people']} | {c['start_iso']} | "
                     f"{c['end_unix'] - c['start_unix']:.0f} | {c['packets']} | {len(c.get('actions', []))} |")
    out = ROOT / f"auracare_dataset_{args.session}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", README.format(session=args.session, chunks="\n".join(lines)))
        z.write(sdir / "session.json", "session.json")
        z.write(ROOT / "read_dataset.py", "read_dataset.py")
        z.write(ROOT / "API.md", "API.md")
        for sub in ("raw", "decoded"):
            for f in sorted((sdir / sub).glob("*")):
                z.write(f, f"{sub}/{f.name}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
