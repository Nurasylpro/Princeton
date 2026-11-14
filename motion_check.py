"""Guided motion test: prompts STILL -> WALK -> STILL and logs what the system sees.

Run from this folder while hub.py and the RuView container are up:
    python motion_check.py            # 30 s per phase
    python motion_check.py --phase 20 # shorter phases

Each second it prints the firmware motion score per node (via hub.py :8770) and
RuView's per-node label (:3000), writes a CSV to data/, and ends with a summary:
time-to-detect after WALK starts, time-to-idle after STOP, and score per phase.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HUB = "http://127.0.0.1:8770/v1/events/latest"
RUVIEW = "http://127.0.0.1:3000/api/v1/sensing/latest"


def get(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as r:
            return json.loads(r.read())
    except Exception:
        return None


def phase_at(t: float, p: float) -> str:
    return "STILL" if t < p else "WALK" if t < 2 * p else "STOP" if t < 3 * p else "END"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=float, default=30.0, help="seconds per phase")
    args = ap.parse_args()
    p = args.phase

    out = Path(__file__).parent / "data" / f"motion_check_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    rows: list[dict] = []
    prompts = {"STILL": "Step away from the laptop and stay completely still (or leave the room).",
               "WALK": ">>> WALK NOW: back and forth between the iPhone and the boards. <<<",
               "STOP": ">>> STOP. Stand or sit completely still. <<<"}
    print(f"Logging to {out}\nStarting in 5 s...")
    time.sleep(5)

    t0, last_phase, bad_parser = time.time(), None, False
    while (t := time.time() - t0) < 3 * p:
        ph = phase_at(t, p)
        if ph != last_phase:
            print(f"\n=== {ph} ===  {prompts[ph]}\n")
            last_phase = ph
        hub, rv = get(HUB), get(RUVIEW)
        m = (hub or {}).get("motion") or {}
        per_node = m.get("firmware_per_node") or m.get("per_node") or {}
        if any(v > 1.0 for v in per_node.values()):
            bad_parser = True
        rv_nodes = {str(n["node_id"]): n for n in (rv or {}).get("node_features", [])}
        row = {"t": round(t, 1), "phase": ph, "hub_state": m.get("state"), "hub_score": m.get("score")}
        for nid in sorted(set(per_node) | set(rv_nodes)):
            row[f"score_n{nid}"] = per_node.get(nid)
            rn = rv_nodes.get(nid, {})
            row[f"rv_label_n{nid}"] = (rn.get("classification") or {}).get("motion_level")
            row[f"rv_motion_power_n{nid}"] = round((rn.get("features") or {}).get("motion_band_power", 0), 1)
        rows.append(row)
        nodes_txt = "  ".join(
            f"n{nid}: score={row.get(f'score_n{nid}')!s:>6} rv={row.get(f'rv_label_n{nid}')!s:<15}"
            for nid in sorted(set(per_node) | set(rv_nodes)))
        print(f"{t:5.1f}s {ph:<5} hub={m.get('state')!s:<6} {nodes_txt}")
        time.sleep(max(0.0, 1.0 - (time.time() - t0 - t)))

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("t", "phase", "hub_state", "hub_score"), k))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("\n=== Summary ===")
    if bad_parser:
        print("WARNING: motion scores > 1 seen: hub.py is still running the old parser. Restart it.")

    def moving(r: dict) -> bool:
        return r.get("hub_state") == "motion"

    for ph in ("STILL", "WALK", "STOP"):
        rs = [r for r in rows if r["phase"] == ph]
        if rs:
            frac = sum(moving(r) for r in rs) / len(rs)
            scores = [r["hub_score"] for r in rs if isinstance(r.get("hub_score"), (int, float))]
            avg = sum(scores) / len(scores) if scores else float("nan")
            print(f"{ph:<5}: hub says motion {frac:4.0%} of the time, mean score {avg:.3f}")
    walk = [r for r in rows if r["phase"] == "WALK" and moving(r)]
    print(f"Time to detect walking: {walk[0]['t'] - p:.1f} s" if walk else "Walking was NEVER detected.")
    stop = [r for r in rows if r["phase"] == "STOP"]
    idle = next((r for r in stop if not moving(r)), None)
    print(f"Time back to idle after STOP: {idle['t'] - 2 * p:.1f} s" if idle else "Never returned to idle.")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
