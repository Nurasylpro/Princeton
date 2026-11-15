"""Train the fall detector from a recorded fall session.

    python train_fall.py F20261114_1900

Labels come from the spoken commands logged in session.json: a 3 s window counts as
"fall" when it ends 1-3 s after a "fall" command (reaction + the fall itself). Windows just
around that range are left out (unclear). Everything else, including fast sitting, lying
down slowly and bending, is "no fall".

Evaluation is per EVENT on chunks the model did not train on (odd vs even repetitions):
  - caught: falls with a detection within 0.5-4 s after the command
  - false alarms: detections anywhere else, per minute, listed per activity
The decision threshold is chosen from those numbers and stored with the model.
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import features as feat
import train_combined as tc

HOP_S = 0.25
POS = (1.0, 3.0)        # window end - command time -> "fall"
IGNORE = (0.0, 4.0)     # anything else inside this range is ambiguous and skipped
CATCH = (0.5, 4.0)      # a detection here counts as catching the fall
MERGE_S = 2.0           # detections closer than this are one alarm
import os
POSITIVE = set(os.environ.get("FALL_POSITIVE", "fall").split(","))  # e.g. "fall,lie_down" = went down to the floor


def chunk_windows(c, nodes):
    falls = [a["t_unix"] for a in c.get("actions", []) if a["action"] in POSITIVE]
    X, y, ends = [], [], []
    end = c["start_unix"] + feat.FALL_WINDOW_S
    while end <= c["end_unix"]:
        rel = [end - f for f in falls]
        if any(POS[0] <= r <= POS[1] for r in rel):
            lab = 1
        elif any(IGNORE[0] <= r <= IGNORE[1] for r in rel):
            lab = None
        else:
            lab = 0
        w = {}
        for n, (t, a, r) in c["series"].items():
            i, j = np.searchsorted(t, end - feat.FALL_WINDOW_S), np.searchsorted(t, end)
            w[n] = (t[i:j], a[i:j], r[i:j])
        X.append(feat.fall_features(w, nodes))
        y.append(lab)
        ends.append(end)
        end += HOP_S
    return np.nan_to_num(np.array(X)), y, np.array(ends), falls


def events(ends, prob, thr):
    hits = ends[prob >= thr]
    out = []
    for t in hits:
        if not out or t - out[-1][-1] > MERGE_S:
            out.append([t])
        else:
            out[-1].append(t)
    return [e[0] for e in out]  # time of first detection of each alarm


def clf():
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                random_state=0, n_jobs=-1)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    session = sys.argv[1]
    meta = json.loads((tc.SESSIONS / session / "session.json").read_text())
    nodes = tuple(r["node_id"] for r in meta["receivers"])
    chunks = []
    for c in meta["chunks"]:
        c = dict(c)
        c["series"] = tc.load(tc.SESSIONS / session / "raw" / f"{c['id']}.bin")
        c["fold"] = int(c["id"].rsplit("_", 1)[-1]) % 2  # odd vs even repetition
        c["win"] = chunk_windows(c, nodes)
        chunks.append(c)
    n_falls = sum(len(c["win"][3]) for c in chunks)
    print(f"{len(chunks)} chunks, {n_falls} logged falls, nodes {nodes}")

    # held-out probabilities for every chunk
    for fold in (0, 1):
        tr = [c for c in chunks if c["fold"] != fold]
        X = np.vstack([c["win"][0][[l is not None for l in c["win"][1]]] for c in tr])
        y = np.concatenate([[l for l in c["win"][1] if l is not None] for c in tr]).astype(int)
        m = clf().fit(X, y)
        for c in chunks:
            if c["fold"] == fold:
                c["prob"] = m.predict_proba(c["win"][0])[:, 1]

    print("\nHeld-out, per event (threshold sweep):")
    print("  thr   caught        false alarms/min   per activity")
    best = None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        caught, fa, minutes, per = 0, 0, 0.0, {}
        for c in chunks:
            _, _, ends, falls = c["win"]
            alarms = events(ends, c["prob"], thr)
            minutes += (c["end_unix"] - c["start_unix"]) / 60
            caught += sum(any(CATCH[0] <= a - f <= CATCH[1] for a in alarms) for f in falls)
            false = [a for a in alarms if not any(-0.5 <= a - f <= CATCH[1] + 1 for f in falls)]
            fa += len(false)
            per[c["state"]] = per.get(c["state"], 0) + len(false)
        rate = fa / minutes
        print(f"  {thr:.1f}  {caught:2d}/{n_falls} ({caught / max(n_falls, 1):4.0%})   {rate:5.2f}            {per}")
        score = caught / max(n_falls, 1) - 0.5 * rate
        if best is None or score >= best[0]:  # ties -> higher threshold, fewer false alarms
            best = (score, thr, caught, rate)
    _, thr, caught, rate = best
    print(f"\nchosen threshold {thr}: caught {caught}/{n_falls}, {rate:.2f} false alarms per minute")

    X = np.vstack([c["win"][0][[l is not None for l in c["win"][1]]] for c in chunks])
    y = np.concatenate([[l for l in c["win"][1] if l is not None] for c in chunks]).astype(int)
    final = clf().fit(X, y)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with open(tc.MODELS / "fall_v1.pkl", "wb") as f:
        pickle.dump({"name": "auracare-fall_v1", "clf": final, "threshold": thr, "nodes": nodes,
                     "amp_mode": "lltf_ht", "window_s": feat.FALL_WINDOW_S, "session": session,
                     "trained_at": trained_at,
                     "cv": {"caught": caught, "falls": n_falls, "false_alarms_per_min": round(rate, 3)}}, f)
    print("Saved models/fall_v1.pkl (the hub loads it automatically on reload/restart)")


if __name__ == "__main__":
    main()
