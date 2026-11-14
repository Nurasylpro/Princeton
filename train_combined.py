"""Train ONE model from several sessions recorded in the same, unmoved layout.

    python train_combined.py combined_v1 P20261114_1734 H20261114_1657

Uses the best variant from the comparison: L-LTF + HT-LTF amplitudes, 5 s windows,
Extra-Trees. Does NOT change the active model; switch with `python use_model.py <name>`.
Held-out test as in train_model.py: train on all *_1 chunks, test on all *_2, and swap.
"""
from __future__ import annotations

import json
import pickle
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

import csi_format as cf
import features as feat
from plans import PLANS

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "data" / "sessions"
MODELS = ROOT / "models"
WINDOW_S, HOP_S, AMP_MODE = 5.0, 0.5, "lltf_ht"
STATES = PLANS["hall"][1]  # in0, in1_still/moving, in2_still/moving, in3p_still/moving


def load(path: Path):
    rows, data, off = {}, path.read_bytes(), 0
    while off + 10 <= len(data):
        t, n = struct.unpack_from("<dH", data, off)
        off += 10
        pkt = cf.parse_csi(data[off:off + n])
        off += n
        if pkt is not None and len(pkt.raw) >= 128:
            rows.setdefault(pkt.node_id, []).append((t, pkt.amplitudes104(), pkt.rssi))
    return {k: (np.array([r[0] for r in v]), np.stack([r[1] for r in v]), np.array([r[2] for r in v], float))
            for k, v in rows.items()}


def windows(series, t0, t1, base, nodes):
    out, end = [], t0 + WINDOW_S
    while end <= t1:
        w = {}
        for n, (t, a, r) in series.items():
            i, j = np.searchsorted(t, end - WINDOW_S), np.searchsorted(t, end)
            w[n] = (t[i:j], a[i:j], r[i:j])
        out.append(feat.window_features(w, base, nodes))
        end += HOP_S
    return out


def baselines(chunks, nodes):
    return {n: feat.node_baseline(np.concatenate([c["series"][n][1] for c in chunks if n in c["series"]]))
            for n in nodes}


def dataset(chunks, base, nodes):
    X, Y = [], []
    for c in chunks:
        w = windows(c["series"], c["start_unix"], c["end_unix"], base, nodes)
        X += w
        Y += [c["state"]] * len(w)
    return np.nan_to_num(np.array(X)), np.array(Y)


def clf():
    return ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced",
                                random_state=0, n_jobs=-1)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    name, sessions = sys.argv[1], sys.argv[2:]
    chunks, nodes = [], None
    for s in sessions:
        meta = json.loads((SESSIONS / s / "session.json").read_text())
        nodes = nodes or tuple(r["node_id"] for r in meta["receivers"])
        for c in meta["chunks"]:
            if c["state"] not in STATES:
                continue
            c = dict(c, session=s, rep=c["id"].rsplit("_", 1)[-1])
            c["series"] = load(SESSIONS / s / "raw" / f"{c['id']}.bin")
            chunks.append(c)
    counts = {}
    for c in chunks:
        counts[c["state"]] = counts.get(c["state"], 0) + 1
    print(f"sessions {sessions} | nodes {nodes} | chunks per state {counts}")

    yt, yp = [], []
    for tr_rep, te_rep in (("1", "2"), ("2", "1")):
        tr = [c for c in chunks if c["rep"] == tr_rep]
        te = [c for c in chunks if c["rep"] == te_rep]
        base = baselines([c for c in tr if c["state"] == "in0"], nodes)
        xtr, ytr = dataset(tr, base, nodes)
        xte, yte = dataset(te, base, nodes)
        model = clf().fit(xtr, ytr)
        yt += list(yte)
        yp += list(model.predict(xte))
    yt, yp = np.array(yt), np.array(yp)
    ct = np.array([STATES[s][0] for s in yt])
    cp = np.array([STATES[s][0] for s in yp])
    mt = np.array([STATES[s][1] for s in yt])
    mp = np.array([STATES[s][1] for s in yp])
    cv = {"windows": int(len(yt)),
          "people_count_accuracy": round(float((ct == cp).mean()), 3),
          "per_count": {("3+" if k == 3 else str(k)): round(float((cp[ct == k] == k).mean()), 3) for k in (0, 1, 2, 3)},
          "presence_accuracy": round(float(((ct > 0) == (cp > 0)).mean()), 3),
          "moving_accuracy": round(float((mt == mp).mean()), 3)}
    print("\nHeld-out results:")
    for k, v in cv.items():
        print(f"  {k:22s} {v}")
    labels = list(STATES)
    cm = confusion_matrix(yt, yp, labels=labels)
    print("\n  confusion (rows = truth, cols = predicted):")
    print("  " + " " * 12 + "".join(f"{l:>12s}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"  {l:>12s}" + "".join(f"{v:12d}" for v in row))

    base = baselines([c for c in chunks if c["state"] == "in0"], nodes)
    x, y = dataset(chunks, base, nodes)
    final = clf().fit(x, y)
    MODELS.mkdir(exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with open(MODELS / f"{name}.pkl", "wb") as f:
        pickle.dump({"name": f"auracare-{name}", "plan": "hall", "clf": final, "baselines": base,
                     "nodes": nodes, "feature_names": feat.feature_names(nodes), "state_map": STATES,
                     "amp_mode": AMP_MODE, "window_s": WINDOW_S, "trained_at": trained_at,
                     "sessions": sessions, "cv": cv}, f)
    print(f"\nSaved models/{name}.pkl (NOT activated). Use it with:  python use_model.py {name}")


if __name__ == "__main__":
    main()
