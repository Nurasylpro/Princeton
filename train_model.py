"""Train our room-state model (people 0/1/2 x still/moving) and RuView's activity classifier.

    python train_model.py S20261114_2010

Honest evaluation: every state was recorded twice (chunks *_1 and *_2). We train on all
*_1 chunks and test on all *_2 chunks, then swap, so every score comes from a minute the
model never saw. The empty-room baseline is also taken only from the training half.
"""
from __future__ import annotations

import argparse
import json
import pickle
import struct
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import csi_format as cf
import features as feat

ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "data" / "sessions"
MODEL_PATH = ROOT / "models" / "count_v1.pkl"
STATES = {"empty": (0, False), "p1_still": (1, False), "p1_moving": (1, True),
          "p2_still": (2, False), "p2_moving": (2, True)}
HOP_S = 0.5
NODES_USED: tuple[int, ...] = feat.NODES  # set from session.json in main()


def load_chunk(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows: dict[int, list] = {}
    data = path.read_bytes()
    off = 0
    while off + 10 <= len(data):
        t, n = struct.unpack_from("<dH", data, off)
        off += 10
        pkt = cf.parse_csi(data[off:off + n])
        off += n
        if pkt is not None and len(pkt.raw) >= 128:
            rows.setdefault(pkt.node_id, []).append((t, pkt.amplitudes52(), pkt.rssi))
    out = {}
    for node, r in rows.items():
        r.sort(key=lambda x: x[0])
        out[node] = (np.array([x[0] for x in r]), np.stack([x[1] for x in r]),
                     np.array([x[2] for x in r], dtype=np.float64))
    return out


def windows(series, t0: float, t1: float, baselines) -> np.ndarray:
    feats = []
    end = t0 + feat.WINDOW_S
    while end <= t1:
        win = {}
        for node, (t, a, r) in series.items():
            i, j = np.searchsorted(t, end - feat.WINDOW_S), np.searchsorted(t, end)
            win[node] = (t[i:j], a[i:j], r[i:j])
        feats.append(feat.window_features(win, baselines, NODES_USED))
        end += HOP_S
    return np.array(feats)


def baselines_from(chunks: list[dict]) -> dict[int, np.ndarray]:
    per_node: dict[int, list] = {}
    for c in chunks:
        for node, (_, a, _) in c["series"].items():
            per_node.setdefault(node, []).append(a)
    return {n: feat.node_baseline(np.concatenate(v)) for n, v in per_node.items()}


def dataset(chunks: list[dict], baselines) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for c in chunks:
        x = windows(c["series"], c["start_unix"], c["end_unix"], baselines)
        xs.append(x)
        ys += [c["state"]] * len(x)
    return np.vstack(xs), np.array(ys)


def make_clf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                  random_state=0, n_jobs=-1)


def post(url: str, timeout: float = 60) -> dict:
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    args = ap.parse_args()
    sdir = SESSIONS_DIR / args.session
    meta = json.loads((sdir / "session.json").read_text())
    global NODES_USED
    NODES_USED = tuple(r["node_id"] for r in meta.get("receivers", [])) or feat.NODES
    print(f"receivers used for training: {NODES_USED}")

    chunks = []
    for c in meta["chunks"]:
        if c["state"] not in STATES:
            continue  # teammates' prompted chunks are not training data for this model
        c["series"] = load_chunk(sdir / "raw" / f"{c['id']}.bin")
        c["rep"] = c["id"].rsplit("_", 1)[-1]
        chunks.append(c)
        counts = {n: len(s[0]) for n, s in sorted(c["series"].items())}
        print(f"  {c['id']:14s} {c['state']:10s} packets per node {counts}")
    missing = sorted(set(STATES) - {c["state"] for c in chunks})
    if missing:
        sys.exit(f"missing states {missing}: record them first (python session.py --resume {args.session})")

    # ---- honest 2-fold evaluation by repetition --------------------------------------------
    y_true, y_pred = [], []
    for train_rep, test_rep in (("1", "2"), ("2", "1")):
        tr = [c for c in chunks if c["rep"] == train_rep]
        te = [c for c in chunks if c["rep"] == test_rep]
        if not tr or not te:
            continue
        base = baselines_from([c for c in tr if c["state"] == "empty"])
        xtr, ytr = dataset(tr, base)
        xte, yte = dataset(te, base)
        clf = make_clf().fit(xtr, ytr)
        y_true += list(yte)
        y_pred += list(clf.predict(xte))
    labels = list(STATES)
    yt, yp = np.array(y_true), np.array(y_pred)
    count_t = np.array([STATES[s][0] for s in yt])
    count_p = np.array([STATES[s][0] for s in yp])
    move_t = np.array([STATES[s][1] for s in yt])
    move_p = np.array([STATES[s][1] for s in yp])
    cv = {"windows": int(len(yt)),
          "state_accuracy": round(float((yt == yp).mean()), 3),
          "people_count_accuracy": round(float((count_t == count_p).mean()), 3),
          "moving_accuracy": round(float((move_t == move_p).mean()), 3),
          "presence_accuracy": round(float(((count_t > 0) == (count_p > 0)).mean()), 3)}
    print("\nHeld-out results (trained on one repetition, tested on the other):")
    for k, v in cv.items():
        print(f"  {k:22s} {v}")
    cm = confusion_matrix(yt, yp, labels=labels)
    print("\n  confusion (rows = truth, cols = predicted):")
    print("  " + " " * 11 + "".join(f"{l:>11s}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"  {l:>11s}" + "".join(f"{v:11d}" for v in row))

    # ---- final model on everything -----------------------------------------------------------
    base = baselines_from([c for c in chunks if c["state"] == "empty"])
    x, y = dataset(chunks, base)
    clf = make_clf().fit(x, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"name": "auracare-state-v1", "clf": clf, "baselines": base,
                     "nodes": NODES_USED, "feature_names": feat.feature_names(NODES_USED),
                     "state_map": STATES,
                     "trained_at": trained_at, "session": args.session, "cv": cv}, f)
    (sdir / "model_report.json").write_text(json.dumps({"trained_at": trained_at, "cv": cv,
                                                         "confusion": {"labels": labels, "matrix": cm.tolist()}},
                                                        indent=2))
    print(f"\nSaved {MODEL_PATH}")

    # ---- RuView's own activity classifier (empty / still / moving) ---------------------------
    try:
        r = post("http://127.0.0.1:3000/api/v1/adaptive/train", timeout=120)
        print("RuView adaptive classifier:", json.dumps(r)[:400])
    except Exception as e:
        print(f"RuView adaptive training failed: {e}")
    try:
        print("Hub reload:", post("http://127.0.0.1:8770/v1/models/reload"))
    except Exception as e:
        print(f"Hub not reachable for reload ({e}); restart hub.py to load the model.")


if __name__ == "__main__":
    main()
