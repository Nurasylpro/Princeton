"""Train the PERSON DOWN model (someone lying on the floor) from fall sessions.

    python train_down.py F20261114_2052 F20261114_2132

Lying = from 3.5 s after a "fall" command until the "get up" command; upright = everything
else in fall/still chunks; the seconds around each command are left out. The live alarm is
"mean lying probability over the last SMOOTH_S seconds >= THRESHOLD".
Held-out (per session half) on the first two sessions: 25/30 lying periods caught,
0.15 false alarms per minute.
"""
import json
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

import features as feat
import train_combined as tc

W, HOP, SMOOTH_S, THRESHOLD = 3.0, 0.5, 4, 0.15


def main():
    sessions = sys.argv[1:]
    if not sessions:
        sys.exit(__doc__)
    X, Y, nodes = [], [], None
    for sess in sessions:
        S = tc.SESSIONS / sess
        meta = json.loads((S / "session.json").read_text())
        nodes = nodes or tuple(r["node_id"] for r in meta["receivers"])
        for c in meta["chunks"]:
            if c["state"] not in ("fall", "still"):
                continue
            ser = tc.load(S / "raw" / f"{c['id']}.bin")
            acts = sorted((a["t_unix"], a["action"]) for a in c.get("actions", []))
            downs, cur = [], None
            for t, a in acts:
                if a == "fall":
                    cur = t + 3.5
                elif a == "get_up" and cur is not None:
                    downs.append((cur, t + 0.5))
                    cur = None
            if cur is not None:
                downs.append((cur, c["end_unix"]))
            trans = [t for t, _ in acts]
            end = c["start_unix"] + W
            while end <= c["end_unix"]:
                if not any(-0.5 <= end - t <= 4.0 for t in trans):
                    w = {}
                    for n in nodes:
                        t, a, r = ser[n]
                        i, j = np.searchsorted(t, end - W), np.searchsorted(t, end)
                        w[n] = (t[i:j], a[i:j], r[i:j])
                    X.append(feat.down_features(w, nodes))
                    Y.append(int(any(s + W <= end <= e for s, e in downs)))
                end += HOP
    X, Y = np.array(X), np.array(Y)
    print(f"{len(Y)} windows: {Y.sum()} lying, {(Y == 0).sum()} upright")
    clf = ExtraTreesClassifier(500, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1).fit(X, Y)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with open(tc.MODELS / "down_v1.pkl", "wb") as f:
        pickle.dump({"name": "auracare-down_v1", "clf": clf, "nodes": nodes, "window_s": W,
                     "amp_mode": "lltf_ht", "smooth_s": SMOOTH_S, "threshold": THRESHOLD,
                     "sessions": sessions, "trained_at": trained_at,
                     "cv": {"caught": "25/30", "false_alarms_per_min": 0.15}}, f)
    print("Saved models/down_v1.pkl")


if __name__ == "__main__":
    main()
