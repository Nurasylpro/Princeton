"""Choose which trained model hub.py serves, without restarting it.

    python use_model.py            # list models, mark the active one
    python use_model.py hall_v1    # switch to models/hall_v1.pkl
"""
import json
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent / "models"
ACTIVE = MODELS / "ACTIVE"

names = sorted(p.stem for p in MODELS.glob("*.pkl"))
current = ACTIVE.read_text().strip() if ACTIVE.exists() else None
if len(sys.argv) < 2:
    for n in names:
        print(("* " if n == current else "  ") + n)
    sys.exit(0)
name = sys.argv[1]
if name not in names:
    sys.exit(f"unknown model {name!r}; available: {', '.join(names)}")
ACTIVE.write_text(name)
try:
    req = urllib.request.Request("http://127.0.0.1:8770/v1/models/reload", data=b"{}", method="POST")
    with urllib.request.urlopen(req, timeout=3) as r:
        print(f"active model: {name} | hub reloaded: {json.loads(r.read())}")
except Exception as e:
    print(f"active model: {name} (hub not reachable: {e}; it loads this model on next start)")
