"""Change the PERSON DOWN threshold (average lying probability over the last 4 s).

    python set_down_threshold.py 0.15
Measured on 30 recorded falls: 0.15 -> 28/30 caught, ~1 false alarm/min;
0.20 -> 28/30, ~1 per 2 min; 0.50 -> 21/30, ~1 per 13 min.
"""
import pickle
import sys
import urllib.request
from pathlib import Path

path = Path(__file__).resolve().parent / "models" / "down_v1.pkl"
m = pickle.loads(path.read_bytes())
if len(sys.argv) < 2:
    sys.exit(f"current threshold: {m['threshold']}  (usage: python set_down_threshold.py 0.15)")
m["threshold"] = float(sys.argv[1])
path.write_bytes(pickle.dumps(m))
try:
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8770/v1/models/reload", data=b"{}",
                                                  method="POST"), timeout=3)
    print(f"PERSON DOWN threshold = {m['threshold']} (hub reloaded)")
except Exception:
    print(f"PERSON DOWN threshold = {m['threshold']} (start hub.py to use it)")
