"""Choose how PERSON DOWN decides (average lying probability over the last 4 s).

    python set_down_threshold.py rise 0.06    # jump of 6 points above your own level (default)
    python set_down_threshold.py level 0.20   # fixed level
    python set_down_threshold.py              # show current setting
Measured on 30 recorded falls (held-out):
  rise 0.06 -> 30/30 caught, ~0.5 false alarms/min, alarm after ~2.8 s
  rise 0.08 -> 29/30, ~0.3/min          level 0.20 -> 28/30, ~0.45/min
"""
import pickle
import sys
import urllib.request
from pathlib import Path

path = Path(__file__).resolve().parent / "models" / "down_v1.pkl"
m = pickle.loads(path.read_bytes())
if len(sys.argv) < 3:
    sys.exit(f"current: mode={m.get('mode', 'level')} rise={m.get('rise')} level={m.get('threshold')}\n"
             + __doc__)
mode, value = sys.argv[1], float(sys.argv[2])
if mode not in ("rise", "level"):
    sys.exit("mode must be 'rise' or 'level'")
m["mode"] = mode
m["rise" if mode == "rise" else "threshold"] = value
path.write_bytes(pickle.dumps(m))
try:
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8770/v1/models/reload", data=b"{}",
                                                  method="POST"), timeout=3)
    print(f"PERSON DOWN: mode={mode} value={value} (hub reloaded)")
except Exception:
    print(f"PERSON DOWN: mode={mode} value={value} (start hub.py to use it)")
