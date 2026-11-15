"""Re-calibrate the empty fingerprint to current conditions. Run before every use / demo.

    python calibrate.py        # 30 s, nobody inside the triangle
    python calibrate.py 60     # longer = steadier
"""
import json
import sys
import time
import urllib.request

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30
input(f"Make sure NOBODY is inside the triangle (crowd outside is fine). Press Enter to start {seconds:.0f} s... ")
for s in range(3, 0, -1):
    print(f"  starting in {s}", flush=True)
    time.sleep(1)
print(f"  calibrating for {seconds:.0f} s, stay out of the triangle...", flush=True)
req = urllib.request.Request("http://127.0.0.1:8770/v1/calibrate", method="POST",
                             data=json.dumps({"seconds": seconds}).encode(),
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=seconds + 15) as r:
        res = json.loads(r.read())
except urllib.error.HTTPError as e:
    res = json.loads(e.read() or b"{}")
if res.get("ok"):
    print(f"  done. packets per board {res['packets']}; fingerprint moved by {res['fingerprint_shift']}")
    print("  (a large shift means conditions had drifted a lot since training)")
    print("  STARTED OVER: room = EMPTY, all timers 0 s, no alarms. Now enter and do your test.")
else:
    print("  FAILED:", res)
