"""Plain-language live view of the trained model (reads hub.py's /v1/events once a second).

    python live.py
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8770/v1/events/latest"
STATE_TEXT = {"empty": "EMPTY ROOM", "p1_still": "1 PERSON, STILL", "p1_moving": "1 PERSON, MOVING",
              "p2_still": "2 PEOPLE, STILL", "p2_moving": "2 PEOPLE, MOVING",
              # hall plan: people INSIDE the triangle
              "in0": "NOBODY INSIDE", "in1_still": "1 INSIDE, STILL", "in1_moving": "1 INSIDE, MOVING",
              "in2_still": "2 INSIDE, STILL", "in2_moving": "2 INSIDE, MOVING",
              "in3p_still": "3+ INSIDE, STILL", "in3p_moving": "3+ INSIDE, MOVING"}

while True:
    try:
        with urllib.request.urlopen(URL, timeout=2) as r:
            e = json.loads(r.read())
        p = e.get("people")
        if e.get("no_data_nodes"):
            boards = ", ".join(str(n) for n in e["no_data_nodes"])
            line = (f"NO DATA from ESP {boards}: is the laptop on the 'iPhone' hotspot "
                    f"and are the boards powered?")
        elif p is None:
            line = "no trained model loaded (run train_model.py)"
        else:
            bar = "#" * int(20 * e["motion"]["score"])
            o = e.get("occupancy") or {}
            occ = (f"OCCUPIED ({o['count']}) {o['seconds']:>4}s" if o.get("occupied") else f"EMPTY {o.get('seconds', 0):>4}s")
            line = (f"{occ:<20} | model: {STATE_TEXT.get(p['state'], p['state']):<18} "
                    f"people={p['count']} ({p['probs'].get(p['count'], 0):.0%})  "
                    f"moving={p['moving_prob']:.0%} [{bar:<20}]  {o.get('reason', '')}")
        dn = e.get("down")
        if dn and dn.get("person_down"):
            line = f"!!! PERSON DOWN for {dn['seconds']}s !!!  " + line
        elif dn:
            line += f"  down={dn['smoothed']:.0%}"
        fl = e.get("fall")
        if fl and fl.get("alert"):
            line = f"!!! FALL DETECTED {fl['seconds_since']}s ago !!!  " + line
        elif fl:
            line += f"  fall={fl['prob']:.0%}"
        print(time.strftime("%H:%M:%S"), line, flush=True)
    except Exception as ex:
        print(time.strftime("%H:%M:%S"), f"hub not reachable: {ex}", flush=True)
    time.sleep(1)
