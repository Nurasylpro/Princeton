"""Plain-language live view of the trained model (reads hub.py's /v1/events once a second).

    python live.py
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:8770/v1/events/latest"
STATE_TEXT = {"empty": "EMPTY ROOM", "p1_still": "1 PERSON, STILL", "p1_moving": "1 PERSON, MOVING",
              "p2_still": "2 PEOPLE, STILL", "p2_moving": "2 PEOPLE, MOVING"}

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
            line = (f"{STATE_TEXT.get(p['state'], p['state']):<18} "
                    f"people={p['count']} ({p['probs'].get(p['count'], 0):.0%})  "
                    f"moving={p['moving_prob']:.0%} [{bar:<20}]")
        print(time.strftime("%H:%M:%S"), line, flush=True)
    except Exception as ex:
        print(time.strftime("%H:%M:%S"), f"hub not reachable: {ex}", flush=True)
    time.sleep(1)
