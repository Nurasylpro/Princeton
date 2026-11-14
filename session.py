"""Guided recording session: run this, follow the screen (and the voice).

    python session.py                 # new session
    python session.py --resume S...   # continue an interrupted session (skips finished chunks)

Needs hub.py and the RuView container running. Each 60 s chunk is recorded twice:
by the hub (raw + decoded CSI for our model and the teammates' dataset) and by RuView
(its own feature recording, used to train RuView's activity classifier).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import winsound
except ImportError:  # not on Windows
    winsound = None

ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "data" / "sessions"
HUB = "http://127.0.0.1:8770"
RUVIEW = "http://127.0.0.1:3000"
CHUNK_S = 60

# ---- room layout: see config.py (metres; origin = a corner without a board) ---------------
import config

ROOM = {"size_m": list(config.ROOM_SIZE_M), "door": "keep the same (open/closed) for recording and demo"}
PHONE = {"name": "iPhone 15 Pro hotspot (transmitter)", "pos_m": list(config.PHONE_POS_M)}
NODES = {n: list(p) for n, p in config.RECEIVERS_M.items()}


@dataclass
class Chunk:
    id: str
    people: int
    state: str               # empty | p1_still | p1_moving | p2_still | p2_moving | (teammates) p1_sitstand | p1_jump
    say: str                 # spoken + shown instruction
    ruview_id: str | None    # RuView recording id (train_* -> activity classifier), None = not used there
    prompts: list[str] = field(default_factory=list)  # repeated spoken actions during the chunk
    teammates_block: bool = False


PLAN: list[Chunk] = [
    Chunk("empty_1", 0, "empty", "Nobody in the room.", "train_empty_1"),
    Chunk("empty_2", 0, "empty", "Nobody in the room.", "train_empty_2"),
    Chunk("p1_still_1", 1, "p1_still", "One person. Sit quietly on the chair.", "train_still_p1_1"),
    Chunk("p1_still_2", 1, "p1_still", "One person. Stand quietly.", "train_still_p1_2"),
    Chunk("p1_moving_1", 1, "p1_moving", "One person. Walk around, turn, wave your arms.", "train_moving_p1_1"),
    Chunk("p1_moving_2", 1, "p1_moving", "One person. Keep moving: walk, turn, wave.", "train_moving_p1_2"),
    Chunk("p2_still_1", 2, "p2_still", "Two people. Both sit or stand quietly.", "train_still_p2_1"),
    Chunk("p2_still_2", 2, "p2_still", "Two people. Both stay still, swap positions first.", "train_still_p2_2"),
    Chunk("p2_moving_1", 2, "p2_moving", "Two people. Both move around.", "train_moving_p2_1"),
    Chunk("p2_moving_2", 2, "p2_moving", "Two people. Both keep moving.", "train_moving_p2_2"),
    Chunk("p1_sitstand_1", 1, "p1_sitstand", "One person, standing next to the chair. Follow the commands.",
          None, ["sit down", "stand up"], True),
    Chunk("p1_sitstand_2", 1, "p1_sitstand", "One person, standing next to the chair. Follow the commands.",
          None, ["sit down", "stand up"], True),
    Chunk("p1_jump_1", 1, "p1_jump", "One person, standing. Jump once on each command.",
          None, ["jump"], True),
]


# ---- helpers ------------------------------------------------------------------------------
class Voice:
    """Persistent PowerShell speech synthesizer (English voice if installed)."""

    SCRIPT = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Volume = 100;"
        "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'en-*' } | Select-Object -First 1;"
        "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) };"
        "while ($null -ne ($l = [Console]::In.ReadLine())) { $s.SpeakAsyncCancelAll(); [void]$s.SpeakAsync($l) }"
    )

    def __init__(self) -> None:
        try:
            self.p = subprocess.Popen(["powershell", "-NoProfile", "-Command", self.SCRIPT],
                                      stdin=subprocess.PIPE, text=True)
        except OSError:
            self.p = None

    def say(self, text: str) -> None:
        if self.p and self.p.poll() is None:
            self.p.stdin.write(text.replace("\n", " ") + "\n")
            self.p.stdin.flush()

    def close(self) -> None:
        if self.p and self.p.poll() is None:
            self.p.stdin.close()


def beep(freq: int = 880, ms: int = 250) -> None:
    if winsound:
        winsound.Beep(freq, ms)


def call(method: str, url: str, body: dict | None = None, timeout: float = 5.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}") | {"http_status": e.code}


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def dist(a, b) -> float:
    return round(math.dist(a, b), 2)


def preflight() -> bool:
    try:
        st = call("GET", f"{HUB}/v1/status")
    except Exception as e:
        print(f"  hub not reachable at {HUB}: {e}\n  Start it first:  python hub.py")
        return False
    ok = True
    for n in [x for x in st["nodes"] if x["node_id"] in NODES]:
        mark = "OK " if n["healthy"] else "!! "
        ok &= n["healthy"]
        print(f"  {mark} ESP {n['node_id']}: {n['pps']:5.1f} packets/s   rssi {n['rssi_dbm']}")
    try:
        h = call("GET", f"{RUVIEW}/health")
        print(f"  {'OK ' if h.get('status') == 'ok' else '!! '} RuView server: {h.get('processing', {}).get('state')}")
    except Exception:
        print("  !!  RuView server not reachable on :3000 (docker start ruview)")
        ok = False
    return ok


def save(meta: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(path)


def record_chunk(chunk: Chunk, session: str, voice: Voice) -> dict:
    call("POST", f"{RUVIEW}/api/v1/recording/stop")  # defensive: nothing may be left running
    actions: list[dict] = []
    prompt_times: list[float] = []
    if chunk.prompts:
        t = 3.0
        while t < CHUNK_S - 4:
            prompt_times.append(t)
            t += random.uniform(5.0, 7.0)

    rec = call("POST", f"{HUB}/v1/record/start", {"session": session, "chunk": chunk.id})
    if not rec.get("ok"):
        raise RuntimeError(f"hub refused to record: {rec}")
    if chunk.ruview_id:
        r = call("POST", f"{RUVIEW}/api/v1/recording/start",
                 {"id": chunk.ruview_id, "session_name": chunk.ruview_id, "label": chunk.state})
        if r.get("success") is False or r.get("error"):
            print(f"  (RuView recording not started: {r})")
    start = time.time()
    beep(1200, 300)
    voice.say("Start.")
    next_prompt, k = 0, 0
    try:
        while (el := time.time() - start) < CHUNK_S:
            if next_prompt < len(prompt_times) and el >= prompt_times[next_prompt]:
                action = chunk.prompts[k % len(chunk.prompts)]
                actions.append({"t_unix": round(time.time(), 3), "t_iso": iso(time.time()),
                                "action": action.replace(" ", "_"), "spoken": action})
                voice.say(action.capitalize() + ".")
                next_prompt, k = next_prompt + 1, k + 1
            filled = int(30 * el / CHUNK_S)
            sys.stdout.write(f"\r  recording [{'#' * filled}{'.' * (30 - filled)}] {CHUNK_S - el:4.0f} s left ")
            sys.stdout.flush()
            time.sleep(0.05)
    finally:
        end = time.time()
        stats = call("POST", f"{HUB}/v1/record/stop")
        if chunk.ruview_id:
            call("POST", f"{RUVIEW}/api/v1/recording/stop")
    beep(600, 400)
    voice.say("Done.")
    print()
    return {"start_unix": round(start, 3), "end_unix": round(end, 3), "start_iso": iso(start),
            "end_iso": iso(end), "packets": stats.get("packets", {}), "actions": actions}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", help="session id to continue")
    args = ap.parse_args()

    session = args.resume or datetime.now().strftime("S%Y%m%d_%H%M")
    sdir = SESSIONS_DIR / session
    sdir.mkdir(parents=True, exist_ok=True)
    meta_path = sdir / "session.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {
            "session": session, "created_iso": iso(time.time()),
            "room": ROOM, "transmitter": PHONE,
            "receivers": [{"node_id": n, "board": config.BOARD, "pos_m": p,
                           "distance_to_phone_m": dist(p, PHONE["pos_m"])} for n, p in NODES.items()],
            "radio": {"channel": 6, "bandwidth_mhz": 20, "csi": "L-LTF, 64 subcarriers, 128 int8 [imag, real]",
                      "nominal_rate_hz": "~45 per node (firmware cap 50), uneven: use ts_unix and seq"},
            "firmware": "RuView esp32-csi-node 0.8.18, local branch esp32-classic",
            "chunks": [],
        }
        save(meta, meta_path)
    done = {c["id"] for c in meta["chunks"]}

    print(f"\n=== AuraCare recording session {session} ===")
    for n in meta["receivers"]:
        print(f"  ESP {n['node_id']} at {n['pos_m']} m, {n['distance_to_phone_m']} m from the phone")
    print("\nChecking boards...")
    while not preflight():
        ans = input("\nNot everything is healthy. [Enter] re-check, [c] continue anyway, [q] quit: ").strip().lower()
        if ans == "c":
            break
        if ans == "q":
            return

    voice = Voice()
    asked_team = False
    todo = [c for c in PLAN if c.id not in done]
    try:
        for i, chunk in enumerate(todo, 1):
            if chunk.teammates_block and not asked_team:
                asked_team = True
                if input("\nRun the teammates' block (sit/stand + jump, 3 min)? [y/n]: ").strip().lower() != "y":
                    break
            while True:
                print(f"\n--- Step {len(done) + i} of {len(PLAN)}: {chunk.id}  ({chunk.people} people in the room) ---")
                print(f"  {chunk.say}")
                voice.say(chunk.say)
                if chunk.people == 0:
                    print("  Everyone must be OUT of the room, door as agreed.")
                input("  Press Enter when ready (recording starts 5 s later)... ")
                for s in range(5, 0, -1):
                    sys.stdout.write(f"\r  starting in {s} ")
                    sys.stdout.flush()
                    if s <= 3:
                        beep(700, 120)
                    time.sleep(1)
                print()
                result = record_chunk(chunk, session, voice)
                pk = result["packets"]
                low = [str(n) for n in NODES if pk.get(str(n), 0) < 20 * CHUNK_S]
                print(f"  packets per board: {pk}" + (f"   WARNING: low on ESP {', '.join(low)}" if low else ""))
                if input("  [Enter] keep and continue, [r] redo this chunk: ").strip().lower() != "r":
                    break
            entry = asdict(chunk) | result
            meta["chunks"] = [c for c in meta["chunks"] if c["id"] != chunk.id] + [entry]
            save(meta, meta_path)
    finally:
        voice.close()

    print(f"\nSession saved in {sdir}")
    print("Next:  python train_model.py " + session + "   then   python package_dataset.py " + session)


if __name__ == "__main__":
    main()
