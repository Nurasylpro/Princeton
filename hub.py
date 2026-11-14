"""AuraCare hub: the single receiver of ESP32 CSI on the laptop.

  ESP32 nodes --UDP 5005--> hub --UDP 127.0.0.1:5006--> RuView sensing server (Docker)
                             |
                             +-- recorder (raw .bin + decoded .jsonl.gz per chunk)
                             +-- people-count model (models/count_v1.pkl)
                             +-- HTTP/WS API on :8770 (see API.md)

Run:  python hub.py            (API on 127.0.0.1 only)
      python hub.py --public   (API reachable from other devices on the hotspot)
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import pickle
import re
import socket
import struct
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np
from aiohttp import web

import csi_format as cf
import features as feat

ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "data" / "sessions"
MODELS_DIR = ROOT / "models"


def active_model_path() -> Path:
    """models/ACTIVE names the model to serve (written by train_model.py / use_model.py)."""
    active = MODELS_DIR / "ACTIVE"
    name = active.read_text().strip() if active.exists() else "room_v1"
    return MODELS_DIR / f"{name}.pkl"
RUVIEW_LATEST = "http://127.0.0.1:3000/api/v1/sensing/latest"
RUVIEW_ADAPTIVE = "http://127.0.0.1:3000/api/v1/adaptive/status"
MOTION_THRESHOLD = 0.20  # firmware CONFIG_ADAPTIVE_MOTION_THRESH_PERMIL=200
MODEL_MOTION_THRESHOLD = 0.50  # P(moving) from our trained model
HEALTHY_PPS = 20.0
ACTIVITY_NAMES = {"absent": "empty", "present_still": "still",
                  "present_moving": "moving", "active": "moving"}
_SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$")  # no "..", no leading dot


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Node:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.buf: deque[tuple[float, np.ndarray, int]] = deque()  # (t, amps52, rssi), last 10 s
        self.arrivals: deque[float] = deque()
        self.last_seen = 0.0
        self.rssi = 0
        self.motion = (0.0, 0.0)  # (score, t)

    def add(self, t: float, pkt: cf.CsiPacket) -> None:
        self.last_seen, self.rssi = t, pkt.rssi
        self.arrivals.append(t)
        self.buf.append((t, pkt.amplitudes104(), pkt.rssi))  # [:52] = L-LTF
        while self.arrivals and self.arrivals[0] < t - 5.0:
            self.arrivals.popleft()
        while self.buf and self.buf[0][0] < t - 10.0:
            self.buf.popleft()

    def pps(self, now: float) -> float:
        return sum(1 for a in self.arrivals if a >= now - 5.0) / 5.0

    def healthy(self, now: float) -> bool:
        return self.pps(now) >= HEALTHY_PPS and now - self.last_seen < 1.0

    def window(self, now: float, seconds: float):
        rows = [r for r in self.buf if r[0] >= now - seconds]
        if not rows:
            return np.empty(0), np.empty((0, 104)), np.empty(0)
        return (np.array([r[0] for r in rows]), np.stack([r[1] for r in rows]),
                np.array([r[2] for r in rows], dtype=np.float64))


class Recorder:
    def __init__(self, session: str, chunk: str):
        base = SESSIONS_DIR / session
        (base / "raw").mkdir(parents=True, exist_ok=True)
        (base / "decoded").mkdir(parents=True, exist_ok=True)
        self.raw = open(base / "raw" / f"{chunk}.bin", "wb")
        self.dec = gzip.open(base / "decoded" / f"{chunk}.jsonl.gz", "wt", encoding="utf-8")
        self.started = time.time()
        self.counts: dict[int, int] = {}

    def write(self, t: float, data: bytes, pkt: cf.CsiPacket | None) -> None:
        self.raw.write(struct.pack("<dH", t, len(data)))
        self.raw.write(data)
        if pkt is not None:
            self.dec.write(json.dumps(pkt.to_json(t), separators=(",", ":")) + "\n")
            self.counts[pkt.node_id] = self.counts.get(pkt.node_id, 0) + 1

    def close(self) -> dict:
        self.raw.close()
        self.dec.close()
        return {"duration_s": round(time.time() - self.started, 2),
                "packets": {str(k): v for k, v in sorted(self.counts.items())}}


class Occupancy:
    """Remembers that someone is inside until there is evidence they LEFT.

    The model judges each moment on its own, so a person who sits very still for a long
    time can start to look like "nobody". But nobody can leave without moving: walking out
    crosses the links. So "inside -> empty" needs movement first, then steady "nobody";
    a long, confident "nobody" (EMPTY_TIMEOUT_S) clears it as a safety net.
    """
    ENTER_CONF, ENTER_S = 0.6, 2       # confident "someone" for 2 s -> occupied
    EMPTY_CONF, EXIT_S = 0.7, 5        # P(nobody) >= 0.7 for 5 s right after movement -> left
    MOTION_LOOKBACK_S = 15             # movement must have happened this recently
    EMPTY_TIMEOUT_S = 600              # confident "nobody" for 10 min -> empty even without movement
    EXIT_GAP_S = 3                     # 'nobody' must start within 3 s of the movement
    SETTLE_S = 10                      # movement right after entering (sitting down) is not an exit

    def __init__(self) -> None:
        self.occupied, self.count = False, "0"
        self.since = time.time()
        self.enter_streak = 0
        self.last_motion = 0.0
        self.empty_since: float | None = None
        self.reason = "start"

    def update(self, now: float, people: dict) -> dict:
        probs = people["probs"]
        p0 = probs.get("0", 0.0)
        best = people["count"]
        if people["moving_prob"] >= MODEL_MOTION_THRESHOLD:
            self.last_motion = now
        if not self.occupied:
            confident = best != "0" and probs.get(best, 0.0) >= self.ENTER_CONF
            self.enter_streak = self.enter_streak + 1 if confident else 0
            if self.enter_streak >= self.ENTER_S:
                self.occupied, self.count, self.since, self.reason = True, best, now, "entered"
                self.empty_since = None
        else:
            if best != "0" and probs.get(best, 0.0) >= self.ENTER_CONF:
                self.count = best
            self.empty_since = (self.empty_since or now) if p0 >= self.EMPTY_CONF else None
            if self.empty_since is not None:
                empty_for = now - self.empty_since
                # "Left" = movement that runs DIRECTLY into "nobody" (walking out of the links).
                # Moving, then sitting still, shows "1 still" in between, so it is not an exit.
                moved_before = (self.empty_since - self.EXIT_GAP_S <= self.last_motion
                                and self.last_motion >= self.since + self.SETTLE_S)
                if (moved_before and empty_for >= self.EXIT_S) or empty_for >= self.EMPTY_TIMEOUT_S:
                    self.occupied, self.count, self.since = False, "0", now
                    self.reason = "left (movement, then empty)" if moved_before else "timeout (long confident empty)"
                    self.enter_streak = 0
            if self.occupied and p0 >= self.EMPTY_CONF and now - self.last_motion > self.MOTION_LOOKBACK_S:
                self.reason = "still inside (no movement seen, so nobody left)"
            elif self.occupied and self.reason.startswith("still inside"):
                self.reason = "inside"
        return {"occupied": self.occupied, "count": self.count, "since": iso(self.since),
                "seconds": int(now - self.since), "reason": self.reason}


class Hub(asyncio.DatagramProtocol):
    def __init__(self, forward: tuple[str, int]):
        self.forward = forward
        # Forward through a SEPARATE socket. On Windows, sending to a closed UDP port (RuView
        # container down) returns ICMP "port unreachable", which surfaces as WSAECONNRESET on
        # the sending socket and stops the receive loop if it is the listening socket.
        self.fwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fwd.setblocking(False)
        if hasattr(socket, "SIO_UDP_CONNRESET"):
            self.fwd.ioctl(socket.SIO_UDP_CONNRESET, False)
        self.nodes: dict[int, Node] = {}
        self.recorder: Recorder | None = None
        self.cal_capture: dict[int, list[np.ndarray]] | None = None  # empty-room re-calibration
        self.csi_clients: set[asyncio.Queue] = set()
        self.event_clients: set[asyncio.Queue] = set()
        self.count_model: dict | None = None
        self.recent_probs: deque[np.ndarray] = deque(maxlen=3)
        self.occupancy = Occupancy()
        self.activity ={"label": None, "confidence": None, "raw_label": None, "source": "ruview"}
        self.adaptive_loaded = False
        self.last_event: dict = {}
        self.transport: asyncio.DatagramTransport | None = None
        self.load_model()

    # ---- UDP -------------------------------------------------------------------
    def connection_made(self, transport) -> None:
        self.transport = transport
        sock = transport.get_extra_info("socket")
        if sock is not None and hasattr(socket, "SIO_UDP_CONNRESET"):
            try:  # belt and braces: never let an ICMP error stop reception
                sock.ioctl(socket.SIO_UDP_CONNRESET, False)
            except (OSError, AttributeError, ValueError):
                pass

    def error_received(self, exc: Exception) -> None:
        pass  # transient socket errors must not stop the hub

    def datagram_received(self, data: bytes, addr) -> None:
        t = time.time()
        try:  # RuView gets every datagram unchanged; never let its absence stop us
            self.fwd.sendto(data, self.forward)
        except OSError:
            pass
        magic = cf.magic_of(data)
        pkt = cf.parse_csi(data) if magic == cf.CSI_MAGIC else None
        if pkt is not None:
            self.nodes.setdefault(pkt.node_id, Node(pkt.node_id)).add(t, pkt)
            if self.cal_capture is not None and len(pkt.raw) >= 128:
                self.cal_capture.setdefault(pkt.node_id, []).append(pkt.amplitudes104())
            if self.csi_clients:
                msg = json.dumps(pkt.to_json(t), separators=(",", ":"))
                for q in list(self.csi_clients):
                    if q.full():
                        q.get_nowait()  # slow client: drop its oldest packet
                    q.put_nowait(msg)
        elif magic == cf.FEATURE_MAGIC:
            fs = cf.parse_feature(data)
            if fs is not None:
                self.nodes.setdefault(fs.node_id, Node(fs.node_id)).motion = (fs.motion_score, t)
        if self.recorder is not None:
            self.recorder.write(t, data, pkt)

    # ---- models ----------------------------------------------------------------
    def load_model(self) -> None:
        path = active_model_path()
        self.count_model = None
        if path.exists():
            with open(path, "rb") as f:
                self.count_model = pickle.load(f)
            self.recent_probs.clear()
            if hasattr(self, "occupancy"):
                self.occupancy = Occupancy()

    def people(self, now: float) -> dict | None:
        """Room state from our model: people 0/1/2 and moving, averaged over the last 3 s."""
        m = self.count_model
        if m is None:
            return None
        nodes = tuple(m.get("nodes", feat.NODES))
        win_s, width = m.get("window_s", feat.WINDOW_S), (104 if m.get("amp_mode") == "lltf_ht" else 52)
        window = {}
        for n in nodes:
            if n in self.nodes:
                t_, a_, r_ = self.nodes[n].window(now, win_s)
                window[n] = (t_, a_[:, :width] if a_.ndim == 2 else a_, r_)
        x = feat.window_features(window, m["baselines"], nodes).reshape(1, -1)
        self.recent_probs.append(m["clf"].predict_proba(x)[0])
        probs = np.mean(self.recent_probs, axis=0)
        states = [str(c) for c in m["clf"].classes_]
        smap = m["state_map"]
        count_p: dict[str, float] = {}
        moving_p = 0.0
        for s, p in zip(states, probs):
            c, mv = smap[s]
            key = "3+" if c >= 3 else str(c)
            count_p[key] = count_p.get(key, 0.0) + float(p)
            moving_p += float(p) if mv else 0.0
        state = states[int(np.argmax(probs))]
        count = max(count_p, key=count_p.get)
        return {"count": count, "probs": {k: round(v, 3) for k, v in sorted(count_p.items())},
                "moving": moving_p >= 0.5, "moving_prob": round(moving_p, 3),
                "state": state, "state_confidence": round(float(probs.max()), 3),
                "source": m.get("name", "auracare-state-v1")}

    def status(self, now: float) -> dict:
        nodes = []
        for n in feat.NODES:
            nd = self.nodes.get(n)
            if nd is None:
                nodes.append({"node_id": n, "pps": 0.0, "rssi_dbm": None,
                              "last_packet_age_ms": None, "healthy": False})
                continue
            nodes.append({"node_id": n, "pps": round(nd.pps(now), 1), "rssi_dbm": nd.rssi,
                          "last_packet_age_ms": int((now - nd.last_seen) * 1000),
                          "healthy": nd.healthy(now)})
        models = {"activity": "ruview-adaptive" if self.adaptive_loaded else None,
                  "people_count": (f"{self.count_model.get('name')} ({self.count_model.get('trained_at')})"
                                   if self.count_model else None)}
        ready = all(x["healthy"] for x in nodes) and all(models.values())
        return {"ts": iso(now), "ts_unix": round(now, 3), "ready": ready, "models": models,
                "nodes": nodes, "recording": self.recorder is not None}

    def event(self, now: float) -> dict:
        st = self.status(now)
        firmware = {str(n): round(nd.motion[0], 3) for n, nd in sorted(self.nodes.items())
                    if n in feat.NODES and now - nd.motion[1] < 5.0}
        act = self.activity
        # A node with no packets in the last 2 s makes every estimate meaningless: say so
        # instead of letting the model guess from an empty window.
        no_data = [n for n in feat.NODES
                   if n not in self.nodes or now - self.nodes[n].last_seen > 2.0]
        if no_data:
            self.recent_probs.clear()
            return {
                "type": "auracare.event", "v": 1, "ts": st["ts"], "ts_unix": st["ts_unix"],
                "ready": False, "no_data_nodes": no_data,
                "activity": act, "people": None,
                "motion": {"state": "no_data", "score": None, "threshold": None, "source": None,
                           "firmware_per_node": firmware},
                "presence": None,
            }
        people = self.people(now)
        occupancy = self.occupancy.update(now, people) if people is not None else None
        if people is not None:
            # Room-trained model: P(moving) is the calibrated movement score.
            presence = people["count"] != "0"
            score, threshold, source = people["moving_prob"], MODEL_MOTION_THRESHOLD, people["source"]
        else:
            # Before training only the firmware's own score exists. It is NOT calibrated for
            # the room (it often sits at 1.0), so it is labelled as such.
            presence = None if act["raw_label"] is None else act["raw_label"] != "absent"
            score = max(firmware.values()) if firmware else 0.0
            threshold, source = MOTION_THRESHOLD, "firmware-uncalibrated"
        return {
            "type": "auracare.event", "v": 1, "ts": st["ts"], "ts_unix": st["ts_unix"],
            "ready": st["ready"],
            "activity": act,
            "people": people,
            "motion": {"state": "motion" if score >= threshold else "idle",
                       "score": round(score, 3), "threshold": threshold, "source": source,
                       "firmware_per_node": firmware},
            "presence": presence,
            "occupancy": occupancy,
        }


async def poll_ruview(hub: Hub) -> None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as http:
        tick = 0
        while True:
            try:
                async with http.get(RUVIEW_LATEST) as r:
                    cls = (await r.json()).get("classification") or {}
                raw = cls.get("motion_level")
                hub.activity = {"label": ACTIVITY_NAMES.get(raw, raw), "raw_label": raw,
                                "confidence": None if cls.get("confidence") is None
                                else round(float(cls["confidence"]), 3),
                                "source": "ruview-adaptive" if hub.adaptive_loaded else "ruview-default"}
                if tick % 10 == 0:
                    async with http.get(RUVIEW_ADAPTIVE) as r:
                        hub.adaptive_loaded = bool((await r.json()).get("loaded"))
            except Exception:
                hub.activity = {"label": None, "confidence": None, "raw_label": None, "source": "ruview-unreachable"}
            tick += 1
            await asyncio.sleep(1.0)


async def broadcast_events(hub: Hub) -> None:
    while True:
        await asyncio.sleep(1.0)
        hub.last_event = hub.event(time.time())
        msg = json.dumps(hub.last_event, separators=(",", ":"))
        for q in list(hub.event_clients):
            if q.full():
                q.get_nowait()
            q.put_nowait(msg)


def make_app(hub: Hub) -> web.Application:
    routes = web.RouteTableDef()

    @routes.get("/v1/status")
    async def status(_):
        return web.json_response(hub.status(time.time()))

    @routes.get("/v1/events/latest")
    async def latest(_):
        return web.json_response(hub.last_event or hub.event(time.time()))

    async def stream(request, clients: set, maxsize: int):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        clients.add(q)
        try:
            while not ws.closed:
                await ws.send_str(await q.get())
        except (ConnectionResetError, RuntimeError):
            pass
        finally:
            clients.discard(q)
        return ws

    @routes.get("/v1/events")
    async def events(request):
        return await stream(request, hub.event_clients, 10)

    @routes.get("/v1/csi")
    async def csi(request):
        return await stream(request, hub.csi_clients, 1000)

    @routes.post("/v1/record/start")
    async def record_start(request):
        body = await request.json()
        session, chunk = str(body.get("session", "")), str(body.get("chunk", ""))
        if not (_SAFE.match(session) and _SAFE.match(chunk)):
            return web.json_response({"error": "session/chunk must match [A-Za-z0-9_.-]"}, status=400)
        if hub.recorder is not None:
            hub.recorder.close()
        hub.recorder = Recorder(session, chunk)
        return web.json_response({"ok": True, "started_unix": hub.recorder.started})

    @routes.post("/v1/record/stop")
    async def record_stop(_):
        if hub.recorder is None:
            return web.json_response({"error": "not recording"}, status=409)
        rec, hub.recorder = hub.recorder, None
        return web.json_response({"ok": True, **rec.close()})

    @routes.post("/v1/calibrate")
    async def calibrate(request):
        """Nobody inside for `seconds`: refresh the model's empty-room fingerprint (drift fix)."""
        body = await request.json() if request.can_read_body else {}
        seconds = float(min(max(body.get("seconds", 30), 10), 120))
        if hub.count_model is None:
            return web.json_response({"error": "no model loaded"}, status=409)
        if hub.cal_capture is not None:
            return web.json_response({"error": "calibration already running"}, status=409)
        hub.cal_capture = {}
        try:
            await asyncio.sleep(seconds)
        finally:
            cap, hub.cal_capture = hub.cal_capture, None
        nodes = tuple(hub.count_model.get("nodes", feat.NODES))
        short = [n for n in nodes if len(cap.get(n, [])) < 10 * seconds]
        if short:
            return web.json_response({"error": f"too few packets from nodes {short}; nothing changed",
                                      "packets": {str(n): len(cap.get(n, [])) for n in nodes}}, status=422)
        width = 104 if hub.count_model.get("amp_mode") == "lltf_ht" else 52
        new = {n: feat.node_baseline(np.stack(cap[n])[:, :width]) for n in nodes}
        old = hub.count_model["baselines"]
        shift = {str(n): round(float(np.linalg.norm(new[n] - old[n])), 3) for n in nodes if n in old}
        hub.count_model["baselines"] = new
        hub.count_model["calibrated_at"] = iso(time.time())
        with open(active_model_path(), "wb") as f:
            pickle.dump(hub.count_model, f)
        hub.recent_probs.clear()
        return web.json_response({"ok": True, "seconds": seconds,
                                  "packets": {str(n): len(cap[n]) for n in nodes},
                                  "fingerprint_shift": shift})

    @routes.post("/v1/models/reload")
    async def reload(_):
        hub.load_model()
        return web.json_response({"ok": True, "people_count": hub.count_model is not None})

    app = web.Application()
    app.add_routes(routes)
    return app


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--udp-port", type=int, default=5005)
    ap.add_argument("--forward-port", type=int, default=5006)
    ap.add_argument("--api-port", type=int, default=8770)
    ap.add_argument("--public", action="store_true", help="bind the API on 0.0.0.0 instead of 127.0.0.1")
    args = ap.parse_args()

    hub = Hub(("127.0.0.1", args.forward_port))
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: hub, local_addr=("0.0.0.0", args.udp_port))
    runner = web.AppRunner(make_app(hub))
    await runner.setup()
    host = "0.0.0.0" if args.public else "127.0.0.1"
    await web.TCPSite(runner, host, args.api_port).start()
    print(f"[hub] UDP 0.0.0.0:{args.udp_port} -> RuView 127.0.0.1:{args.forward_port} | "
          f"API http://{host}:{args.api_port} | count model: {'loaded' if hub.count_model else 'none'}",
          flush=True)
    await asyncio.gather(poll_ruview(hub), broadcast_events(hub))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
