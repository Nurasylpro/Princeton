# AuraCare live outputs (v1)

All outputs come from one process on the sensing laptop, `auracare/hub.py`. It receives
the ESP32 CSI packets, forwards them to the RuView sensing server, runs our room-trained
model, and serves the streams below.

- **Base URL:** `http://<laptop>:8770` (`ws://<laptop>:8770` for streams). On the
  laptop itself this is `127.0.0.1`. On the iPhone hotspot the laptop is `172.20.10.4`,
  reachable when the hub is started with `--public`.
- **Time:** every message has `ts_unix` (float seconds, laptop clock, set when the
  packet arrives) and `ts` (ISO-8601 UTC).
- **Setup:** a 2.5 × 2.5 m room. One transmitter, the iPhone 15 Pro hotspot at the room
  centre, 0.76 m high (2.4 GHz, channel 6, 20 MHz). Two receivers,
  **`node_id` 2 and 3**: ESP32-WROOM-32 DevKit V1 boards in two adjacent corners,
  0.40 m high, about 1.5 m from the phone. Each node delivers about 30–45 packets/s; the
  firmware caps it at 50/s. The rate is uneven, so use the timestamps and `seq`, not a
  fixed rate.

## 1. `GET /v1/status`: health and readiness

```json
{
  "ts": "2026-11-14T21:02:11.250Z", "ts_unix": 1794693731.250,
  "ready": true,
  "models": {"activity": "ruview-adaptive", "people_count": "2026-11-14T20:42:00Z"},
  "nodes": [
    {"node_id": 2, "pps": 41.8, "rssi_dbm": -56, "last_packet_age_ms": 21, "healthy": true},
    {"node_id": 3, "pps": 38.1, "rssi_dbm": -54, "last_packet_age_ms": 9,  "healthy": true}
  ],
  "recording": false
}
```

`healthy` means the node sent at least 20 packets/s over the last 5 s and its last packet
is under 1 s old. `ready` means both nodes are healthy and both models are loaded.

## 2. `ws /v1/events`: room state and motion (1 message/s)

```json
{
  "type": "auracare.event", "v": 1,
  "ts": "2026-11-14T21:02:11.250Z", "ts_unix": 1794693731.250,
  "ready": true,
  "people":   {"count": "1", "probs": {"0": 0.02, "1": 0.91, "2": 0.07},
               "moving": true, "moving_prob": 0.84,
               "state": "p1_moving", "state_confidence": 0.79,
               "source": "auracare-state-v1"},
  "motion":   {"state": "motion", "score": 0.84, "threshold": 0.5,
               "source": "auracare-state-v1", "firmware_per_node": {"2": 1.0, "3": 1.0}},
  "presence": true,
  "activity": {"label": "moving", "confidence": 0.66,
               "raw_label": "present_moving", "source": "ruview-adaptive"}
}
```

- **`people`** comes from our classifier, trained in this room on the last 3 s of CSI
  from both nodes and averaged over 3 s. `count` is `"0"`, `"1"` or `"2"`. `state` is
  one of `empty`, `p1_still`, `p1_moving`, `p2_still`, `p2_moving`.
- **`motion.score`** is the calibrated movement score: P(moving) from the same model.
  `state` is `motion` when the score is at or above `threshold`, otherwise `idle`.
- **Firmware motion score:** the firmware's own on-board score is kept in
  `firmware_per_node` for reference only. It is not calibrated for the room and usually
  sits at 1.0. Its threshold in the firmware is 0.20
  (`CONFIG_ADAPTIVE_MOTION_THRESH_PERMIL=200`). Before our model is trained,
  `motion.score` falls back to it and `source` says `firmware-uncalibrated`.
- **`activity`** is RuView's own classifier (`empty` / `still` / `moving`), trained on
  the same session.

## 3. `ws /v1/csi`: raw CSI, one message per packet

```json
{
  "type": "auracare.csi", "v": 1,
  "ts_unix": 1794693731.2513,
  "node_id": 2, "seq": 184223,
  "rssi_dbm": -57, "noise_floor_dbm": -95,
  "channel": 6, "freq_mhz": 2437, "bandwidth_mhz": 20,
  "raw_len": 256, "sig_mode": "HT",
  "first_word_sanitized": false,
  "raw": [3, -12, 0, 0, 5, 9, "... 256 int8 values in total ..."]
}
```

- **`raw`** holds the int8 values exactly as the ESP32 reported them, as
  **[imag, real]** pairs.
  - The packets the iPhone sends are 802.11n (HT, 20 MHz), so `raw_len` is **256**.
  - **`raw[0:128]` is the L-LTF:** 64 subcarriers in the order `0..31, -32..-1`. This is
    the same 64-subcarrier legacy layout as a non-HT capture. If your WiFall pipeline
    expects 52 legacy subcarriers, start from these 128 values. In the standard layout
    the 12 null subcarriers are `0`, `27..31` and `-32..-27`.
  - **`raw[128:256]`** is the HT-LTF of the same packet (64 slots, same order).
  - If a packet ever arrives as non-HT, `raw_len` is 128 and only the L-LTF is present.
- **`sig_mode`** is worked out from `raw_len` (256 = HT, 128 = non-HT). On the original
  ESP32 the firmware's packet-type byte reports both the same way.
- **`first_word_sanitized`:** when true, the first 4 bytes were flagged invalid by the
  ESP32 and set to 0. This affects subcarriers 0 and 1.
- **`seq`** counts up by one per packet on each node. A gap means a packet was lost.
- **No MAC addresses** appear anywhere. Nodes are identified only by `node_id`.

## Recorded dataset (zip)

Each recording chunk is 60 s long. For every chunk the zip contains:

- `raw/<chunk>.bin`: every UDP datagram exactly as received. Each record is a
  little-endian `f64 ts_unix`, then a `u16 length`, then the datagram. The datagram is
  RuView ADR-018: a 20-byte header (magic `0xC5110001`, node_id, n_antennas,
  n_subcarriers, freq_mhz, seq, rssi, noise_floor, ppdu_type, flags), then the raw values.
- `decoded/<chunk>.jsonl.gz`: one JSON line per packet, using the same fields as `/v1/csi`.
- `session.json`: one entry per chunk with state, number of people, start and end time,
  node positions and node-to-phone distances. For prompted chunks it also lists each
  **action** (`sit_down`, `stand_up`, `jump`) with the time the prompt was spoken.
  People react about 0.5–1 s after the prompt.
- `read_dataset.py`: a reader for both formats.
