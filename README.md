# AuraCare: WiFi-CSI room sensing

Detects **room occupancy**, **motion**, and **a person down on the floor** from WiFi signals.
No cameras are used. It is built on the [RuView](https://github.com/ruvnet/RuView) ESP32 CSI firmware.

## How it works

```
iPhone hotspot (TX, room centre)
      │ WiFi 2.4 GHz ch6
      ▼
ESP32 node 2 ─┐  CSI packets over UDP
ESP32 node 3 ─┤──────────────────────► hub.py (laptop)
              │                          ├─ decodes CSI (csi_format.py)
              │                          ├─ forwards raw packets → RuView server (:3000, Docker)
              │                          ├─ features.py → ML models (models/*.pkl)
              │                          └─ serves API + dashboard on :8770
                                              ├─ GET /v1/status
                                              ├─ ws  /v1/events  (room state, 1/s)
                                              └─ dashboard.html (big-text view)
```

- **Hardware:** 2 ESP32-WROOM-32 receivers in adjacent corners of a 2.5 × 2.5 m room. An iPhone hotspot is the transmitter. Layout is set in `config.py`.
- **Where the models are:** `models/`. The hub loads them with pickle (scikit-learn, Extra-Trees).
  - `models/ACTIVE` names the main model the hub serves (currently `presence_v1`).
  - `presence_v1.pkl`: occupied/empty plus moving/still. **Active.**
  - `down_v1.pkl`: **PERSON DOWN** (someone lying on the floor). Always loaded, independent of ACTIVE.
  - `combined_v1.pkl`, `room_v1.pkl`, `hall_v1.pkl`: earlier people-count/state models. Switch with `python use_model.py <name>`.
  - `fall_v1_*.pkl.off`: fall-moment detectors, disabled. Rename one to `fall_v1.pkl` to enable it.

## Files

| File | What it does |
|---|---|
| `hub.py` | Main process: receives CSI, runs the models, serves the API and dashboard |
| `config.py` | Room size and node positions |
| `csi_format.py` | Decodes the ESP32 UDP packets |
| `features.py` | Window features shared by training and live inference |
| `calibrate.py` | Re-learns the empty room. **Run before every use or demo** |
| `live.py` / `dashboard.html` | Plain-text and big-text live views |
| `session.py`, `plans.py` | Guided data recording (voice prompts) |
| `train_model.py`, `train_combined.py`, `train_down.py`, `train_fall.py` | Training scripts |
| `use_model.py` | Switches the active model without restarting the hub |
| `set_down_threshold.py` | Tunes the PERSON DOWN alarm threshold |
| `motion_check.py` | Sanity test: still → walk → still |
| `package_dataset.py`, `read_dataset.py` | Export the dataset for teammates and read it |
| `API.md` | Full API reference |

## Run

```
docker start ruview         # RuView sensing server (:3000)
python hub.py --public      # receive CSI + serve :8770
python calibrate.py         # empty room, nobody inside
open http://localhost:8770  # dashboard
```

The recorded data (`data/`, about 900 MB) is not in this repo.
