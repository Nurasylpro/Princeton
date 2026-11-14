"""Window features for the people-count model (shared by training and live inference).

A window is ~3 s of CSI from each node: arrays of arrival times (s), 52-subcarrier
amplitudes and RSSI. Per node we describe (a) how much the channel fluctuates and at
which rates, (b) how many independent things are moving (energy spread over principal
components), and (c) how far the static channel has shifted from the empty room
(people standing or sitting still change the multipath even without moving).
"""
from __future__ import annotations

import numpy as np

from config import ACTIVE_NODES

NODES = ACTIVE_NODES
WINDOW_S = 3.0
RESAMPLE_HZ = 40.0
BANDS_HZ = ((0.1, 0.6), (0.6, 2.0), (2.0, 6.0), (6.0, 15.0))
MIN_PACKETS = 30  # below this the node counts as missing for the window

PER_NODE = [
    "std_mean", "std_med", "std_p90", "std_max",
    "diff_mean", "diff_p90",
    *[f"band_{lo:g}_{hi:g}" for lo, hi in BANDS_HZ],
    "pc1_share", "pc3_share", "sc_corr",
    "rssi_mean", "rssi_std",
    "base_dist", "base_cos",
    "present",
]
def feature_names(nodes=NODES) -> list[str]:
    return [f"n{n}_{name}" for n in nodes for name in PER_NODE]


def _normalise(amps: np.ndarray) -> np.ndarray:
    """Divide each packet by its mean amplitude: removes AGC/gain jumps between packets."""
    scale = amps.mean(axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return amps / scale


def node_baseline(amps: np.ndarray) -> np.ndarray:
    """Empty-room mean profile of normalised amplitudes for one node."""
    return _normalise(amps.astype(np.float64)).mean(axis=0)


def node_features(t: np.ndarray, amps: np.ndarray, rssi: np.ndarray,
                  baseline: np.ndarray | None) -> list[float]:
    if len(t) < MIN_PACKETS:
        return [0.0] * (len(PER_NODE) - 1) + [0.0]
    a = _normalise(amps.astype(np.float64))
    std = a.std(axis=0)
    diff = np.abs(np.diff(a, axis=0)).mean(axis=1)

    centred = a - a.mean(axis=0)
    # Principal components of the subcarrier time series.
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    energy = s ** 2
    total = energy.sum() or 1.0
    pc1 = centred @ vt[0]

    # Resample PC1 onto a uniform grid (arrivals are uneven) and take band powers.
    grid = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
    if len(grid) >= 16:
        x = np.interp(grid, t, pc1)
        x = (x - x.mean()) * np.hanning(len(x))
        spec = np.abs(np.fft.rfft(x)) ** 2 / len(x)
        freqs = np.fft.rfftfreq(len(x), 1.0 / RESAMPLE_HZ)
        bands = [float(spec[(freqs >= lo) & (freqs < hi)].sum()) for lo, hi in BANDS_HZ]
    else:
        bands = [0.0] * len(BANDS_HZ)

    corr = np.corrcoef(centred.T)
    sc_corr = float(np.nanmean(np.abs(corr[np.triu_indices_from(corr, 1)]))) if corr.ndim == 2 else 0.0

    prof = a.mean(axis=0)
    if baseline is not None:
        base_dist = float(np.linalg.norm(prof - baseline))
        base_cos = float(prof @ baseline / (np.linalg.norm(prof) * np.linalg.norm(baseline) + 1e-9))
    else:
        base_dist, base_cos = 0.0, 1.0

    return [
        float(std.mean()), float(np.median(std)), float(np.percentile(std, 90)), float(std.max()),
        float(diff.mean()), float(np.percentile(diff, 90)),
        *[float(np.log10(b + 1e-9)) for b in bands],
        float(energy[0] / total), float(energy[:3].sum() / total), sc_corr,
        float(rssi.mean()), float(rssi.std()),
        base_dist, base_cos,
        1.0,
    ]


def window_features(window: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
                    baselines: dict[int, np.ndarray], nodes=NODES) -> np.ndarray:
    """window: node_id -> (t, amps[N,52], rssi[N]). Returns the feature_names(nodes) vector."""
    out: list[float] = []
    for n in nodes:
        t, amps, rssi = window.get(n, (np.empty(0), np.empty((0, 52)), np.empty(0)))
        out.extend(node_features(t, amps, rssi, baselines.get(n)))
    return np.asarray(out, dtype=np.float64)


def count_class(n_people: int) -> str:
    return "3+" if n_people >= 3 else str(n_people)
