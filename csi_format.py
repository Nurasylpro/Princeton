"""Decode the UDP datagrams the RuView ESP32 firmware sends (ADR-018 family).

Only two packet kinds matter here:
  * 0xC5110001 raw CSI: 20-byte header + n_antennas * n_subcarriers * 2 int8 values
  * 0xC5110006 feature state (rv_feature_state.c): per-node motion/presence scores
Everything else (vitals 0xC5110002, ...) is forwarded to RuView untouched and ignored here.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

CSI_MAGIC = 0xC5110001
FEATURE_MAGIC = 0xC5110006

_CSI_HDR = struct.Struct("<IBBHIIbbBB")  # magic, node, n_ant, n_sc, freq, seq, rssi, nf, ppdu, flags
# rv_feature_state_t (packed, 60 bytes): magic, node_id, mode, seq, ts_us, 9 floats
# (motion, presence, resp_bpm, resp_conf, hr_bpm, hr_conf, anomaly, env_shift, coherence),
# quality_flags, reserved, crc32
_FEATURE = struct.Struct("<IBBHQfffffffffHHI")
assert _FEATURE.size == 60
FLAG_FIRST_WORD_SANITIZED = 0x20

# ESP32 L-LTF buffer: 64 subcarriers stored as [imag, real] pairs in the order
# 0..31 then -32..-1. The 52 occupied subcarriers are -26..-1 and 1..26.
_SC_INDEX = np.array(list(range(0, 32)) + list(range(-32, 0)))
VALID_POS = np.where((np.abs(_SC_INDEX) >= 1) & (np.abs(_SC_INDEX) <= 26))[0]  # 52 positions


@dataclass(slots=True)
class CsiPacket:
    node_id: int
    seq: int
    rssi: int
    noise_floor: int
    freq_mhz: int
    ppdu_type: int
    flags: int
    n_subcarriers: int
    raw: bytes  # the I/Q payload exactly as sent (128 bytes for non-HT L-LTF)

    @property
    def channel(self) -> int:
        return (self.freq_mhz - 2407) // 5 if 2412 <= self.freq_mhz <= 2472 else (14 if self.freq_mhz == 2484 else 0)

    @property
    def sig_mode(self) -> str:
        # Classic ESP32 reports non-HT and HT with the same ppdu byte; the buffer
        # length tells them apart (L-LTF only = 128 bytes = non-HT).
        return "non-HT" if len(self.raw) == 128 else "HT"

    def raw_int8(self) -> np.ndarray:
        return np.frombuffer(self.raw, dtype=np.int8)

    def amplitudes52(self) -> np.ndarray:
        v = self.raw_int8().astype(np.float32)
        if v.size < 128:
            return np.zeros(52, np.float32)
        imag, real = v[0:128:2], v[1:128:2]
        return np.sqrt(real * real + imag * imag)[VALID_POS]

    def to_json(self, ts_unix: float) -> dict:
        return {
            "type": "auracare.csi", "v": 1, "ts_unix": round(ts_unix, 4),
            "node_id": self.node_id, "seq": self.seq,
            "rssi_dbm": self.rssi, "noise_floor_dbm": self.noise_floor,
            "channel": self.channel, "freq_mhz": self.freq_mhz, "bandwidth_mhz": 20,
            "raw_len": len(self.raw), "sig_mode": self.sig_mode,
            "first_word_sanitized": bool(self.flags & FLAG_FIRST_WORD_SANITIZED),
            "raw": self.raw_int8().tolist(),
        }


@dataclass(slots=True)
class FeatureState:
    node_id: int
    motion_score: float
    presence_score: float


def magic_of(data: bytes) -> int:
    return struct.unpack_from("<I", data, 0)[0] if len(data) >= 4 else 0


def parse_csi(data: bytes) -> CsiPacket | None:
    if len(data) < _CSI_HDR.size or magic_of(data) != CSI_MAGIC:
        return None
    _, node, n_ant, n_sc, freq, seq, rssi, nf, ppdu, flags = _CSI_HDR.unpack_from(data, 0)
    payload = data[_CSI_HDR.size:_CSI_HDR.size + n_ant * n_sc * 2]
    return CsiPacket(node, seq, rssi, nf, freq, ppdu, flags, n_sc, bytes(payload))


def parse_feature(data: bytes) -> FeatureState | None:
    if len(data) < _FEATURE.size or magic_of(data) != FEATURE_MAGIC:
        return None
    _, node, _mode, _seq, _ts_us, motion, presence, *_rest = _FEATURE.unpack_from(data, 0)
    return FeatureState(node_id=node, motion_score=motion, presence_score=presence)
