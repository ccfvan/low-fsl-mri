"""Paper microvascular duration and time-step policy.

These formulas follow the transfer-fitting script used for the paper
tables, which itself follows the ideal-preparation simulation entry.
Frequencies are the integers 1, 2, ..., 100 Hz. The time step is chosen
so that one modulation period is an integer number of steps:

    n = max(4, round(1 / (f * dt_ref)))
    dt = 1 / (f * n)

where dt_ref is 12.5 microseconds for the paper baseline. Ordinary,
composite, and balanced preparations use 13 durations from 0 to
min(0.2 s, 4/f), snapped to multiples of 4*dt so ideal pulse centers
are representable. CPMG-MLEV16 (ideal pulses) uses completed modulation
periods: 3 periods at f <= 11 Hz, and min(5, floor(0.25*f + 1e-10))
periods at higher f.

A 1e-10 term appears in the fitting script; the simulation source used
floor(0.25*f) without it. The two agree for all frequencies used in the
paper. This module keeps the fitting-script form.
"""
from __future__ import annotations

import math
import numpy as np

PAPER_FREQUENCIES_HZ = tuple(range(1, 101))
PAPER_CONVERGENCE_FREQUENCIES_HZ = (10, 20, 50)
PAPER_PREPARATIONS = ("ordinary", "composite", "balanced", "cpmg")
PAPER_BASELINE_DT_US = 12.5
PAPER_BASELINE_GRID = 64
PAPER_REFINED_GRID = 96
PAPER_HALVED_DT_US = 6.25
PAPER_RADIUS_UM = 6.0
PAPER_VOLUME_FRACTION = 0.035
PAPER_DELTA_CHI = 0.18e-6
PAPER_B0_T = 3.0
PAPER_D_UM2_MS = 0.8
PAPER_T1_S = 1.3
PAPER_T2_S = 0.08


def steps_per_period(frequency_hz: float, dt_ref_us: float = PAPER_BASELINE_DT_US) -> int:
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency must be positive and finite")
    if not math.isfinite(dt_ref_us) or dt_ref_us <= 0:
        raise ValueError("reference time step must be positive and finite")
    return max(4, round(1.0 / (frequency_hz * dt_ref_us * 1e-6)))


def period_dt_s(frequency_hz: float, dt_ref_us: float = PAPER_BASELINE_DT_US) -> float:
    return 1.0 / (frequency_hz * steps_per_period(frequency_hz, dt_ref_us))


def cpmg_period_count(frequency_hz: float) -> int:
    if frequency_hz <= 11:
        return 3
    return min(5, int(math.floor(0.250 * frequency_hz + 1e-10)))


def preparation_times_s(
    method: str,
    frequency_hz: float,
    dt_ref_us: float = PAPER_BASELINE_DT_US,
) -> np.ndarray:
    method = method.lower()
    if method not in PAPER_PREPARATIONS:
        raise ValueError(f"unsupported preparation: {method}")
    if method == "cpmg":
        return np.arange(cpmg_period_count(frequency_hz) + 1, dtype=float) / frequency_hz
    dt_s = period_dt_s(frequency_hz, dt_ref_us)
    raw = np.linspace(0.0, min(0.20, 4.0 / frequency_hz), 13)
    return np.unique(np.round(raw / (4.0 * dt_s)) * 4.0 * dt_s)
