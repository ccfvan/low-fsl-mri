"""Preparation waveforms, finite-duration spectra, Bloch propagation, and RF cost.

The sinusoidal waveform is
``omega_x(t) = beta * omega_t * cos(omega_t * t)``, sampled and integrated
with the same piecewise-constant Bloch propagator as the other preparations.

``PAPER_METHODS`` is the six-preparation comparison used in the accompanying
paper: conventional CW, composite, balanced, quadruple-refocused, sinusoidal,
and CPMG-MLEV16. Triple-refocused spin lock remains buildable and is not part
of that comparison.

The default field grid samples B0 from -250 to 250 Hz in 5 Hz steps and B1
from -25% to 25% in 0.5% steps. Summary windows of |B0| <= 100 Hz and
|B0| <= 250 Hz are both available. The paper Table 2 uses the |B0| <= 250 Hz
window. This package implements the simulation methods; it is not a
complete rerun of the paper figures.

For CW, composite, balanced, and quadruple-refocused preparations, the
constant locking amplitude passed to ``build_core`` is used directly. The
paper comparison sets that amplitude equal to the target frequency
(20, 50, or 100 Hz). No peak-alignment calibration is applied.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    t1_s: float = 1.2
    t2_s: float = 0.080
    m0: float = 1.0
    hard_90_width_s: float = 0.4e-3
    hard_180_width_s: float = 0.8e-3
    rf_peak_hz: float = 625.0
    b0_min_hz: float = -250.0
    b0_max_hz: float = 250.0
    b0_step_hz: float = 5.0
    b1_min_pct: float = -25.0
    b1_max_pct: float = 25.0
    b1_step_pct: float = 0.5
    spectrum_dt_s: float = 10e-6
    spectrum_zero_pad_factor: int = 32
    spectrum_max_hz: float = 500.0
    b1_derivative_step: float = 1e-3
    representative_20hz_tsl_ms: float = 400.0
    representative_50hz_tsl_ms: float = 160.0
    representative_100hz_tsl_ms: float = 80.0
    rotary_center_pi_hz: float = 625.0
    rotary_center_pi_width_s: float = 0.8e-3
    cpmg_foltz_delta: float = 0.824
    sinusoidal_beta: float = 2.4048256
    sinusoidal_dt_s: float = 25e-6

    @property
    def b0_hz(self) -> np.ndarray:
        return np.arange(
            self.b0_min_hz,
            self.b0_max_hz + self.b0_step_hz / 2,
            self.b0_step_hz,
        )

    @property
    def b1_pct(self) -> np.ndarray:
        return np.arange(
            self.b1_min_pct,
            self.b1_max_pct + self.b1_step_pct / 2,
            self.b1_step_pct,
        )


class Operation(NamedTuple):
    duration_s: float
    fx_hz: float
    fy_hz: float
    label: str




@dataclass(frozen=True)
class Method:
    key: str
    directory: str
    label: str
    short_label: str
    color: str
    linestyle: str | tuple
    marker: str


# Foltz et al. MLEV-16 phase cycle, written about the x axis to match the
# locking-axis convention used here.  This is a coordinate rotation of the
# +/-y phase table reported for a 90x excitation in the source article.
MLEV16_SIGNS = (
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    -1,
    -1,
    1,
)


def append_op(
    operations: list[Operation],
    duration_s: float,
    fx_hz: float,
    fy_hz: float,
    label: str,
) -> None:
    if duration_s > 1e-12:
        operations.append(Operation(float(duration_s), float(fx_hz), float(fy_hz), label))


def append_hard_90(
    operations: list[Operation], p: Params, axis: str, sign: int, label: str
) -> None:
    if axis == "x":
        append_op(operations, p.hard_90_width_s, sign * p.rf_peak_hz, 0.0, label)
    elif axis == "y":
        append_op(operations, p.hard_90_width_s, 0.0, sign * p.rf_peak_hz, label)
    else:
        raise ValueError(f"Unknown RF axis: {axis}")


def append_hard_180(
    operations: list[Operation], p: Params, axis: str, sign: int, label: str
) -> None:
    if axis == "x":
        append_op(operations, p.hard_180_width_s, sign * p.rf_peak_hz, 0.0, label)
    elif axis == "y":
        append_op(operations, p.hard_180_width_s, 0.0, sign * p.rf_peak_hz, label)
    else:
        raise ValueError(f"Unknown RF axis: {axis}")


def append_composite_180(
    operations: list[Operation], p: Params, parent_axis: str, sign: int, label: str
) -> None:
    if parent_axis == "x":
        append_hard_90(operations, p, "y", sign, f"{label}:90Y")
        append_hard_180(operations, p, "x", sign, f"{label}:180X")
        append_hard_90(operations, p, "y", sign, f"{label}:90Y")
    elif parent_axis == "y":
        append_hard_90(operations, p, "x", sign, f"{label}:90X")
        append_hard_180(operations, p, "y", sign, f"{label}:180Y")
        append_hard_90(operations, p, "x", sign, f"{label}:90X")
    else:
        raise ValueError(f"Unknown composite parent axis: {parent_axis}")


def refocused_definition(method_key: str, tsl_s: float) -> tuple[np.ndarray, tuple[int, ...]]:
    if method_key == "balanced":
        return np.array([tsl_s / 4, 3 * tsl_s / 4]), (1, -1)
    if method_key == "triple_refocused":
        return np.array([tsl_s / 4, tsl_s / 2, 3 * tsl_s / 4]), (1, 1, -1)
    if method_key == "quadruple_refocused":
        return (
            np.array([tsl_s / 8, 3 * tsl_s / 8, 5 * tsl_s / 8, 7 * tsl_s / 8]),
            (1, 1, -1, -1),
        )
    raise ValueError(f"Not a fixed refocused method: {method_key}")


def build_refocused_core(p: Params, method_key: str, tsl_s: float, target_hz: float) -> list[Operation]:
    centers, pulse_signs = refocused_definition(method_key, tsl_s)
    operations: list[Operation] = []
    cursor = 0.0
    lock_sign = 1
    for index, (center_s, pulse_sign) in enumerate(zip(centers, pulse_signs, strict=True), start=1):
        start_s = center_s - p.hard_180_width_s / 2
        if start_s < cursor - 1e-12:
            raise ValueError(f"{method_key}: finite pulses overlap at TSL={1e3*tsl_s:g} ms")
        append_op(operations, start_s - cursor, lock_sign * target_hz, 0.0, f"SL{lock_sign:+d}")
        append_hard_180(operations, p, "x", pulse_sign, f"pi{index}:{pulse_sign:+d}X")
        cursor = center_s + p.hard_180_width_s / 2
        lock_sign *= -1
    append_op(operations, tsl_s - cursor, lock_sign * target_hz, 0.0, f"SL{lock_sign:+d}")
    return operations


def build_cpmg_core(p: Params, tsl_s: float, target_hz: float) -> list[Operation]:
    spacing_s = 1.0 / (2.0 * target_hz)
    n_pulses_float = tsl_s / spacing_s
    n_pulses = int(round(n_pulses_float))
    if not np.isclose(n_pulses_float, n_pulses, atol=1e-10):
        raise ValueError(
            f"CPMG TSL={1e3*tsl_s:g} ms does not contain an integer number of "
            f"target-{target_hz:g}-Hz half periods"
        )
    composite_width_s = 2 * p.hard_180_width_s
    if composite_width_s > spacing_s + 1e-12:
        raise ValueError("Finite composite inversion does not fit the CPMG spacing")
    operations: list[Operation] = []
    cursor = 0.0
    for pulse_index in range(n_pulses):
        center_s = (pulse_index + 0.5) * spacing_s
        start_s = center_s - composite_width_s / 2
        append_op(operations, start_s - cursor, 0.0, 0.0, "gap")
        sign = MLEV16_SIGNS[pulse_index % len(MLEV16_SIGNS)]
        append_composite_180(
            operations,
            p,
            "x",
            sign,
            f"MLEV16_{pulse_index + 1}:{sign:+d}X",
        )
        cursor = center_s + composite_width_s / 2
    append_op(operations, tsl_s - cursor, 0.0, 0.0, "gap")
    return operations


def build_rotary_echo_core(
    p: Params, tsl_s: float, lock_hz: float
) -> list[Operation]:
    """Build +SL, center 180X, and -SL within the requested TSL."""

    if lock_hz <= 0:
        raise ValueError("Rotary spin-lock amplitude must be positive")
    if tsl_s < p.rotary_center_pi_width_s - 1e-12:
        raise ValueError("TSL is shorter than the rotary center pi pulse")

    flank_s = (tsl_s - p.rotary_center_pi_width_s) / 2
    operations: list[Operation] = []
    append_op(operations, flank_s, lock_hz, 0.0, "rotary +SL")
    append_op(
        operations,
        p.rotary_center_pi_width_s,
        p.rotary_center_pi_hz,
        0.0,
        "rotary center pi:180X",
    )
    append_op(operations, flank_s, -lock_hz, 0.0, "rotary -SL")
    return operations


def build_sinusoidal_core(p: Params, tsl_s: float, target_hz: float) -> list[Operation]:
    """Piecewise-constant samples of omega_x(t) = beta * omega_t * cos(omega_t t).

    The step count is chosen so the samples exactly fill ``tsl_s``. The default
    requested step is 25 microseconds. A caller can override
    ``Params.sinusoidal_dt_s``.
    """

    if target_hz <= 0:
        raise ValueError("Sinusoidal target frequency must be positive")
    if p.sinusoidal_dt_s <= 0:
        raise ValueError("Sinusoidal time step must be positive")
    n_steps = int(round(tsl_s / p.sinusoidal_dt_s))
    if n_steps < 1:
        raise ValueError("Sinusoidal preparation is shorter than one time step")
    dt_s = tsl_s / n_steps
    omega = 2 * math.pi * target_hz
    operations: list[Operation] = []
    for step in range(n_steps):
        t_mid_s = (step + 0.5) * dt_s
        fx_hz = p.sinusoidal_beta * target_hz * math.cos(omega * t_mid_s)
        operations.append(Operation(dt_s, fx_hz, 0.0, "sinusoidal SL"))
    return operations


def build_core(p: Params, method_key: str, tsl_s: float, target_hz: float) -> list[Operation]:
    if tsl_s <= 0:
        return []
    if method_key == "ordinary":
        return [Operation(tsl_s, target_hz, 0.0, "continuous SL")]
    if method_key == "rotary_echo":
        return build_rotary_echo_core(p, tsl_s, target_hz)
    if method_key in {"balanced", "triple_refocused", "quadruple_refocused"}:
        return build_refocused_core(p, method_key, tsl_s, target_hz)
    if method_key == "sinusoidal":
        return build_sinusoidal_core(p, tsl_s, target_hz)
    if method_key == "phase_cycled_cpmg":
        return build_cpmg_core(p, tsl_s, target_hz)
    raise ValueError(f"Unknown method: {method_key}")


def build_full_preparation(
    p: Params, method_key: str, tsl_s: float, target_hz: float
) -> list[Operation]:
    if tsl_s <= 0:
        return []
    operations: list[Operation] = []
    append_hard_90(operations, p, "y", 1, "tip-down +90Y")
    operations.extend(build_core(p, method_key, tsl_s, target_hz))
    append_hard_90(operations, p, "y", -1, "projection -90Y")
    return operations


def rodrigues_matrices(
    field_hz: np.ndarray, duration_s: float | np.ndarray
) -> np.ndarray:
    magnitude_hz = np.linalg.norm(field_hz, axis=1)
    unit = np.zeros_like(field_hz)
    nonzero = magnitude_hz > 1e-14
    unit[nonzero] = field_hz[nonzero] / magnitude_hz[nonzero, None]
    unit[~nonzero, 0] = 1.0
    angle = 2 * np.pi * magnitude_hz * duration_s
    c = np.cos(angle)
    s = np.sin(angle)
    one_minus_c = 1 - c
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    rot = np.empty((field_hz.shape[0], 3, 3), dtype=float)
    rot[:, 0, 0] = c + x * x * one_minus_c
    rot[:, 0, 1] = x * y * one_minus_c - z * s
    rot[:, 0, 2] = x * z * one_minus_c + y * s
    rot[:, 1, 0] = y * x * one_minus_c + z * s
    rot[:, 1, 1] = c + y * y * one_minus_c
    rot[:, 1, 2] = y * z * one_minus_c - x * s
    rot[:, 2, 0] = z * x * one_minus_c - y * s
    rot[:, 2, 1] = z * y * one_minus_c + x * s
    rot[:, 2, 2] = c + z * z * one_minus_c
    return rot


def finite_propagator(
    p: Params,
    operation: Operation,
    b0_hz: np.ndarray,
    b1_scale: np.ndarray,
) -> np.ndarray:
    wx = 2 * np.pi * operation.fx_hz * b1_scale
    wy = 2 * np.pi * operation.fy_hz * b1_scale
    dw = 2 * np.pi * b0_hz
    r1 = 1.0 / p.t1_s
    r2 = 1.0 / p.t2_s

    generator = np.zeros((b0_hz.size, 4, 4), dtype=float)
    generator[:, 0, 0] = -r2
    generator[:, 0, 1] = dw
    generator[:, 0, 2] = -wy
    generator[:, 1, 0] = -dw
    generator[:, 1, 1] = -r2
    generator[:, 1, 2] = wx
    generator[:, 2, 0] = wy
    generator[:, 2, 1] = -wx
    generator[:, 2, 2] = -r1
    generator[:, 2, 3] = r1 * p.m0
    return expm_pade13(generator * operation.duration_s)


_PADE13 = np.array(
    [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ]
)
_PADE13_THETA = 5.371920351148152


def expm_pade13(matrix: np.ndarray) -> np.ndarray:
    """Scaling-and-squaring Pade approximation of order 13.

    This is the propagator used for the nominal-amplitude paper grids. On the
    full 101 by 101 field grid it matches ``scipy.linalg.expm`` to about
    1e-14 in the propagated signal.
    """

    one_norm = np.max(np.sum(np.abs(matrix), axis=-2), axis=-1)
    scale = int(math.ceil(math.log2(max(float(np.max(one_norm)) / _PADE13_THETA, 1.0))))
    scaled = matrix / (2**scale)
    identity = np.eye(matrix.shape[-1])
    square = scaled @ scaled
    fourth = square @ square
    sixth = fourth @ square
    odd = scaled @ (
        sixth @ (_PADE13[13] * sixth + _PADE13[11] * fourth + _PADE13[9] * square)
        + _PADE13[7] * sixth
        + _PADE13[5] * fourth
        + _PADE13[3] * square
        + _PADE13[1] * identity
    )
    even = (
        sixth @ (_PADE13[12] * sixth + _PADE13[10] * fourth + _PADE13[8] * square)
        + _PADE13[6] * sixth
        + _PADE13[4] * fourth
        + _PADE13[2] * square
        + _PADE13[0] * identity
    )
    result = np.linalg.solve(even - odd, even + odd)
    for _ in range(scale):
        result = result @ result
    return result


def grid_sha256(values: np.ndarray) -> str:
    """Hash every stored grid value, not only the endpoints and length."""

    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def propagator_cache_key(
    p: Params,
    operation: Operation,
    b0_hz: np.ndarray,
    b1_scale: np.ndarray,
) -> tuple:
    return (
        round(p.t1_s, 12),
        round(p.t2_s, 12),
        round(p.m0, 12),
        grid_sha256(b0_hz),
        grid_sha256(b1_scale),
        int(np.asarray(b0_hz).size),
        int(np.asarray(b1_scale).size),
        round(operation.duration_s, 12),
        round(operation.fx_hz, 9),
        round(operation.fy_hz, 9),
    )


def simulate_preparation(
    p: Params,
    operations: list[Operation],
    b0_hz: np.ndarray,
    b1_scale: np.ndarray,
    cache: dict[tuple, np.ndarray],
    initial_axis: int = 2,
    readout_axis: int = 2,
) -> np.ndarray:
    if not operations:
        return np.ones_like(b0_hz, dtype=float)
    magnetization = np.zeros((b0_hz.size, 4), dtype=float)
    magnetization[:, initial_axis] = 1.0
    magnetization[:, 3] = 1.0
    for operation in operations:
        key = propagator_cache_key(p, operation, b0_hz, b1_scale)
        if key not in cache:
            cache[key] = finite_propagator(p, operation, b0_hz, b1_scale)
        magnetization = np.einsum("nij,nj->ni", cache[key], magnetization)
    return np.abs(magnetization[:, readout_axis])


def fit_exp_batch(tsl_s: np.ndarray, signal: np.ndarray) -> dict[str, np.ndarray]:
    values = np.clip(np.abs(np.asarray(signal, dtype=float)), 1e-10, None)
    n_grid = values.shape[0]
    times = np.asarray(tsl_s, dtype=float)
    if times.ndim == 1:
        times = times[None, :]
    t = np.broadcast_to(times, values.shape)
    log_values = np.log(values)
    t_mean = t.mean(axis=1, keepdims=True)
    y_mean = log_values.mean(axis=1, keepdims=True)
    denominator = np.sum((t - t_mean) ** 2, axis=1)
    slope = np.sum((t - t_mean) * (log_values - y_mean), axis=1) / np.maximum(
        denominator, 1e-15
    )
    tfit = np.where(slope < -1e-8, -1 / slope, 0.1)
    tfit = np.clip(tfit, 1e-3, 10.0)
    log_tfit = np.log(tfit)
    log_s0 = np.mean(log_values + t / tfit[:, None], axis=1)

    for _ in range(18):
        tfit = np.exp(log_tfit)
        prediction = np.exp(log_s0[:, None] - t / tfit[:, None])
        residual = values - prediction
        j0 = prediction
        j1 = prediction * t / tfit[:, None]
        a00 = np.sum(j0 * j0, axis=1) + 1e-10
        a01 = np.sum(j0 * j1, axis=1)
        a11 = np.sum(j1 * j1, axis=1) + 1e-10
        b0 = np.sum(j0 * residual, axis=1)
        b1 = np.sum(j1 * residual, axis=1)
        determinant = np.maximum(a00 * a11 - a01 * a01, 1e-16)
        d0 = (b0 * a11 - b1 * a01) / determinant
        d1 = (b1 * a00 - b0 * a01) / determinant
        log_s0 += np.clip(d0, -1.0, 1.0)
        log_tfit += np.clip(d1, -1.0, 1.0)
        log_tfit = np.clip(log_tfit, math.log(1e-3), math.log(10.0))

    tfit = np.exp(log_tfit)
    s0 = np.exp(log_s0)
    fitted = s0[:, None] * np.exp(-t / tfit[:, None])
    residual = values - fitted
    rss = np.sum(residual**2, axis=1)
    tss = np.sum((values - values.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = np.where(tss > 1e-15, 1 - rss / tss, np.where(rss < 1e-15, 1.0, np.nan))
    rmse = np.sqrt(rss / values.shape[1])
    return {
        "Tfit_s": tfit,
        "S0_fit": s0,
        "fit_R2": r2,
        "fit_RMSE": rmse,
        "fitted": fitted.reshape(n_grid, -1),
    }


def foltz_fixed_delta_times(
    p: Params,
    nominal_tsl_s: np.ndarray,
    pulse_counts: np.ndarray,
) -> np.ndarray:
    """Return preparation times corrected using the fixed Foltz delta."""

    composite_width_s = 2 * p.hard_90_width_s + p.hard_180_width_s
    correction = (
        pulse_counts
        * 0.5
        * composite_width_s
        * p.cpmg_foltz_delta
    )
    corrected = nominal_tsl_s - correction
    corrected[0] = 0.0
    return corrected


def fit_foltz_fixed_delta_batch(
    p: Params,
    nominal_tsl_s: np.ndarray,
    pulse_counts: np.ndarray,
    signal: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fit a decay using the fixed-delta Foltz time correction."""

    corrected_1d = foltz_fixed_delta_times(p, nominal_tsl_s, pulse_counts)
    corrected = np.broadcast_to(corrected_1d[None, :], signal.shape).copy()
    return fit_exp_batch(corrected, signal), corrected


def operation_rotation(operation: Operation, rf_scale: float, duration_s: float | None = None) -> np.ndarray:
    duration = operation.duration_s if duration_s is None else duration_s
    field = np.array([[operation.fx_hz * rf_scale, operation.fy_hz * rf_scale, 0.0]])
    return rodrigues_matrices(field, duration)[0]


def sample_control_modulation(
    operations: list[Operation], tsl_s: float, requested_dt_s: float, rf_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n_time = max(32, int(round(tsl_s / requested_dt_s)))
    dt_s = tsl_s / n_time
    times_s = (np.arange(n_time) + 0.5) * dt_s
    rf_x = np.zeros(n_time, dtype=float)
    rf_y = np.zeros(n_time, dtype=float)
    modulation = np.empty((n_time, 3), dtype=float)
    control_at_start = np.eye(3)
    operation_start = 0.0
    assigned = np.zeros(n_time, dtype=bool)

    for operation in operations:
        operation_end = operation_start + operation.duration_s
        selected = (times_s >= operation_start - 1e-12) & (times_s < operation_end - 1e-12)
        if np.any(selected):
            local_times = times_s[selected] - operation_start
            field = np.tile(
                [operation.fx_hz * rf_scale, operation.fy_hz * rf_scale, 0.0],
                (local_times.size, 1),
            )
            local_rotation = rodrigues_matrices(field, local_times)
            control = np.matmul(local_rotation, control_at_start)
            modulation[selected] = control[:, 2, :]
            rf_x[selected] = operation.fx_hz
            rf_y[selected] = operation.fy_hz
            assigned[selected] = True
        control_at_start = operation_rotation(operation, rf_scale) @ control_at_start
        operation_start = operation_end

    if not np.all(assigned):
        raise RuntimeError("Control-modulation sampling left unassigned time points")
    if not np.isclose(operation_start, tsl_s, atol=1e-10):
        raise RuntimeError(f"Core duration {operation_start:g} differs from TSL {tsl_s:g}")
    return times_s, modulation, rf_x, rf_y, dt_s


def finite_time_spectrum(
    p: Params, operations: list[Operation], tsl_s: float
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    times_s, modulation, rf_x, rf_y, dt_s = sample_control_modulation(
        operations, tsl_s, p.spectrum_dt_s
    )
    n_time = times_s.size
    n_fft_min = p.spectrum_zero_pad_factor * n_time
    n_fft = 1 << int(math.ceil(math.log2(n_fft_min)))
    transform = dt_s * np.fft.fft(modulation, n=n_fft, axis=0)
    frequencies_hz = np.fft.fftfreq(n_fft, d=dt_s)
    order = np.argsort(frequencies_hz)
    frequencies_hz = frequencies_hz[order]
    transform = transform[order]
    vector_power = np.sum(np.abs(transform) ** 2, axis=1)
    q_complex = modulation[:, 2] + 1j * modulation[:, 1]
    q_transform = dt_s * np.fft.fft(q_complex, n=n_fft)[order]
    p_q_t_per_hz = np.abs(q_transform) ** 2 / tsl_s
    p_q_t_per_rad_s = p_q_t_per_hz / (2 * np.pi)
    keep = np.abs(frequencies_hz) <= p.spectrum_max_hz
    spectrum = pd.DataFrame(
        {
            "frequency_Hz": frequencies_hz[keep],
            "vectorPower_s2": vector_power[keep],
            "qPower_s2": np.abs(q_transform[keep]) ** 2,
            "pQT_per_Hz": p_q_t_per_hz[keep],
            "pQT_per_rad_s": p_q_t_per_rad_s[keep],
            "Cx_real_s": transform[keep, 0].real,
            "Cx_imag_s": transform[keep, 0].imag,
            "Cy_real_s": transform[keep, 1].real,
            "Cy_imag_s": transform[keep, 1].imag,
            "Cz_real_s": transform[keep, 2].real,
            "Cz_imag_s": transform[keep, 2].imag,
        }
    )

    positive = spectrum[spectrum.frequency_Hz.ge(0)].copy()
    total_energy = float(np.trapezoid(spectrum.vectorPower_s2, spectrum.frequency_Hz))
    target_band_negative = spectrum[
        spectrum.frequency_Hz.between(-100.0, -20.0, inclusive="both")
    ]
    target_band_positive = spectrum[
        spectrum.frequency_Hz.between(20.0, 100.0, inclusive="both")
    ]
    target_band_energy = float(
        np.trapezoid(
            target_band_negative.vectorPower_s2,
            target_band_negative.frequency_Hz,
        )
        + np.trapezoid(
            target_band_positive.vectorPower_s2,
            target_band_positive.frequency_Hz,
        )
    )
    nonzero = positive[positive.frequency_Hz.ge(max(1.0, 0.5 / tsl_s))]
    dominant = nonzero.iloc[int(np.argmax(nonzero.vectorPower_s2.to_numpy()))]
    dc_vector = np.mean(modulation, axis=0)
    _, plus_modulation, _, _, _ = sample_control_modulation(
        operations, tsl_s, p.spectrum_dt_s, 1 + p.b1_derivative_step
    )
    _, minus_modulation, _, _, _ = sample_control_modulation(
        operations, tsl_s, p.spectrum_dt_s, 1 - p.b1_derivative_step
    )
    dc_derivative = (
        np.mean(plus_modulation, axis=0) - np.mean(minus_modulation, axis=0)
    ) / (2 * p.b1_derivative_step)

    q_positive = spectrum[
        spectrum.frequency_Hz.between(20.0, 100.0, inclusive="both")
    ]
    q_negative = spectrum[
        spectrum.frequency_Hz.between(-100.0, -20.0, inclusive="both")
    ]
    q_positive_energy = float(np.trapezoid(q_positive.pQT_per_Hz, q_positive.frequency_Hz))
    q_negative_energy = float(np.trapezoid(q_negative.pQT_per_Hz, q_negative.frequency_Hz))
    q_total_energy = float(np.trapezoid(spectrum.pQT_per_Hz, spectrum.frequency_Hz))
    q_time_energy = float(dt_s * np.sum(np.abs(q_complex) ** 2))
    q_balance = min(q_positive_energy, q_negative_energy) / max(
        q_positive_energy, q_negative_energy, 1e-15
    )
    q_nonzero = spectrum[
        np.abs(spectrum.frequency_Hz).ge(max(1.0, 0.5 / tsl_s))
    ]
    q_dominant = q_nonzero.iloc[int(np.argmax(q_nonzero.pQT_per_Hz.to_numpy()))]

    metrics: dict[str, float] = {
        "TSL_ms": 1e3 * tsl_s,
        "frequencyResolution_Hz": float(abs(frequencies_hz[1] - frequencies_hz[0])),
        "dominantPositiveFrequency_Hz": float(dominant.frequency_Hz),
        "dominantPositivePower_s2": float(dominant.vectorPower_s2),
        "targetBand20to100Fraction": target_band_energy / max(total_energy, 1e-15),
        "dcVectorNorm": float(np.linalg.norm(dc_vector)),
        "b1DerivativeDcVectorNorm": float(np.linalg.norm(dc_derivative)),
        "dominantQFrequency_Hz": float(q_dominant.frequency_Hz),
        "dominantQFrequencyMagnitude_Hz": float(abs(q_dominant.frequency_Hz)),
        "dominantQPower_s2": float(q_dominant.qPower_s2),
        "dominantPQT_per_Hz": float(q_dominant.pQT_per_Hz),
        "qTargetBand20to100Fraction": (q_positive_energy + q_negative_energy)
        / max(q_total_energy, 1e-15),
        "qPositiveNegativeBandBalance": q_balance,
        "qCapturedPowerFractionWithin500Hz": q_total_energy
        / max(q_time_energy / tsl_s, 1e-15),
        "capturedPowerFractionWithin500Hz": total_energy / max(tsl_s, 1e-15),
    }
    abs_frequency = np.abs(spectrum.frequency_Hz.to_numpy())
    power = spectrum.pQT_per_Hz.to_numpy()
    for corner_hz in (20.0, 50.0, 100.0):
        lorentzian = 1.0 / (1.0 + (abs_frequency / corner_hz) ** 2)
        response = float(np.trapezoid(power * lorentzian, spectrum.frequency_Hz))
        metrics[f"normalizedLorentzianResponse_fc{corner_hz:g}Hz"] = response / max(
            q_total_energy, 1e-15
        )

    waveform = pd.DataFrame(
        {
            "time_ms": 1e3 * times_s,
            "RFx_Hz": rf_x,
            "RFy_Hz": rf_y,
            "control_x": modulation[:, 0],
            "control_y": modulation[:, 1],
            "control_z": modulation[:, 2],
        }
    )
    return spectrum, metrics, waveform


def locking_amplitude_hz(method_key: str, target_hz: float, p: Params | None = None) -> float:
    """Return the constant locking amplitude used for a target frequency.

    CW, composite, balanced, and quadruple-refocused preparations use
    ``target_hz`` itself. CPMG has no constant lock; its timing frequency is
    ``1/(2*tau)=target_hz`` and its refocusing amplitude stays at
    ``Params.rf_peak_hz``. Sinusoidal modulation uses frequency ``target_hz``
    and peak amplitude ``beta*target_hz``.
    """

    if method_key == "sinusoidal":
        params = p if p is not None else Params()
        return params.sinusoidal_beta * target_hz
    if method_key == "phase_cycled_cpmg":
        return target_hz
    if method_key in {"ordinary", "rotary_echo", "balanced", "quadruple_refocused", "triple_refocused"}:
        return target_hz
    raise ValueError(f"Unknown method: {method_key}")


def rf_metrics(
    p: Params,
    method_key: str,
    operations: list[Operation],
    tsl_s: float,
    target_hz: float,
) -> dict[str, float]:
    amplitude_hz = np.array([math.hypot(op.fx_hz, op.fy_hz) for op in operations])
    durations_s = np.array([op.duration_s for op in operations])
    energy_hz2_s = float(np.sum(durations_s * amplitude_hz**2))
    high_power_duration_s = float(
        np.sum(durations_s[amplitude_hz > max(150.0, 2.0 * target_hz)])
    )
    if method_key == "rotary_echo":
        n_refocusing = 1
    elif method_key in {"balanced", "triple_refocused", "quadruple_refocused"}:
        n_refocusing = {"balanced": 2, "triple_refocused": 3, "quadruple_refocused": 4}[method_key]
    elif method_key == "phase_cycled_cpmg":
        n_refocusing = sum(":180" in op.label for op in operations)
    else:
        n_refocusing = 0
    return {
        "rfPeak_Hz": float(np.max(amplitude_hz, initial=0.0)),
        "rfRMS_Hz": math.sqrt(energy_hz2_s / tsl_s),
        "rfEnergy_Hz2_s": energy_hz2_s,
        "rfOnDutyFraction": float(np.sum(durations_s[amplitude_hz > 0]) / tsl_s),
        "highPowerPulseDutyFraction": high_power_duration_s / tsl_s,
        "nRefocusingPulses": int(n_refocusing),
    }


# Order and display styles for the six paper preparations.
# The composite preparation keeps the historical key "rotary_echo".
_PAPER_STYLE = {
    "ordinary": ("#0072B2", "-", "o", "Conventional continuous-wave spin lock", "CW"),
    "rotary_echo": ("#56B4E9", "--", "s", "Composite spin lock", "Composite"),
    "balanced": ("#E69F00", "-.", "^", "Balanced spin lock", "Balanced"),
    "quadruple_refocused": ("#CC79A7", (0, (5, 1)), "v", "Quadruple-refocused spin lock", "Quadruple"),
    "sinusoidal": ("#009E73", "-", "D", "Sinusoidal spin lock", "Sinusoidal"),
    "phase_cycled_cpmg": ("#D55E00", (0, (3, 1, 1, 1)), "P", "CPMG-MLEV16", "CPMG"),
}

PAPER_METHODS = tuple(
    Method(key, key, label, short_label, color, linestyle, marker)
    for key, (color, linestyle, marker, label, short_label) in _PAPER_STYLE.items()
)
PAPER_METHOD_KEYS = tuple(method.key for method in PAPER_METHODS)
EXCLUDED_FROM_PAPER = ("triple_refocused",)
METHOD_BY_KEY = {method.key: method for method in PAPER_METHODS}
TARGETS_HZ = (20.0, 50.0, 100.0)
CPMG_PULSE_COUNTS = np.array([0, 4, 8, 16], dtype=int)
TSL_MS_BY_TARGET = {
    target_hz: 1e3 * CPMG_PULSE_COUNTS / (2.0 * target_hz)
    for target_hz in TARGETS_HZ
}
SUMMARY_B0_WINDOWS_HZ = (100.0, 250.0)


def field_scope_label(b0_limit_hz: float, b1_min_pct: float, b1_max_pct: float) -> str:
    if np.isclose(b1_min_pct, -b1_max_pct):
        b1_scope = f"B1_pm{b1_max_pct:g}pct"
    else:
        b1_scope = f"B1_{b1_min_pct:g}_to_{b1_max_pct:g}pct"
    return f"absB0_le_{b0_limit_hz:g}Hz_{b1_scope}"


def relaxation_error_windows(
    b0_hz: np.ndarray,
    error_pct: np.ndarray,
    windows_hz: tuple[float, ...] = SUMMARY_B0_WINDOWS_HZ,
) -> list[dict[str, float | int | bool]]:
    """Return both candidate |B0| windows and select neither as the paper statistic."""

    b0 = np.asarray(b0_hz, dtype=float).ravel()
    error = np.asarray(error_pct, dtype=float).ravel()
    if b0.shape != error.shape:
        raise ValueError("B0 offsets and errors must have the same number of points")
    rows: list[dict[str, float | int | bool]] = []
    for limit in windows_hz:
        selected = error[np.abs(b0) <= float(limit) + 1e-9]
        rows.append(
            {
                "abs_b0_max_hz": float(limit),
                "n_points": int(selected.size),
                "median_abs_error_pct": (
                    float(np.median(np.abs(selected))) if selected.size else float("nan")
                ),
                "manuscript_statistic_selected": False,
            }
        )
    return rows
