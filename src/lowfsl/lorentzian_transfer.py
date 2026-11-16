"""Uncorrected and RF-spectrum-corrected Lorentzian transfer, paper method.

Both models are fitted by unweighted least squares to conventional CW
(ordinary) rates only, then used to predict the other preparations
without refitting.

Uncorrected (traditional) model, with omega = 2*pi*f:

    R(f) = R_bg + A / (1 + omega^2 tau^2)

Corrected (spectrum) model: the predicted signal at each paper duration T is

    S(T) = exp(-R_bg T - A' * G(T, tau))

where G is the double time integral of exp(-|t-t'|/tau) *
cos(phi(t)-phi(t')). G is evaluated with the same causal exponential
filter and midpoint quadrature as the paper fitting script, then S(T)
is refitted monoexponentially with the same Gauss-Newton scheme.

Default preparations are CW, composite, balanced, and ideal-pulse CPMG.
This module does not plot and does not ship previously computed rate tables.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter

from .schedules import PAPER_PREPARATIONS, preparation_times_s

TAU_GRID = np.geomspace(0.001, 0.09, 48)
LOG_TAU_GRID = np.log(TAU_GRID)
TRADITIONAL_X0_TAU_S = 0.01
TRADITIONAL_BOUNDS = ([11.0, 0.0, 0.001], [15.0, 10.0, 0.09])
SPECTRUM_X0 = np.array([12.5, 25.0, 0.013])
SPECTRUM_BOUNDS = ([11.0, 0.0, 0.001], [15.0, 400.0, 0.09])
LEAST_SQUARES_OPTIONS = dict(max_nfev=75, ftol=5e-6, xtol=5e-6, gtol=5e-6)


def modulation_phase(method: str, frequency_hz: float, duration_s: float, time_s: np.ndarray) -> np.ndarray:
    """Ideal-preparation phase phi(t) used by the paper spectrum model."""
    method = method.lower()
    t = np.asarray(time_s, dtype=float)
    f = float(frequency_hz)
    T = float(duration_s)
    if method == "ordinary":
        return 2 * np.pi * f * t
    if method == "balanced":
        a = T / 4.0
        b = 3.0 * T / 4.0
        return np.where(
            t < a,
            2 * np.pi * f * t,
            np.where(
                t < b,
                2 * np.pi * f * a + np.pi - 2 * np.pi * f * (t - a),
                2 * np.pi * f * a + np.pi - 2 * np.pi * f * (b - a) - np.pi + 2 * np.pi * f * (t - b),
            ),
        )
    if method == "composite":
        return np.where(t < T / 2.0, 2 * np.pi * f * t, np.pi + 2 * np.pi * f * (T - t))
    if method == "cpmg":
        half = 1.0 / (2.0 * f)
        count = np.maximum(0, np.floor((t - half / 2.0) / half).astype(int) + 1)
        return np.pi * count
    raise ValueError(f"unsupported preparation: {method}")


def correlation_integral(method: str, frequency_hz: float, duration_s: float, tau_s: float, hmax: float = 0.00025) -> float:
    """G(T, tau) = ∫∫ exp(-|t-t'|/tau) cos(Δphi) dt dt', paper quadrature."""
    if duration_s <= 0:
        return 0.0
    n = max(2, int(np.ceil(duration_s / hmax)))
    h = duration_s / n
    t = (np.arange(n) + 0.5) * h
    u = np.exp(1j * modulation_phase(method, frequency_hz, duration_s, t))
    z = lfilter([1.0], [1.0, -np.exp(-h / tau_s)], np.conjugate(u))
    return max(0.0, float((2.0 * np.real(np.sum(u * z)) - n) * h * h))


def fit_monoexponential_rate(time_s: np.ndarray, signal: np.ndarray) -> float:
    """Nine-step Gauss-Newton fit of S0*exp(-R T), paper implementation."""
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    p = np.array([1.0, 12.5])
    for _ in range(9):
        decay = np.exp(-p[1] * time_s)
        residual = p[0] * decay - signal
        j0 = decay
        j1 = -p[0] * time_s * decay
        h00 = j0 @ j0
        h01 = j0 @ j1
        h11 = j1 @ j1
        g0 = j0 @ residual
        g1 = j1 @ residual
        det = h00 * h11 - h01 * h01
        if det < 1e-22:
            break
        ds = (h11 * g0 - h01 * g1) / det
        dr = (-h01 * g0 + h00 * g1) / det
        p[0] = max(0.0, p[0] - ds)
        p[1] = max(0.0, p[1] - dr)
        if abs(dr) < 1e-9:
            break
    return float(p[1])


def traditional_rate(params: np.ndarray, frequency_hz: float) -> float:
    r_bg, amplitude, tau_s = params
    return float(r_bg + amplitude / (1.0 + (2 * np.pi * frequency_hz * tau_s) ** 2))


@dataclass
class SpectrumKernel:
    times: dict[tuple[str, float], np.ndarray]
    values: dict[tuple[str, float], np.ndarray]

    def rate(self, params: np.ndarray, key: tuple[str, float]) -> float:
        r_bg, amplitude, tau_s = params
        times = self.times[key]
        table = self.values[key]
        j = int(np.clip(np.searchsorted(LOG_TAU_GRID, np.log(tau_s)) - 1, 0, len(LOG_TAU_GRID) - 2))
        weight = (np.log(tau_s) - LOG_TAU_GRID[j]) / (LOG_TAU_GRID[j + 1] - LOG_TAU_GRID[j])
        g = (1.0 - weight) * table[j] + weight * table[j + 1]
        return fit_monoexponential_rate(times, np.exp(-r_bg * times - amplitude * g))


def build_spectrum_kernel(keys: list[tuple[str, float]], dt_ref_us: float = 12.5) -> SpectrumKernel:
    times = {key: preparation_times_s(key[0], key[1], dt_ref_us) for key in keys}
    values = {}
    for key, t_grid in times.items():
        values[key] = np.array(
            [[correlation_integral(key[0], key[1], float(duration), float(tau)) for duration in t_grid] for tau in TAU_GRID],
            dtype=np.float64,
        )
    return SpectrumKernel(times=times, values=values)


def _valid_rows(frame: pd.DataFrame, preparations: tuple[str, ...]) -> pd.DataFrame:
    required = {"preparation", "f_Hz", "R_fit_s_1", "valid"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"rate table missing columns: {sorted(missing)}")
    data = frame.copy()
    data["preparation"] = data["preparation"].astype(str).str.lower()
    data["f_Hz"] = pd.to_numeric(data["f_Hz"])
    data["R_fit_s_1"] = pd.to_numeric(data["R_fit_s_1"])
    valid = data["valid"].isin([True, "True", "true", "TRUE", 1, "1"])
    data = data[valid & data["R_fit_s_1"].notna() & data["preparation"].isin(preparations)].copy()
    return data


def _metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rmse = float(np.sqrt(np.mean((predicted - y) ** 2)))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if denom == 0 else float(1.0 - np.sum((predicted - y) ** 2) / denom)
    return {"n": int(len(y)), "RMSE_s1": rmse, "R2": r2}


def _window_rows(frame: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    return frame[(frame["frequency_Hz"] >= low) & (frame["frequency_Hz"] <= high)]


def fit_transfer(
    rates: pd.DataFrame,
    preparations: tuple[str, ...] = PAPER_PREPARATIONS,
    dt_ref_us: float = 12.5,
) -> dict[str, pd.DataFrame]:
    """Fit both models to CW rates and predict the other preparations."""
    data = _valid_rows(rates, preparations)
    if "ordinary" not in set(data["preparation"]):
        raise ValueError("CW/ordinary rates are required for the paper fit")
    keys = sorted({(str(m), float(f)) for m, f in zip(data["preparation"], data["f_Hz"])})
    kernel = build_spectrum_kernel(keys, dt_ref_us)
    ordinary = data[data["preparation"] == "ordinary"].sort_values("f_Hz")
    if ordinary.empty:
        raise ValueError("no valid CW/ordinary rows")
    frequencies = ordinary["f_Hz"].to_numpy()
    observed = ordinary["R_fit_s_1"].to_numpy()
    parameters = []
    metrics = []
    predictions = []
    models = {
        "traditional": (
            lambda params, key: traditional_rate(params, key[1]),
            np.array([12.5, float(observed.max() - observed.min()), TRADITIONAL_X0_TAU_S]),
            TRADITIONAL_BOUNDS,
        ),
        "spectrum": (
            lambda params, key: kernel.rate(params, key),
            SPECTRUM_X0.copy(),
            SPECTRUM_BOUNDS,
        ),
    }
    fitted = {}
    for model, (fun, x0, bounds) in models.items():
        result = least_squares(
            lambda p: np.array([fun(p, ("ordinary", float(f))) for f in frequencies]) - observed,
            x0,
            bounds=bounds,
            **LEAST_SQUARES_OPTIONS,
        )
        params = result.x
        fitted[model] = params
        parameters.append(
            dict(
                model=model,
                Rbg_s1=float(params[0]),
                A=float(params[1]),
                tau_ms=float(params[2] * 1000.0),
                successful=bool(result.success),
            )
        )
        for prep in preparations:
            subset = data[data["preparation"] == prep].sort_values("f_Hz")
            if subset.empty:
                continue
            x = subset["f_Hz"].to_numpy()
            y = subset["R_fit_s_1"].to_numpy()
            yp = np.array([fun(params, (prep, float(f))) for f in x])
            for low, high, window in ((1.0, 100.0, "1-100"), (20.0, 100.0, "20-100")):
                mask = (x >= low) & (x <= high)
                if not np.any(mask):
                    continue
                metrics.append(dict(model=model, preparation=prep, window_Hz=window, **_metrics(y[mask], yp[mask])))
            for f, y0, y1 in zip(x, y, yp):
                predictions.append(
                    dict(
                        model=model,
                        preparation=prep,
                        frequency_Hz=float(f),
                        simulation_s1=float(y0),
                        prediction_s1=float(y1),
                        residual_s1=float(y0 - y1),
                    )
                )
    pred = pd.DataFrame(predictions)
    pdev_rows = []
    ordinary_sim = {
        float(row.frequency_Hz): float(row.simulation_s1)
        for row in pred[(pred.model == "traditional") & (pred.preparation == "ordinary")].itertuples()
    }
    for model, params in fitted.items():
        pool_sim = []
        pool_pred = []
        for prep in preparations:
            if prep == "ordinary":
                continue
            sim_dev = []
            pred_dev = []
            subset = pred[(pred.model == model) & (pred.preparation == prep)]
            for row in subset.itertuples():
                if row.frequency_Hz not in ordinary_sim:
                    continue
                sim_dev.append(row.simulation_s1 - ordinary_sim[row.frequency_Hz])
                if model == "traditional":
                    pred_dev.append(0.0)
                else:
                    pred_dev.append(
                        kernel.rate(params, (prep, float(row.frequency_Hz)))
                        - kernel.rate(params, ("ordinary", float(row.frequency_Hz)))
                    )
            sim_dev = np.array(sim_dev)
            pred_dev = np.array(pred_dev)
            if sim_dev.size == 0 or np.sum(sim_dev**2) == 0:
                continue
            explained = 100.0 * (1.0 - np.sum((sim_dev - pred_dev) ** 2) / np.sum(sim_dev**2))
            pdev_rows.append(dict(model=model, preparation=prep, PDEV_pct=float(explained), n=int(sim_dev.size)))
            pool_sim.extend(sim_dev)
            pool_pred.extend(pred_dev)
        pool_sim = np.array(pool_sim)
        pool_pred = np.array(pool_pred)
        if pool_sim.size and np.sum(pool_sim**2) > 0:
            explained = 100.0 * (1.0 - np.sum((pool_sim - pool_pred) ** 2) / np.sum(pool_sim**2))
            pdev_rows.append(
                dict(
                    model=model,
                    preparation="pooled_composite_balanced_cpmg",
                    PDEV_pct=float(explained),
                    n=int(pool_sim.size),
                )
            )
    return {
        "parameters": pd.DataFrame(parameters),
        "metrics": pd.DataFrame(metrics),
        "predictions": pred,
        "pdev": pd.DataFrame(pdev_rows),
    }
