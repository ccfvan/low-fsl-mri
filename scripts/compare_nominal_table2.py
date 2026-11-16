"""Reproduce the paper Table 2 medians and the 160 ms spectral peaks.

CW, composite, balanced, and quadruple-refocused preparations use a constant
locking amplitude equal to the target frequency. CPMG uses pulse spacing
1/(2*target) and 625 Hz refocusing. Sinusoidal modulation frequency equals the
target and its peak is beta*target; its Table 2 values come from a separate
continuous-waveform integration and are not recomputed here.

Run from the repository root:

    python scripts/compare_nominal_table2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lowfsl.engine as engine

# Paper Table 2 after setting CW, composite, balanced, and quadruple-refocused
# locking amplitudes equal to the target. |B0| <= 250 Hz and |B1| <= 25%.
EXPECTED = {
    ("ordinary", 20.0): (24.782064892710288, 65.61151037508803, 20.027160644531246),
    ("ordinary", 50.0): (40.40506563264614, 79.5230590655551, 49.97253417968749),
    ("ordinary", 100.0): (44.94129121798569, 88.17281011574481, 99.94506835937499),
    ("rotary_echo", 20.0): (9.390144029385583, 46.64439614390468, 22.315979003906246),
    ("rotary_echo", 50.0): (14.95405282411224, 48.308494297323115, 49.59106445312499),
    ("rotary_echo", 100.0): (15.921468138097495, 43.8523072512639, 99.75433349609374),
    ("balanced", 20.0): (24.50628879117761, 67.06250515067943, -21.362304687499996),
    ("balanced", 50.0): (29.85390941519125, 79.36535749682182, -49.59106445312499),
    ("balanced", 100.0): (23.66352698072297, 76.79964601472233, -99.75433349609374),
    ("quadruple_refocused", 20.0): (12.481049425368429, 45.77657553928951, -24.223327636718746),
    ("quadruple_refocused", 50.0): (21.570811715473393, 56.03024411498081, -49.78179931640624),
    ("quadruple_refocused", 100.0): (26.546612890146037, 52.24625249734069, -99.94506835937499),
    ("phase_cycled_cpmg", 20.0): (1.8042, 19.652, -20.027160644531246),
    ("phase_cycled_cpmg", 50.0): (2.665, 24.99, -50.16326904296874),
    ("phase_cycled_cpmg", 100.0): (3.0711, 21.192, -100.13580322265624),
}
# Sinusoidal Table 2 is listed for reference. This command does not rerun it; use compare_sinusoidal_table2.py.
SINUSOIDAL_REFERENCE = {
    20.0: (38.312090033175785, 80.15381461282769),
    50.0: (45.80025371630807, 95.55620513288355),
    100.0: (47.35163938050045, 111.57382945283057),
}
RERUN = (
    "ordinary",
    "rotary_echo",
    "balanced",
    "quadruple_refocused",
    "phase_cycled_cpmg",
)


def grid(p: engine.Params):
    b0, b1 = np.meshgrid(p.b0_hz, p.b1_pct, indexing="ij")
    return b0.ravel(), b1.ravel(), 1.0 + b1.ravel() / 100.0


def one_case(p, method_key, target_hz, b0, b1_pct, b1_scale, cache):
    columns = []
    for tsl_ms in engine.TSL_MS_BY_TARGET[target_hz]:
        operations = engine.build_core(p, method_key, float(tsl_ms) * 1e-3, target_hz)
        columns.append(
            engine.simulate_preparation(p, operations, b0, b1_scale, cache, initial_axis=0, readout_axis=0)
        )
    signal = np.column_stack(columns)
    if method_key == "phase_cycled_cpmg":
        fit, _ = engine.fit_foltz_fixed_delta_batch(
            p, engine.TSL_MS_BY_TARGET[target_hz] * 1e-3, engine.CPMG_PULSE_COUNTS, signal
        )
    else:
        fit = engine.fit_exp_batch(engine.TSL_MS_BY_TARGET[target_hz] * 1e-3, signal)
    error = 100.0 * np.abs(fit["Tfit_s"] / p.t2_s - 1.0)
    selected = error[np.abs(b0) <= 250.0]
    spectrum_ops = spectrum_operations(p, method_key, target_hz)
    _, metrics, _ = engine.finite_time_spectrum(p, spectrum_ops, 0.160)
    return float(np.median(selected)), float(np.quantile(selected, 0.95)), float(metrics["dominantQFrequency_Hz"])


def spectrum_operations(p, method_key, target_hz):
    """160 ms spectrum core. CPMG keeps only pulses that fit the window."""

    if method_key != "phase_cycled_cpmg":
        return engine.build_core(p, method_key, 0.160, target_hz)
    spacing_s = 1.0 / (2.0 * target_hz)
    width_s = 2.0 * p.hard_180_width_s
    operations = []
    cursor = 0.0
    pulse_index = 0
    while True:
        center = (pulse_index + 0.5) * spacing_s
        start = center - width_s / 2.0
        end = center + width_s / 2.0
        if end > 0.160 + 1e-12:
            break
        engine.append_op(operations, start - cursor, 0.0, 0.0, "gap")
        sign = engine.MLEV16_SIGNS[pulse_index % len(engine.MLEV16_SIGNS)]
        engine.append_composite_180(operations, p, "x", sign, f"MLEV16_{pulse_index + 1}:{sign:+d}X")
        cursor = end
        pulse_index += 1
    engine.append_op(operations, 0.160 - cursor, 0.0, 0.0, "gap")
    return operations


def main() -> None:
    p = engine.Params()
    b0, b1_pct, b1_scale = grid(p)
    if b0.size != 10201:
        raise SystemExit(f"unexpected grid size {b0.size}")
    rows = []
    failed = False
    for method_key in RERUN:
        cache: dict = {}
        for target_hz in engine.TARGETS_HZ:
            median, p95, peak = one_case(p, method_key, target_hz, b0, b1_pct, b1_scale, cache)
            expected_median, expected_p95, expected_peak = EXPECTED[(method_key, target_hz)]
            median_ok = abs(median - expected_median) < 5e-3
            p95_ok = abs(p95 - expected_p95) < 5e-3
            peak_ok = abs(peak - expected_peak) < 1e-6
            failed |= not (median_ok and p95_ok and peak_ok)
            rows.append(
                {
                    "method": method_key,
                    "target_Hz": target_hz,
                    "locking_Hz": engine.locking_amplitude_hz(method_key, target_hz, p),
                    "median_abs_error_pct": median,
                    "p95_abs_error_pct": p95,
                    "dominant_peak_160ms_Hz": peak,
                    "expected_median": expected_median,
                    "expected_p95": expected_p95,
                    "expected_peak_Hz": expected_peak,
                    "match": median_ok and p95_ok and peak_ok,
                }
            )
            print(
                f"{method_key:22} {target_hz:6.0f}  {median:8.4f} ({p95:8.4f})  peak {peak:9.4f}  "
                f"{'OK' if rows[-1]['match'] else 'DIFF'}",
                flush=True,
            )
    out = Path("results")
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "nominal_table2_comparison.csv", index=False)
    print("sinusoidal reference, not recomputed:")
    for target, (median, p95) in SINUSOIDAL_REFERENCE.items():
        print(f"  {target:6.0f} Hz  {median:.4f} ({p95:.4f})")
    if failed:
        raise SystemExit("one or more Table 2 or 160 ms peak values differ")


if __name__ == "__main__":
    main()
