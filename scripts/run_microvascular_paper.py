#!/usr/bin/env python3
"""Ideal-preparation microvascular rates on the paper frequency schedule.

Geometry and timing defaults follow the paper: radius 6 um, volume fraction
0.035, 0.18 ppm, 3 T, D = 0.8 um^2/ms, T1 = 1.3 s, T2 = 80 ms, 64^3 reference
grid with axial reduction, 12.5 us reference step, and every integer frequency
from 1 to 100 Hz. Preparations are CW, composite, balanced, and ideal-pulse
CPMG. This is not a stored historical table and does not include Lorentzian
fitting; use fit_lorentzian_transfer.py on the resulting rates.

A full 1-100 Hz, 64-grid run is computationally substantial. Use a smaller
--grid or a frequency subset only for timing checks, not as a substitute for
the paper sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lowfsl.microvascular import Params, make_fft_cylinder_field, Evolution
from lowfsl.schedules import (
    PAPER_BASELINE_DT_US,
    PAPER_BASELINE_GRID,
    PAPER_FREQUENCIES_HZ,
    PAPER_PREPARATIONS,
    PAPER_B0_T,
    PAPER_D_UM2_MS,
    PAPER_DELTA_CHI,
    PAPER_RADIUS_UM,
    PAPER_T1_S,
    PAPER_T2_S,
    PAPER_VOLUME_FRACTION,
    period_dt_s,
    preparation_times_s,
)


def fit_decay(time_s, signal):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_s) < 4 or (not np.all(np.isfinite(signal))) or np.min(signal) < 1e-12:
        return dict(R_fit_s_1=float("nan"), S0=float("nan"), R2=float("nan"), valid=False)
    result = least_squares(
        lambda p: p[0] * np.exp(-p[1] * time_s) - signal,
        [signal[0], 12.5],
        bounds=([0, 0], [np.inf, np.inf]),
    )
    predicted = result.x[0] * np.exp(-result.x[1] * time_s)
    denom = np.sum((signal - signal.mean()) ** 2)
    r2 = float("nan") if denom == 0 else 1.0 - np.sum((predicted - signal) ** 2) / denom
    return dict(
        R_fit_s_1=float(result.x[1]),
        S0=float(result.x[0]),
        R2=float(r2),
        valid=bool(result.success and r2 >= 0.95 and np.min(signal) >= 1e-12),
    )


def parse_frequencies(text: str) -> list[float]:
    if text.strip() in {"paper", "1:100"}:
        return list(PAPER_FREQUENCIES_HZ)
    if ":" in text:
        start, stop = map(int, text.split(":"))
        return list(range(start, stop + 1))
    return [float(part) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=PAPER_BASELINE_GRID)
    parser.add_argument("--dt-us", type=float, default=PAPER_BASELINE_DT_US)
    parser.add_argument("--frequencies", default="paper")
    parser.add_argument("--output", default="microvascular_paper_output")
    args = parser.parse_args()
    frequencies = parse_frequencies(args.frequencies)
    if args.grid < 4 or args.dt_us <= 0:
        raise ValueError("grid must be >= 4 and dt must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    decays = []
    fits = []
    field_info = None
    for frequency in frequencies:
        p = Params(
            nx=args.grid,
            ny=args.grid,
            nz=args.grid,
            dt_s=period_dt_s(frequency, args.dt_us),
            radius_um=PAPER_RADIUS_UM,
            volume_fraction=PAPER_VOLUME_FRACTION,
            delta_chi=PAPER_DELTA_CHI,
            b0_t=PAPER_B0_T,
            d_um2_ms=PAPER_D_UM2_MS,
            t1_s=PAPER_T1_S,
            t2_s=PAPER_T2_S,
        )
        field = make_fft_cylinder_field(p)
        field_info = field
        evolution = Evolution(p, field, axial=True)
        for prep in PAPER_PREPARATIONS:
            times = preparation_times_s(prep, frequency, args.dt_us)
            started = time.monotonic()
            if prep in ("ordinary", "cpmg"):
                values = [y for _, y in evolution.samples(frequency, prep, times)]
            else:
                values = [evolution.independent(frequency, prep, float(t)) for t in times]
            for t, y in zip(times, values):
                decays.append(
                    dict(
                        condition="radius6",
                        f_Hz=frequency,
                        preparation=prep,
                        dt_us=p.dt_s * 1e6,
                        TSL_s=float(t),
                        signed_mean_Mx=float(y),
                    )
                )
            row = dict(
                condition="radius6",
                f_Hz=frequency,
                preparation=prep,
                grid=args.grid,
                dt_us=p.dt_s * 1e6,
                nTSL=len(times),
                seconds=round(time.monotonic() - started, 3),
                **fit_decay(times, values),
            )
            print(row, flush=True)
            fits.append(row)
    for name, rows in (("decays.csv", decays), ("fits.csv", fits)):
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out / "metadata.json").write_text(
        json.dumps(
            dict(
                geometry="periodic straight cylinder, axial reduction of a cubic cell",
                radius_um=PAPER_RADIUS_UM,
                volume_fraction=PAPER_VOLUME_FRACTION,
                actual_volume_fraction=None if field_info is None else field_info["actual_volume_fraction"],
                delta_chi_ppm=0.18,
                B0_T=PAPER_B0_T,
                D_um2_ms=PAPER_D_UM2_MS,
                T1_s=PAPER_T1_S,
                T2_s=PAPER_T2_S,
                reference_grid=args.grid,
                dt_ref_us=args.dt_us,
                frequencies_Hz=frequencies,
                preparations=list(PAPER_PREPARATIONS),
                finite_pulse_width=False,
                note="Ideal instantaneous refocusing pulses. Not a previously stored rate table.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
