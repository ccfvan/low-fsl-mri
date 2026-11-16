#!/usr/bin/env python3
"""Spatial and temporal convergence at the paper radius-6 geometry.

The paper reports this check at 10, 20, and 50 Hz: baseline 64^3 with a
12.5 us reference step, a halved 6.25 us step on the same grid, and a
refined 96^3 grid at 12.5 us. Axial reduction is used. No previously
computed rates are reused.

A full paper-resolution run can take several minutes. --grid-baseline and
--frequencies may be reduced only for driver tests; such reductions are
not the paper check.
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
from lowfsl.microvascular import Params, make_fft_cylinder_field, Evolution, diffusion_alphas
from lowfsl.schedules import (
    PAPER_BASELINE_DT_US,
    PAPER_BASELINE_GRID,
    PAPER_CONVERGENCE_FREQUENCIES_HZ,
    PAPER_HALVED_DT_US,
    PAPER_PREPARATIONS,
    PAPER_REFINED_GRID,
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


def fit_rate(time_s, signal):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_s) < 4 or (not np.all(np.isfinite(signal))) or np.min(signal) < 1e-12:
        return float("nan"), False
    result = least_squares(
        lambda p: p[0] * np.exp(-p[1] * time_s) - signal,
        [signal[0], 12.5],
        bounds=([0, 0], [np.inf, np.inf]),
    )
    predicted = result.x[0] * np.exp(-result.x[1] * time_s)
    denom = np.sum((signal - signal.mean()) ** 2)
    r2 = float("nan") if denom == 0 else 1.0 - np.sum((predicted - signal) ** 2) / denom
    valid = bool(result.success and r2 >= 0.95 and np.min(signal) >= 1e-12)
    return float(result.x[1]), valid


def one_setting(grid: int, dt_us: float, frequencies: list[float]) -> list[dict]:
    rows = []
    for frequency in frequencies:
        p = Params(
            nx=grid,
            ny=grid,
            nz=grid,
            dt_s=period_dt_s(frequency, dt_us),
            radius_um=PAPER_RADIUS_UM,
            volume_fraction=PAPER_VOLUME_FRACTION,
            delta_chi=PAPER_DELTA_CHI,
            b0_t=PAPER_B0_T,
            d_um2_ms=PAPER_D_UM2_MS,
            t1_s=PAPER_T1_S,
            t2_s=PAPER_T2_S,
        )
        field = make_fft_cylinder_field(p)
        alpha_sum = sum(diffusion_alphas(p, field))
        if alpha_sum > 0.49:
            for prep in PAPER_PREPARATIONS:
                rows.append(
                    dict(
                        setting=f"grid{grid}_dt{dt_us:g}",
                        grid=grid,
                        dt_us=dt_us,
                        f_Hz=frequency,
                        preparation=prep,
                        R_fit_s_1=float("nan"),
                        valid=False,
                        error="CFL limit exceeded",
                        alpha_sum=alpha_sum,
                        seconds=0.0,
                    )
                )
            continue
        evolution = Evolution(p, field, axial=True)
        for prep in PAPER_PREPARATIONS:
            times = preparation_times_s(prep, frequency, dt_us)
            started = time.monotonic()
            if prep in ("ordinary", "cpmg"):
                values = [y for _, y in evolution.samples(frequency, prep, times)]
            else:
                values = [evolution.independent(frequency, prep, float(t)) for t in times]
            rate, valid = fit_rate(times, values)
            rows.append(
                dict(
                    setting=f"grid{grid}_dt{dt_us:g}",
                    grid=grid,
                    dt_us=dt_us,
                    f_Hz=frequency,
                    preparation=prep,
                    R_fit_s_1=rate,
                    valid=valid,
                    error="",
                    alpha_sum=alpha_sum,
                    seconds=round(time.monotonic() - started, 3),
                )
            )
            print(rows[-1], flush=True)
    return rows


def compare(baseline: list[dict], other: list[dict], label: str) -> list[dict]:
    index = {(r["f_Hz"], r["preparation"]): r for r in baseline}
    rows = []
    for item in other:
        ref = index[(item["f_Hz"], item["preparation"])]
        if not (item["valid"] and ref["valid"]):
            rel = float("nan")
            absdiff = float("nan")
        else:
            absdiff = item["R_fit_s_1"] - ref["R_fit_s_1"]
            rel = abs(absdiff) / max(abs(ref["R_fit_s_1"]), 1e-3)
        rows.append(
            dict(
                comparison=label,
                f_Hz=item["f_Hz"],
                preparation=item["preparation"],
                baseline_R_s1=ref["R_fit_s_1"],
                compare_R_s1=item["R_fit_s_1"],
                abs_diff_s1=absdiff,
                rel_diff=rel,
                valid=bool(item["valid"] and ref["valid"]),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-baseline", type=int, default=PAPER_BASELINE_GRID)
    parser.add_argument("--grid-refined", type=int, default=PAPER_REFINED_GRID)
    parser.add_argument("--dt-us", type=float, default=PAPER_BASELINE_DT_US)
    parser.add_argument("--dt-halved-us", type=float, default=PAPER_HALVED_DT_US)
    parser.add_argument("--frequencies", default="10,20,50")
    parser.add_argument("--skip-refined", action="store_true")
    parser.add_argument("--output", default="microvascular_convergence")
    args = parser.parse_args()
    frequencies = [float(part) for part in args.frequencies.split(",") if part.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    baseline = one_setting(args.grid_baseline, args.dt_us, frequencies)
    halved = one_setting(args.grid_baseline, args.dt_halved_us, frequencies)
    comparisons = compare(baseline, halved, "halved_timestep")
    all_rates = baseline + halved
    if not args.skip_refined:
        refined = one_setting(args.grid_refined, args.dt_us, frequencies)
        comparisons.extend(compare(baseline, refined, "refined_grid"))
        all_rates.extend(refined)
    for name, rows in (("rates.csv", all_rates), ("comparisons.csv", comparisons)):
        with (out / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out / "metadata.json").write_text(
        json.dumps(
            dict(
                geometry="paper radius-6 um straight cylinder",
                frequencies_Hz=frequencies,
                baseline={"grid": args.grid_baseline, "dt_us": args.dt_us},
                halved_timestep_us=args.dt_halved_us,
                refined_grid=None if args.skip_refined else args.grid_refined,
                paper_reported="At 10, 20, and 50 Hz, halving the 12.5 us step changed R_fit by at most 4e-6 relative; refining 64^3 to 96^3 changed it by at most 0.11% (0.015 s^-1, at 10 Hz).",
                note="This command recomputes the check. It does not load stored historical rates.",
            ),
            indent=2,
        )
    )
    print("comparisons:", flush=True)
    for row in comparisons:
        print(row, flush=True)


if __name__ == "__main__":
    main()
