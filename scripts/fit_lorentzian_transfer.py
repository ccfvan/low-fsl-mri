#!/usr/bin/env python3
"""Fit and cross-predict the paper Lorentzian models. No plots.

Input: a rate table with columns preparation, f_Hz, R_fit_s_1, valid.
The table can be fits.csv from run_microvascular_paper.py. Historical
rate tables are not included. Both models are fitted to CW/ordinary
rows only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lowfsl.lorentzian_transfer import fit_transfer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dt-us", type=float, default=12.5)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    result = fit_transfer(frame, dt_ref_us=args.dt_us)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(out / f"{name}.csv", index=False)
        print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
