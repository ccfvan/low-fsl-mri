"""Small-grid pipeline and convergence-driver checks. Not paper-resolution results."""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PipelineTests(unittest.TestCase):
    def test_small_grid_rates_then_transfer_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rates = tmp / "rates"
            fitdir = tmp / "fit"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_microvascular_paper.py"),
                    "--grid",
                    "8",
                    "--dt-us",
                    "50",
                    "--frequencies",
                    "50,100",
                    "--output",
                    str(rates),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0)
            fits = pd.read_csv(rates / "fits.csv")
            self.assertEqual(set(fits.preparation), {"ordinary", "composite", "balanced", "cpmg"})
            self.assertEqual(set(fits.f_Hz), {50.0, 100.0})
            self.assertTrue(fits.valid.all())
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "fit_lorentzian_transfer.py"),
                    "--input",
                    str(rates / "fits.csv"),
                    "--output",
                    str(fitdir),
                    "--dt-us",
                    "50",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            parameters = pd.read_csv(fitdir / "parameters.csv")
            self.assertEqual(set(parameters.model), {"traditional", "spectrum"})
            self.assertTrue(parameters.successful.all())

    def test_convergence_driver_on_a_tiny_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "check_microvascular_convergence.py"),
                    "--grid-baseline",
                    "8",
                    "--grid-refined",
                    "10",
                    "--dt-us",
                    "50",
                    "--dt-halved-us",
                    "25",
                    "--frequencies",
                    "50",
                    "--output",
                    str(tmp / "conv"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0)
            comparisons = pd.read_csv(tmp / "conv" / "comparisons.csv")
            self.assertEqual(set(comparisons.comparison), {"halved_timestep", "refined_grid"})
            self.assertTrue(comparisons.valid.all())
            # On this tiny grid the numbers are not paper results, but finite relative errors must exist.
            self.assertTrue(comparisons.rel_diff.notna().all())
            self.assertTrue((comparisons.rel_diff >= 0).all())


if __name__ == "__main__":
    unittest.main()
