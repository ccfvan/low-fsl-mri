"""Checks of the paper schedule and Lorentzian transfer method, not full-paper reruns."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lowfsl.schedules import (
    PAPER_FREQUENCIES_HZ,
    cpmg_period_count,
    period_dt_s,
    preparation_times_s,
    steps_per_period,
)
from lowfsl.lorentzian_transfer import (
    correlation_integral,
    fit_monoexponential_rate,
    fit_transfer,
    modulation_phase,
    traditional_rate,
)


class ScheduleTests(unittest.TestCase):
    def test_paper_frequencies_are_every_integer_1_to_100(self) -> None:
        self.assertEqual(PAPER_FREQUENCIES_HZ, tuple(range(1, 101)))
        self.assertNotIn(9.5, PAPER_FREQUENCIES_HZ)

    def test_period_steps_match_the_fitting_script(self) -> None:
        for frequency in (1, 10, 11, 20, 50, 100):
            expected = max(4, round(1 / (frequency * 12.5e-6)))
            self.assertEqual(steps_per_period(frequency), expected)
            self.assertAlmostEqual(period_dt_s(frequency) * expected * frequency, 1.0, places=12)

    def test_cpmg_counts_and_low_frequency_exception(self) -> None:
        self.assertEqual(cpmg_period_count(1), 3)
        self.assertEqual(cpmg_period_count(11), 3)
        self.assertEqual(cpmg_period_count(12), min(5, int(math.floor(0.250 * 12 + 1e-10))))
        self.assertEqual(cpmg_period_count(20), 5)
        self.assertEqual(cpmg_period_count(100), 5)
        # Fitting-script 1e-10 term agrees with floor(0.25*f) on the paper grid.
        for frequency in PAPER_FREQUENCIES_HZ:
            if frequency > 11:
                self.assertEqual(cpmg_period_count(frequency), min(5, int(math.floor(0.25 * frequency))))

    def test_durations_are_representable_and_balanced_centers_land(self) -> None:
        for frequency in (1, 10, 20, 50, 100):
            dt = period_dt_s(frequency)
            for prep in ("ordinary", "composite", "balanced", "cpmg"):
                times = preparation_times_s(prep, frequency)
                self.assertGreaterEqual(len(times), 4)
                self.assertAlmostEqual(times[0], 0.0, places=12)
                for t in times:
                    self.assertLess(abs(round(t / dt) * dt - t), 1e-9)
            for t in preparation_times_s("balanced", frequency):
                self.assertLess(abs(round(t / (4 * dt)) * 4 * dt - t), 1e-9)


class TransferMethodTests(unittest.TestCase):
    def test_phases_at_ideal_boundaries(self) -> None:
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        f, T = 1.0, 1.0
        np.testing.assert_allclose(modulation_phase("ordinary", f, T, t), 2 * np.pi * t)
        composite = modulation_phase("composite", f, T, t)
        self.assertAlmostEqual(composite[0], 0.0)
        self.assertAlmostEqual(composite[-1], np.pi)
        self.assertGreater(composite[1], composite[0])
        balanced = modulation_phase("balanced", f, T, np.array([0.0, 0.2, 0.5, 0.8, 1.0]))
        # After the first ideal pi at T/4, the lock reverses.
        self.assertAlmostEqual(balanced[0], 0.0)
        # The paper phase applies the instantaneous pi jumps at T/4 and 3T/4
        # (the comparison is t < T/4, not t <= T/4).
        mid = modulation_phase("balanced", f, T, np.array([0.25, 0.75]))
        self.assertAlmostEqual(mid[0], 2 * np.pi * f * 0.25 + np.pi)
        self.assertAlmostEqual(mid[1], 2 * np.pi * f * 0.25 - 2 * np.pi * f * 0.5)
        cpmg = modulation_phase("cpmg", 50.0, 0.04, np.array([0.0, 0.009, 0.011, 0.04]))
        self.assertEqual(cpmg[0], 0.0)
        self.assertGreater(cpmg[-1], cpmg[0])

    def test_correlation_integral_limits(self) -> None:
        self.assertEqual(correlation_integral("ordinary", 50.0, 0.0, 0.03), 0.0)
        value = correlation_integral("ordinary", 50.0, 0.04, 0.03)
        self.assertGreater(value, 0.0)
        # A coarser quadrature must stay close; this is the paper integral, not a new model.
        coarse = correlation_integral("ordinary", 50.0, 0.04, 0.03, hmax=0.001)
        self.assertLess(abs(coarse - value) / value, 0.05)

    def test_gauss_newton_recovers_a_known_decay(self) -> None:
        t = np.array([0.0, 0.04, 0.08, 0.16])
        rate = fit_monoexponential_rate(t, np.exp(-12.5 * t))
        self.assertAlmostEqual(rate, 12.5, places=6)

    def test_traditional_formula_and_cw_only_fit(self) -> None:
        truth = np.array([12.52, 1.87, 0.0299])
        frequencies = np.arange(1, 101, dtype=float)
        rows = []
        for f in frequencies:
            rows.append(dict(preparation="ordinary", f_Hz=f, R_fit_s_1=traditional_rate(truth, f), valid=True))
            # Other preparations deliberately differ so a CW-only fit must not use them.
            rows.append(dict(preparation="composite", f_Hz=f, R_fit_s_1=traditional_rate(truth, f) + 0.2, valid=True))
            rows.append(dict(preparation="balanced", f_Hz=f, R_fit_s_1=traditional_rate(truth, f) + 0.3, valid=True))
            rows.append(
                dict(
                    preparation="cpmg",
                    f_Hz=f,
                    R_fit_s_1=traditional_rate(truth, f) + 0.1,
                    valid=f > 1,
                )
            )
        result = fit_transfer(pd.DataFrame(rows))
        traditional = result["parameters"].set_index("model").loc["traditional"]
        self.assertTrue(traditional["successful"])
        self.assertAlmostEqual(traditional["Rbg_s1"], truth[0], places=3)
        self.assertAlmostEqual(traditional["A"], truth[1], places=3)
        self.assertAlmostEqual(traditional["tau_ms"], truth[2] * 1000, places=2)
        cw = result["metrics"]
        cw = cw[(cw.model == "traditional") & (cw.preparation == "ordinary") & (cw.window_Hz == "1-100")].iloc[0]
        self.assertLess(cw["RMSE_s1"], 1e-6)
        # Traditional predictions are preparation-independent, so PDEV is zero by construction.
        trad_pdev = result["pdev"][(result["pdev"].model == "traditional") & (result["pdev"].preparation == "pooled_composite_balanced_cpmg")].iloc[0]
        self.assertAlmostEqual(trad_pdev["PDEV_pct"], 0.0, places=6)
        # Invalid 1 Hz CPMG must be excluded.
        cpmg_pred = result["predictions"]
        self.assertFalse(((cpmg_pred.preparation == "cpmg") & (cpmg_pred.frequency_Hz == 1)).any())

    def test_spectrum_model_is_not_the_traditional_formula(self) -> None:
        # On a short grid, the spectrum model must still run and remain finite.
        frequencies = np.array([20.0, 50.0, 100.0])
        rows = []
        for f in frequencies:
            rows.append(dict(preparation="ordinary", f_Hz=f, R_fit_s_1=12.5 + 1.5 / (1 + (2 * np.pi * f * 0.03) ** 2), valid=True))
            rows.append(dict(preparation="composite", f_Hz=f, R_fit_s_1=12.7, valid=True))
            rows.append(dict(preparation="balanced", f_Hz=f, R_fit_s_1=12.8, valid=True))
            rows.append(dict(preparation="cpmg", f_Hz=f, R_fit_s_1=12.6, valid=True))
        result = fit_transfer(pd.DataFrame(rows))
        spectrum = result["parameters"].set_index("model").loc["spectrum"]
        self.assertTrue(np.isfinite(spectrum["Rbg_s1"]))
        self.assertTrue(np.isfinite(spectrum["A"]))
        self.assertTrue(np.isfinite(spectrum["tau_ms"]))
        self.assertGreater(spectrum["A"], 0.0)
        # Spectrum predictions may depend on preparation; traditional ones do not.
        pred = result["predictions"]
        trad = pred[(pred.model == "traditional") & (pred.frequency_Hz == 50)]
        self.assertEqual(len(set(np.round(trad.prediction_s1, 10))), 1)
        spec = pred[(pred.model == "spectrum") & (pred.frequency_Hz == 50)]
        self.assertGreaterEqual(len(spec), 4)


if __name__ == "__main__":
    unittest.main()
