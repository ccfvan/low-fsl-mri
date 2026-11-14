"""Checks for the manuscript six-preparation set and the shared Bloch kernels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lowfsl.engine as engine


class PaperMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.p = engine.Params()

    def test_paper_methods_are_the_six_preparations(self) -> None:
        self.assertEqual(
            engine.PAPER_METHOD_KEYS,
            (
                "ordinary",
                "rotary_echo",
                "balanced",
                "quadruple_refocused",
                "sinusoidal",
                "phase_cycled_cpmg",
            ),
        )
        self.assertNotIn("triple_refocused", engine.PAPER_METHOD_KEYS)
        self.assertIn("triple_refocused", engine.EXCLUDED_FROM_PAPER)
        labels = [method.short_label for method in engine.PAPER_METHODS]
        self.assertEqual(
            labels,
            ["CW", "Composite", "Balanced", "Quadruple", "Sinusoidal", "CPMG"],
        )

    def test_triple_refocused_builder_is_not_a_paper_entry(self) -> None:
        operations = engine.build_core(self.p, "triple_refocused", 0.08, 50.0)
        self.assertEqual(
            sum(op.label.startswith("pi") for op in operations),
            3,
        )

    def test_all_core_durations_equal_requested_tsl(self) -> None:
        for method in engine.PAPER_METHODS:
            for target_hz in engine.TARGETS_HZ:
                for tsl_ms in engine.TSL_MS_BY_TARGET[target_hz][1:]:
                    operations = engine.build_core(
                        self.p,
                        method.key,
                        float(tsl_ms) * 1e-3,
                        target_hz,
                    )
                    duration_s = sum(operation.duration_s for operation in operations)
                    self.assertAlmostEqual(duration_s, float(tsl_ms) * 1e-3, places=9)

    def test_default_field_grid_matches_figure_caption_sampling(self) -> None:
        self.assertEqual(len(self.p.b0_hz), 101)
        self.assertEqual(len(self.p.b1_pct), 101)
        self.assertAlmostEqual(self.p.b0_hz[0], -250.0)
        self.assertAlmostEqual(self.p.b0_hz[-1], 250.0)
        self.assertAlmostEqual(float(np.diff(self.p.b0_hz).min()), 5.0)
        self.assertAlmostEqual(self.p.b1_pct[0], -25.0)
        self.assertAlmostEqual(self.p.b1_pct[-1], 25.0)

    def test_summary_windows_are_reported_and_not_selected(self) -> None:
        rows = engine.relaxation_error_windows(
            np.array([-250.0, -100.0, 0.0, 100.0, 250.0]),
            np.array([10.0, 4.0, 0.0, 4.0, 10.0]),
        )
        self.assertEqual([row["abs_b0_max_hz"] for row in rows], [100.0, 250.0])
        self.assertEqual([row["n_points"] for row in rows], [3, 5])
        self.assertTrue(all(row["manuscript_statistic_selected"] is False for row in rows))
        self.assertEqual(
            engine.field_scope_label(100.0, -25.0, 25.0),
            "absB0_le_100Hz_B1_pm25pct",
        )
        self.assertEqual(
            engine.field_scope_label(250.0, -25.0, 25.0),
            "absB0_le_250Hz_B1_pm25pct",
        )

    def test_refocused_phase_tables_and_counts(self) -> None:
        expected = {
            "balanced": (2, (1, -1)),
            "triple_refocused": (3, (1, 1, -1)),
            "quadruple_refocused": (4, (1, 1, -1, -1)),
        }
        for method_key, (pulse_count, signs) in expected.items():
            _, actual_signs = engine.refocused_definition(method_key, 0.08)
            self.assertEqual(actual_signs, signs)
            operations = engine.build_core(self.p, method_key, 0.08, 50.0)
            self.assertEqual(
                sum(":180" not in op.label and op.label.startswith("pi") for op in operations),
                pulse_count,
            )

    def test_cpmg_uses_finite_composite_mlev16_pulses(self) -> None:
        operations = engine.build_cpmg_core(self.p, 0.08, 100.0)
        central = [operation for operation in operations if ":180" in operation.label]
        flanking = [operation for operation in operations if ":90" in operation.label]
        self.assertEqual(len(central), 16)
        self.assertEqual(len(flanking), 32)
        self.assertTrue(all(np.isclose(op.duration_s, self.p.hard_180_width_s) for op in central))
        self.assertTrue(all(np.isclose(op.duration_s, self.p.hard_90_width_s) for op in flanking))
        actual_signs = tuple(int(np.sign(op.fx_hz)) for op in central)
        self.assertEqual(actual_signs, engine.MLEV16_SIGNS)

    def test_rotary_echo_is_positive_lock_center_pi_negative_lock(self) -> None:
        operations = engine.build_rotary_echo_core(self.p, 0.08, 50.0)
        self.assertEqual([operation.label for operation in operations], [
            "rotary +SL",
            "rotary center pi:180X",
            "rotary -SL",
        ])
        self.assertAlmostEqual(operations[0].fx_hz, 50.0)
        self.assertAlmostEqual(operations[2].fx_hz, -50.0)
        center = operations[1]
        self.assertAlmostEqual(360 * center.fx_hz * center.duration_s, 180.0, places=12)

    def test_nominal_signal_matches_t2_decay_for_spinlock_methods(self) -> None:
        tsl_s = 0.02
        expected = float(np.exp(-tsl_s / self.p.t2_s))
        spinlock_keys = {
            "ordinary",
            "rotary_echo",
            "balanced",
            "quadruple_refocused",
            "sinusoidal",
        }
        for method in engine.PAPER_METHODS:
            if method.key not in spinlock_keys:
                continue
            operations = engine.build_core(self.p, method.key, tsl_s, 100.0)
            signal = engine.simulate_preparation(
                self.p,
                operations,
                np.array([0.0]),
                np.array([1.0]),
                {},
                initial_axis=0,
                readout_axis=0,
            )[0]
            self.assertAlmostEqual(signal, expected, places=6, msg=method.key)

    def test_sinusoidal_samples_use_manuscript_beta(self) -> None:
        operations = engine.build_sinusoidal_core(self.p, 1.0 / 50.0, 50.0)
        self.assertAlmostEqual(sum(op.duration_s for op in operations), 0.02, places=12)
        self.assertAlmostEqual(self.p.sinusoidal_beta, 2.4048256)
        peak = max(abs(op.fx_hz) for op in operations)
        self.assertAlmostEqual(peak, self.p.sinusoidal_beta * 50.0, delta=0.05)

    def test_pqt_matches_theory_definition(self) -> None:
        duration_s = 0.08
        operations = engine.build_core(self.p, "ordinary", duration_s, 50.0)
        spectrum, metrics, _ = engine.finite_time_spectrum(self.p, operations, duration_s)
        np.testing.assert_allclose(
            spectrum.pQT_per_rad_s,
            spectrum.qPower_s2 / (2 * np.pi * duration_s),
            rtol=1e-12,
            atol=1e-15,
        )
        self.assertGreater(metrics["dominantPQT_per_Hz"], 0.0)
        self.assertGreater(metrics["qCapturedPowerFractionWithin500Hz"], 0.99)
        self.assertLessEqual(metrics["qCapturedPowerFractionWithin500Hz"], 1.001)

    def test_foltz_fixed_delta_correction(self) -> None:
        tsl_s = engine.TSL_MS_BY_TARGET[100.0] * 1e-3
        signal = np.empty((1, tsl_s.size), dtype=float)
        for index, value_s in enumerate(tsl_s):
            signal[:, index] = engine.simulate_preparation(
                self.p,
                engine.build_cpmg_core(self.p, float(value_s), 100.0),
                np.array([0.0]),
                np.array([1.0]),
                {},
                initial_axis=0,
                readout_axis=0,
            )
        fit, corrected = engine.fit_foltz_fixed_delta_batch(
            self.p,
            tsl_s,
            engine.CPMG_PULSE_COUNTS,
            signal,
        )
        expected = engine.foltz_fixed_delta_times(self.p, tsl_s, engine.CPMG_PULSE_COUNTS)
        np.testing.assert_allclose(corrected[0], expected, rtol=0, atol=1e-15)
        self.assertLess(abs(fit["Tfit_s"][0] / self.p.t2_s - 1), 0.025)
        self.assertTrue(np.all(corrected <= tsl_s[None, :]))

    def test_rf_cost_distinguishes_continuous_and_cpmg(self) -> None:
        tsl_s = 0.08
        ordinary = engine.rf_metrics(
            self.p,
            "ordinary",
            engine.build_core(self.p, "ordinary", tsl_s, 50.0),
            tsl_s,
            50.0,
        )
        cpmg = engine.rf_metrics(
            self.p,
            "phase_cycled_cpmg",
            engine.build_core(self.p, "phase_cycled_cpmg", tsl_s, 100.0),
            tsl_s,
            100.0,
        )
        sinusoidal = engine.rf_metrics(
            self.p,
            "sinusoidal",
            engine.build_core(self.p, "sinusoidal", tsl_s, 50.0),
            tsl_s,
            50.0,
        )
        self.assertEqual(ordinary["nRefocusingPulses"], 0)
        self.assertEqual(sinusoidal["nRefocusingPulses"], 0)
        self.assertEqual(cpmg["nRefocusingPulses"], 16)
        self.assertGreater(cpmg["rfRMS_Hz"], ordinary["rfRMS_Hz"])
        self.assertAlmostEqual(ordinary["rfRMS_Hz"], 50.0, places=6)


if __name__ == "__main__":
    unittest.main()
