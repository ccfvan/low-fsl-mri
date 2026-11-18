"""Help/invalid options must exit before simulation or output creation."""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

class CLISafetyTests(unittest.TestCase):
    def test_every_script_help_is_read_only(self):
        for script in sorted((ROOT / "scripts").glob("*.py")):
            with self.subTest(script=script.name), tempfile.TemporaryDirectory() as tmp:
                result = subprocess.run(
                    [sys.executable, str(script), "--help"], cwd=tmp,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_nominal_help_and_invalid_option_do_not_start_simulation(self):
        spec = importlib.util.spec_from_file_location(
            "nominal_cli_safety", ROOT / "scripts/compare_nominal_table2.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for arguments, status in [(["--help"], 0), (["--not-a-valid-option"], 2)]:
            with self.subTest(arguments=arguments), patch.object(
                module.engine, "Params", side_effect=AssertionError("simulation started")
            ) as constructor:
                with self.assertRaises(SystemExit) as caught:
                    module.main(arguments)
                self.assertEqual(caught.exception.code, status)
                constructor.assert_not_called()

if __name__ == "__main__":
    unittest.main()
