import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import append_monthly_metrics as amm

NOW = datetime(2026, 8, 14, 13, 18, 22, tzinfo=timezone.utc)


class AppendMonthlyMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "metrics"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_monthly_path_uses_utc_year_month(self):
        path = amm.monthly_path(self.root, "ci", now=NOW)
        self.assertEqual(path, self.root / "ci" / "2026-08.json")

    def test_empty_monthly_file_creates_array(self):
        incoming = {"datetime": "2026-08-14 13:18:22Z", "run_id": 1, "metrics": {"loc": {}}}
        monthly_file, appended = amm.append_monthly_metrics(self.root, "ci", incoming, now=NOW)
        self.assertTrue(appended)
        self.assertEqual(monthly_file, self.root / "ci" / "2026-08.json")
        self.assertEqual(json.loads(monthly_file.read_text()), [incoming])
        self.assertTrue(monthly_file.read_text().endswith("\n"))
        self.assertIn("\n   ", monthly_file.read_text())

    def test_append_second_entry(self):
        first = {"run_id": 1, "metrics": {"a": 1}}
        second = {"run_id": 2, "metrics": {"a": 2}}
        amm.append_monthly_metrics(self.root, "ci", first, now=NOW)
        monthly_file, appended = amm.append_monthly_metrics(self.root, "ci", second, now=NOW)
        self.assertTrue(appended)
        self.assertEqual(json.loads(monthly_file.read_text()), [first, second])

    def test_run_id_idempotency(self):
        incoming = {"run_id": 11, "metrics": {"n": 1}}
        amm.append_monthly_metrics(self.root, "ci", incoming, now=NOW)
        before = (self.root / "ci" / "2026-08.json").read_text()
        monthly_file, appended = amm.append_monthly_metrics(
            self.root, "ci", {"run_id": 11, "metrics": {"n": 99}}, now=NOW
        )
        self.assertFalse(appended)
        self.assertEqual(monthly_file.read_text(), before)
        self.assertEqual(json.loads(monthly_file.read_text()), [incoming])

    def test_reject_non_array_monthly_file(self):
        monthly_file = self.root / "ci" / "2026-08.json"
        monthly_file.parent.mkdir(parents=True)
        monthly_file.write_text("{}\n")
        with self.assertRaises(TypeError) as ctx:
            amm.append_monthly_metrics(self.root, "ci", {"run_id": 1}, now=NOW)
        self.assertIn("not a JSON array", str(ctx.exception))

    def test_reject_invalid_metric_id(self):
        with self.assertRaises(ValueError):
            amm.append_monthly_metrics(self.root, "../oops", {"run_id": 1}, now=NOW)
        with self.assertRaises(ValueError):
            amm.append_monthly_metrics(self.root, "ci/all", {"run_id": 1}, now=NOW)

    def test_cli_appends_and_prints_status(self):
        incoming_path = Path(self.tmp.name) / "incoming.json"
        incoming_path.write_text(json.dumps({"run_id": 7, "metrics": {}}) + "\n")
        script = Path(__file__).resolve().with_name("append_monthly_metrics.py")
        proc = subprocess.run(
            [sys.executable, str(script), str(self.root), "ci", str(incoming_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        year_month = datetime.now(timezone.utc).strftime("%Y-%m")
        self.assertEqual(proc.stdout, f"appended metrics/ci/{year_month}.json\n")
        self.assertTrue((self.root / "ci" / f"{year_month}.json").is_file())


if __name__ == "__main__":
    unittest.main()
