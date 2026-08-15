import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_metrics_datasets as mmd


def write_snapshot(root, name, payload):
    aggregated = Path(root) / "aggregated"
    aggregated.mkdir(parents=True, exist_ok=True)
    path = aggregated / name
    path.write_text(json.dumps(payload, indent=3) + "\n")
    return path


class MakeMetricsDatasetsTests(unittest.TestCase):
    def test_build_doxygen_warnings_skips_incomplete_and_prunes_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_snapshot(
                root,
                "2026-08-15_12-17-37Z.json",
                {
                    "datetime": "2026-08-15 12:17:37Z",
                    "sha": "abc123",
                    "metrics": {"doxygen_warn": {"doxygen_warning_count": 8171}},
                },
            )
            write_snapshot(
                root,
                "2026-08-16_00-00-00Z.json",
                {
                    "datetime": "2026-08-16 00:00:00Z",
                    "sha": "def456",
                    "metrics": {"loc": {"total": {"code": 1}}},
                },
            )
            write_snapshot(
                root,
                "2026-08-17_00-00-00Z.json",
                {
                    "datetime": "2026-08-17 00:00:00Z",
                    "sha": "ghi789",
                    "metrics": {"doxygen_warn": {"doxygen_warning_count": 8000}},
                },
            )

            output_dir = root / "output"
            output_dir.mkdir()
            stale = output_dir / "compile_times.json"
            stale.write_text("{}\n")

            produced = mmd.build_datasets(root, output_dir)
            self.assertEqual([path.name for path in produced], ["doxygen_warnings.json"])
            self.assertFalse(stale.exists())
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()), ["doxygen_warnings.json"]
            )

            doxygen = json.loads((output_dir / "doxygen_warnings.json").read_text())
            self.assertEqual(doxygen["name"], "doxygen_warnings")
            self.assertEqual(
                doxygen["data"],
                [
                    {
                        "datetime": "2026-08-15 12:17:37Z",
                        "sha": "abc123",
                        "doxygen_warning_count": 8171,
                    },
                    {
                        "datetime": "2026-08-17 00:00:00Z",
                        "sha": "ghi789",
                        "doxygen_warning_count": 8000,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
