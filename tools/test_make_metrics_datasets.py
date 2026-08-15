import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_metrics_datasets as mmd

SAMPLE_REPORT = """\
Analyzing build trace from 'capture_build.bin'...
**** Time summary:
Compilation (434 times):
  Parsing (frontend):         9395.6 s
  Codegen & opts (backend):   7453.7 s

**** Files that took longest to parse (compiler frontend):
127941 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/pySPHModel.cpp.o
 73355 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Solver.cpp.o
 63510 ms: ./src/shammodels/gsph/CMakeFiles/shammodels_gsph.dir/src/pyGSPHModel.cpp.o
 62234 ms: ./src/shammodels/gsph/CMakeFiles/shammodels_gsph.dir/src/Solver.cpp.o
 62116 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/ComputeEos.cpp.o
 56076 ms: ./src/shamrock/CMakeFiles/shamlib.dir/src/patch/PatchDataLayer.cpp.o
 54276 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Model.cpp.o
 51221 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/modules/render/CartesianRender.cpp.o
 50558 ms: ./src/shammodels/ramses/CMakeFiles/shammodels_ramses.dir/src/Solver.cpp.o
 49117 ms: ./src/shammodels/ramses/CMakeFiles/shammodels_ramses.dir/src/pyRamsesModel.cpp.o
 48000 ms: ./src/extra/CMakeFiles/extra.dir/src/ShouldBeDropped.cpp.o

**** Files that took longest to codegen (compiler backend):
202994 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/pySPHModel.cpp.o
195319 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Solver.cpp.o
168192 ms: ./src/shamrock/CMakeFiles/shamlib.dir/src/patch/PatchDataField.cpp.o

**** Files with highest peak RSS (compiler process):
  1234.5 MB: ./src/shammodels/sph/src/pySPHModel.cpp
"""


def write_snapshot(root, name, payload):
    aggregated = Path(root) / "aggregated"
    aggregated.mkdir(parents=True, exist_ok=True)
    path = aggregated / name
    path.write_text(json.dumps(payload, indent=3) + "\n")
    return path


class MakeMetricsDatasetsTests(unittest.TestCase):
    def test_iso8601_and_object_label(self):
        self.assertEqual(mmd.to_iso8601("2026-08-15 12:17:37Z"), "2026-08-15T12:17:37Z")
        self.assertEqual(mmd.to_iso8601("2026-08-15T12:17:37Z"), "2026-08-15T12:17:37Z")
        self.assertEqual(
            mmd.object_file_label(
                "./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/pySPHModel.cpp.o"
            ),
            "src/shammodels/sph/src/pySPHModel.cpp",
        )
        self.assertEqual(
            mmd.object_file_label("/__w/Shamrock/Shamrock/src/shammodels/sph/src/pySPHModel.cpp"),
            "src/shammodels/sph/src/pySPHModel.cpp",
        )

    def test_parse_time_summary_and_top_files(self):
        summary = mmd.parse_time_summary(SAMPLE_REPORT)
        self.assertEqual(summary["parsing_s"], 9395.6)
        self.assertEqual(summary["codegen_s"], 7453.7)
        self.assertEqual(summary["total_s"], 9395.6 + 7453.7)

        parse_rows = mmd.parse_file_times(SAMPLE_REPORT, mmd.PARSE_FILES_HEADER)
        self.assertEqual(len(parse_rows), 10)
        self.assertEqual(parse_rows[0]["rank"], 1)
        self.assertEqual(parse_rows[0]["time_ms"], 127941)
        self.assertEqual(parse_rows[0]["label"], "src/shammodels/sph/src/pySPHModel.cpp")
        self.assertTrue(all("ShouldBeDropped" not in row["label"] for row in parse_rows))

        codegen_rows = mmd.parse_file_times(SAMPLE_REPORT, mmd.CODEGEN_FILES_HEADER)
        self.assertEqual(len(codegen_rows), 3)
        self.assertEqual(codegen_rows[2]["label"], "src/shamrock/src/patch/PatchDataField.cpp")

        rss_rows = mmd.parse_file_rss(SAMPLE_REPORT, mmd.RSS_FILES_HEADER)
        self.assertEqual(len(rss_rows), 1)
        self.assertEqual(rss_rows[0]["rss_mb"], 1234.5)
        self.assertEqual(rss_rows[0]["label"], "src/shammodels/sph/src/pySPHModel.cpp")

    def test_parse_rss_usage_prefers_structured_memlog(self):
        usage = [
            {"rss_mb": 100.0, "path": "/__w/Shamrock/Shamrock/src/a.cpp"},
            {
                "rss_mb": 2693.9,
                "path": "/__w/Shamrock/Shamrock/src/shammodels/sph/src/pySPHModel.cpp",
            },
            {"rss_mb": 50.0, "path": "/__w/Shamrock/Shamrock/src/b.cpp"},
        ]
        rows = mmd.parse_rss_usage(usage)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["rss_mb"], 2693.9)
        self.assertEqual(rows[0]["label"], "src/shammodels/sph/src/pySPHModel.cpp")
        self.assertEqual(mmd.peak_rss_mb({"compile_memory_usage": usage}), 2693.9)
        self.assertIsNone(mmd.peak_rss_mb({}))

    def test_build_datasets_skips_incomplete_and_prunes_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_snapshot(
                root,
                "2026-08-15_12-17-37Z.json",
                {
                    "datetime": "2026-08-15 12:17:37Z",
                    "sha": "abc123",
                    "metrics": {
                        "doxygen_warn": {"doxygen_warning_count": 8171},
                        "build_profile": {"data": SAMPLE_REPORT},
                    },
                },
            )
            write_snapshot(
                root,
                "2026-08-16_00-00-00Z.json",
                {
                    "datetime": "2026-08-16 00:00:00Z",
                    "sha": "def456",
                    "metrics": {"doxygen_warn": {"doxygen_warning_count": 8000}},
                },
            )
            write_snapshot(
                root,
                "2026-08-17_00-00-00Z.json",
                {
                    "datetime": "2026-08-17 00:00:00Z",
                    "sha": "ghi789",
                    "metrics": {
                        "doxygen_warn": {"doxygen_warning_count": 7900},
                        "build_profile": {
                            "data": SAMPLE_REPORT,
                            "compile_memory_usage": [
                                {
                                    "rss_mb": 97.0,
                                    "path": "/__w/Shamrock/Shamrock/src/shamterm/src/color.cpp",
                                },
                                {
                                    "rss_mb": 2693.9,
                                    "path": "/__w/Shamrock/Shamrock/src/shammodels/sph/src/pySPHModel.cpp",
                                },
                            ],
                        },
                    },
                },
            )

            output_dir = root / "output"
            output_dir.mkdir()
            stale = output_dir / "stale.json"
            stale.write_text("{}\n")

            produced = mmd.build_datasets(root, output_dir)
            names = sorted(path.name for path in produced)
            self.assertEqual(names, sorted(mmd.DATASET_FILES))
            self.assertFalse(stale.exists())

            doxygen = json.loads((output_dir / "doxygen_warnings.json").read_text())
            self.assertEqual(doxygen["name"], "doxygen_warnings")
            self.assertEqual(
                [row["doxygen_warning_count"] for row in doxygen["data"]],
                [8171, 8000, 7900],
            )
            self.assertEqual(doxygen["data"][0]["datetime"], "2026-08-15T12:17:37Z")

            compile_times = json.loads((output_dir / "compile_times.json").read_text())
            self.assertEqual(len(compile_times["data"]), 2)
            self.assertEqual(compile_times["data"][0]["sha"], "abc123")
            self.assertNotIn("peak_rss_mb", compile_times["data"][0])
            self.assertAlmostEqual(compile_times["data"][0]["total_s"], 9395.6 + 7453.7)
            self.assertEqual(compile_times["data"][1]["sha"], "ghi789")
            self.assertEqual(compile_times["data"][1]["peak_rss_mb"], 2693.9)

            top_parse = json.loads((output_dir / "top_parse_files.json").read_text())
            self.assertEqual(len(top_parse["data"]), 20)
            self.assertEqual(top_parse["data"][0]["rank"], 1)

            top_codegen = json.loads((output_dir / "top_codegen_files.json").read_text())
            self.assertEqual(len(top_codegen["data"]), 6)

            top_rss = json.loads((output_dir / "top_rss_files.json").read_text())
            # First snapshot falls back to the text RSS section (1 row).
            # Third snapshot uses compile_memory_usage (2 rows, ranked by rss).
            self.assertEqual(len(top_rss["data"]), 3)
            self.assertEqual(top_rss["data"][0]["sha"], "abc123")
            self.assertEqual(top_rss["data"][0]["rss_mb"], 1234.5)
            self.assertEqual(top_rss["data"][1]["sha"], "ghi789")
            self.assertEqual(top_rss["data"][1]["rss_mb"], 2693.9)
            self.assertEqual(
                top_rss["data"][1]["label"],
                "src/shammodels/sph/src/pySPHModel.cpp",
            )


if __name__ == "__main__":
    unittest.main()
