# Tests for tools/clang_build_analyzer_to_json.py

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from clang_build_analyzer_to_json import parse_report

SAMPLE = """\
Analyzing build trace from 'capture_build.bin'...
**** Time summary:
Compilation (418 times):
  Parsing (frontend):         1335.9 s
  Codegen & opts (backend):   1322.7 s

**** Files that took longest to parse (compiler frontend):
 21505 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/pySPHModel.cpp.o
 11682 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Solver.cpp.o

**** Files that took longest to codegen (compiler backend):
 32901 ms: ./src/shamrock/CMakeFiles/shamlib.dir/src/patch/PatchDataField.cpp.o
 32321 ms: ./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Solver.cpp.o

**** Templates that took longest to instantiate:
 35029 ms: shamrock::patch::FieldVariant<...>::visit (2057 times, avg 17 ms)
 20080 ms: nlohmann::basic_json<>::parse<const char *> (136 times, avg 147 ms)

**** Template sets that took longest to instantiate:
 94402 ms: std::vector<$>::emplace_back<$> (14030 times, avg 6 ms)
 90068 ms: std::visit<$> (4056 times, avg 22 ms)

**** Functions that took longest to compile:
   985 ms: void test_karras_alg<unsigned int>() (/tmp/src/tests/shamrock/tree/karrasTests.cpp)
   334 ms: shammodels::basegodunov::Solver<...>::Solver(...) (/tmp/src/shammodels/ramses/src/Solver.cpp)

**** Function sets that took longest to compile / optimize:
 12026 ms: fmt::v12::detail::format_dragon(fmt::v12::detail::basic_fp<$>, unsigned int) (312 times, avg 38 ms)
  1918 ms: void test_karras_alg<$>() (2 times, avg 959 ms)

**** Expensive headers:
393716 ms: /tmp/src/shambackends/include/shambackends/sycl.hpp (included 369 times, avg 1066 ms), included via:
  31x: shamtest.hpp Test.hpp TestResult.hpp TestAssertList.hpp
  5x: <direct include>
  ...

215990 ms: /tmp/src/shambackends/include/shambackends/DeviceBuffer.hpp (included 266 times, avg 811 ms), included via:
  29x: <direct include>
  17x: reduction.hpp flatten.hpp

  done in 0.5s.
"""


class ParseReportTest(unittest.TestCase):
    def setUp(self):
        self.result = parse_report(SAMPLE)

    def test_time_summary(self):
        self.assertEqual(self.result["compilation_count"], 418)
        self.assertEqual(self.result["parse_frontend_s"], 1335.9)
        self.assertEqual(self.result["codegen_backend_s"], 1322.7)

    def test_files_parse(self):
        self.assertEqual(
            self.result["files_parse"],
            [
                {
                    "ms": 21505,
                    "file": "./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/pySPHModel.cpp.o",
                },
                {
                    "ms": 11682,
                    "file": "./src/shammodels/sph/CMakeFiles/shammodels_sph.dir/src/Solver.cpp.o",
                },
            ],
        )

    def test_files_codegen(self):
        self.assertEqual(self.result["files_codegen"][0]["ms"], 32901)
        self.assertTrue(self.result["files_codegen"][0]["file"].endswith("PatchDataField.cpp.o"))

    def test_templates(self):
        self.assertEqual(self.result["templates"][0]["count"], 2057)
        self.assertEqual(self.result["templates"][0]["avg_ms"], 17)
        self.assertEqual(self.result["template_sets"][0]["ms"], 94402)

    def test_functions(self):
        first = self.result["functions"][0]
        self.assertEqual(first["ms"], 985)
        self.assertEqual(first["name"], "void test_karras_alg<unsigned int>()")
        self.assertTrue(first["file"].endswith("karrasTests.cpp"))

    def test_function_sets(self):
        self.assertEqual(self.result["function_sets"][1]["count"], 2)
        self.assertEqual(self.result["function_sets"][1]["avg_ms"], 959)

    def test_headers(self):
        first = self.result["expensive_headers"][0]
        self.assertEqual(first["ms"], 393716)
        self.assertEqual(first["count"], 369)
        self.assertEqual(first["avg_ms"], 1066)
        self.assertEqual(
            first["included_via"][0],
            {
                "count": 31,
                "chain": ["shamtest.hpp", "Test.hpp", "TestResult.hpp", "TestAssertList.hpp"],
            },
        )
        self.assertEqual(first["included_via"][1]["chain"], ["<direct include>"])
        self.assertEqual(len(self.result["expensive_headers"]), 2)

    def test_strips_ansi(self):
        colored = "\x1b[1m\x1b[35m**** Time summary:\x1b[0m\nCompilation (2 times):\n"
        colored += "  Parsing (frontend):         1.5 s\n"
        colored += "  Codegen & opts (backend):   2.0 s\n"
        result = parse_report(colored)
        self.assertEqual(result["compilation_count"], 2)
        self.assertEqual(result["parse_frontend_s"], 1.5)
        self.assertEqual(result["codegen_backend_s"], 2.0)

    def test_docs_example(self):
        docs = Path(__file__).resolve().parents[1] / "doc/sphinx/source/dev_doc/build-profiling.md"
        text = docs.read_text()
        start = text.index("```\nAnalyzing build trace")
        end = text.index("```", start + 3)
        example = text[start + 4 : end]
        result = parse_report(example)
        self.assertEqual(result["compilation_count"], 418)
        self.assertEqual(result["parse_frontend_s"], 1335.9)
        self.assertEqual(result["codegen_backend_s"], 1322.7)
        self.assertEqual(len(result["files_parse"]), 20)
        self.assertEqual(result["files_parse"][0]["ms"], 21505)
        self.assertEqual(len(result["files_codegen"]), 20)
        self.assertEqual(result["files_codegen"][0]["ms"], 32901)
        self.assertEqual(len(result["templates"]), 30)
        self.assertEqual(result["templates"][0]["ms"], 35029)
        self.assertEqual(len(result["template_sets"]), 30)
        self.assertEqual(result["template_sets"][0]["name"], "std::vector<$>::emplace_back<$>")
        self.assertGreater(len(result["functions"]), 20)
        self.assertTrue(result["functions"][0]["file"].endswith("karrasTests.cpp"))
        self.assertGreater(len(result["function_sets"]), 20)
        self.assertGreaterEqual(len(result["expensive_headers"]), 10)
        sycl = result["expensive_headers"][0]
        self.assertTrue(sycl["file"].endswith("sycl.hpp"))
        self.assertEqual(sycl["count"], 369)
        self.assertEqual(sycl["included_via"][12]["chain"], ["<direct include>"])

    def test_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.txt"
            out = Path(tmp) / "metric__build_profile.json"
            report.write_text(SAMPLE)
            script = Path(__file__).resolve().parent / "clang_build_analyzer_to_json.py"
            subprocess.check_output([sys.executable, str(script), str(report), str(out)])
            loaded = json.loads(out.read_text())
            self.assertEqual(loaded["compilation_count"], 418)
            self.assertEqual(loaded["parse_frontend_s"], 1335.9)


if __name__ == "__main__":
    unittest.main()
