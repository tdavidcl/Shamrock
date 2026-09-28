# Step 1: move `compute_histogram_impl` out of the header

**Read first:** `00-background.md`. It has the context, the constraints, the code background and
the common verification and commit rules.

**Depends on:** nothing.

## Why

`compute_histogram_impl` is an `inline` global in
`src/shamalgs/include/shamalgs/primitives/compute_histogram.hpp:56-63`. Every includer holds its
own copy, and only the dynamic linker merges them. The includers are:
- `src/shampylib/src/pyShamalgs.cpp`;
- `src/shammodels/common/src/pyCommonUtils.cpp`;
- `src/tests/shamalgs/primitives/compute_histogram_tests.cpp`.

Step 2 makes every global register itself in the registry at static initialization, so there
must be exactly one definition. See "Build, linking and static-initialization facts" in the
background.

This step is a pure refactor: **no behavior change**.

## Scope

1. **`compute_histogram.hpp`:**
   - Keep the alternative structs (`Reference`, `NaiveGpu`, `GpuTeamFetching`,
     `GpuOversubscribe`).
   - Replace the `inline` global with
     `using ComputeHistogramImpl = shamalgs::ImplVariantGlobal<Reference, NaiveGpu, GpuTeamFetching, GpuOversubscribe>;`
     and `extern ComputeHistogramImpl compute_histogram_impl;`, both in `namespace impl`.
   - Leave the 5 inline per-algorithm functions (lines 65-91) and the dispatch site (line 374)
     unchanged. They compile against the `extern` declaration.
2. **New `src/shamalgs/src/primitives/compute_histogram.cpp`:**
   - the license header and a doxygen `@file` block;
   - include `shamalgs/primitives/compute_histogram.hpp`;
   - define `ComputeHistogramImpl compute_histogram_impl{...}` in
     `shamalgs::primitives::impl`, with the current lambda moved **verbatim**, including its
     `prop.type == sham::DeviceType::GPU` check.
3. **`src/shamalgs/CMakeLists.txt`:** add `src/primitives/compute_histogram.cpp` to the explicit
   `Sources` list.

## Out of scope for this step

- The registry, the name constant and the `ImplRegistrar` (step 2).
- The null check on the scheduler. It will live in `impl_registry::autoselect_impl`; do not add
  it to the lambda.
- Any other algorithm.

## Done when

- No `inline` global remains in `compute_histogram.hpp`, and exactly one definition exists, in
  the new `.cpp`.
- `compute_histogram_tests` passes, and `examples/benchmarks/run_compute_histogram.py` runs
  unchanged.

## Verification

From "Common verification" in the background:
1. **Build:** `shammake shamalgs shampylib shammodels_common`, then the full build.
2. **Unit tests:** at least the `compute_histogram` tests (use `--run-only` if the full suite is
   slow), then the full `--unittest` run.
3. **Python scripts:** `run_compute_histogram.py`.
4. **Lint:** pre-commit on the changed files, and `clang-tidy-check.py` on the new `.cpp`.
5. **Commit.**
