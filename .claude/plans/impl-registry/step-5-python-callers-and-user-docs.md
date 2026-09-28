# Step 5: benchmark scripts and user docs on the new Python API

**Read first:** `00-background.md`. It has the context, the call-site inventory ("Benchmark
scripts" and "Docs") and the common verification and commit rules.

**Depends on:** step 3, which provides the name-keyed `shamrock.algs` API.

## Why

sphinx-gallery executes every `examples/benchmarks/run_*.py` in the docs CI
(`doc/sphinx/source/conf.py`). These scripts must move to the new API before step 6 deletes the
old one.

## Scope

1. **The 7 benchmark scripts** listed under "Benchmark scripts" in the call-site inventory. The
   rename is mechanical:
   - `shamrock.algs.<fn>_<alg>(x)` becomes `shamrock.algs.<fn>("<alg>", x)`;
   - `shamrock.tree.<fn>_clbvh_dual_tree_traversal(...)`, and the
     `get_current_impl_clbvh_dual_tree_traversal_impl` getter, become
     `shamrock.algs.<fn>("clbvh_dual_tree_traversal", ...)`.

   `run_segmented_sort_in_place_performance.py` also gets
   `if not shamrock.algs.is_impl_set("segmented_sort_in_place"): shamrock.algs.autoselect_impl("segmented_sort_in_place")`
   before it reads the current implementation.

   Keep each script's output, keys and plots identical.
2. **`doc/sphinx/source/dev_doc/implementation_selection.md`, "User side (Python)" section
   (lines 20-82) only:**
   - describe the name-keyed API, with `get_registered_algs()` to discover the names;
   - update the Python examples;
   - fix "three functions" at line 22;
   - keep the explanation of the JSON config string and of the lazy default.

## Out of scope for this step

- The "Developer side (C++)" part of the doc, and the `ImplVariant.hpp` doc comment (step 6).
- Deleting the old bindings (step 6).

## Done when

- `git grep -nE '(get_default_impl_list|get_current_impl|is_impl_set|set_impl|autoselect_impl)_[a-z]' examples`
  returns nothing.
- The Python section of `implementation_selection.md` no longer names per-algorithm functions.
- Each of the 7 scripts runs to completion under `--rscript`.

## Verification

From "Common verification" in the background:
1. **Python scripts:** each of the 7 scripts. Shrink the sizes temporarily if they are slow,
   and don't commit the shrink.
2. **Lint:** pre-commit on the changed files, which covers the Python formatting and the
   markdown hooks.
3. **Commit.**
