# Step 6: delete the per-algorithm API

**Read first:** `00-background.md`. It has the context, the "The 8 algorithms" table, the
call-site inventory and the common verification and commit rules.

**Depends on:** steps 4 and 5. By now nothing outside the old functions and bindings calls them:
the C++ callers moved in step 4, and the Python callers in step 5.

## Scope

1. **C++ per-algorithm functions.** Using the "Per-algorithm functions" column of "The 8
   algorithms" table:
   - delete `get_default_impl_list_X`, `get_current_impl_X`, `is_impl_set_X`, `set_impl_X` and
     `autoselect_impl_X` from the 6 `primitives/*.cpp` files and their headers;
   - delete the inline ones at `compute_histogram.hpp:65-91`;
   - delete the DTT ones in `CLBVHDualTreeTraversal.cpp` and `CLBVHDualTreeTraversal.hpp:66-83`;
   - delete any `namespace impl` block in a header that ends up empty, and drop any include
     that was only needed by them.

   Keep the globals, the lambdas, the name constants and the registrars.
2. **Python bindings.**
   - Delete the per-algorithm blocks in `src/shampylib/src/pyShamalgs.cpp` (the table under
     "Python bindings" in the inventory). Keep the name-keyed block from step 3 and the legacy
     `impl_param` binding.
   - Delete the DTT block at `src/shampylib/src/pyShamtree.cpp:93-112`.
3. **Developer docs.**
   - `implementation_selection.md` "Developer side (C++)" (lines 84-228): keep the explanation of
     the `AutoselectFn` constructor lambda and the scheduler, and document the name constant,
     the `ImplRegistrar` and dispatch through `impl_registry::autoselect_impl`.
   - The "Wire it up end to end" list (around lines 215-228): the header now declares nothing,
     and no Python binding is needed per algorithm.
   - The `ImplVariant.hpp` class doc comment: replace the sentence about making the
     "get_default_impl_list_X / get_current_impl_X / set_impl_X free functions" one-liners with
     a mention of `impl_registry`.

## Out of scope for this step

The later features listed in "Out of scope ... but stay compatible with it" in the background.

## Done when

- This returns nothing in `src/`, `examples/` or `doc/`:
  `git grep -nE '(get_default_impl_list|get_current_impl|is_impl_set|set_impl|autoselect_impl)_[a-z]'`
- The full unit-test suite passes.
- All 7 benchmark scripts run under `--rscript`.
- `shamrock.algs` exposes only the name-keyed implementation functions.

## Verification

From "Common verification" in the background:
1. **Build:** the full build. Deleting declarations touches many targets.
2. **Unit tests:** the full `--unittest` run.
3. **Python scripts:** the 7 benchmark scripts, plus a one-liner checking that
   `shamrock.algs.set_impl_reduction` no longer exists.
4. **Lint:** pre-commit, and `clang-tidy-check.py` on one edited algorithm `.cpp` and
   `pyShamalgs.cpp`.
5. **Commit.**
