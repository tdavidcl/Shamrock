# Step 3: name-keyed Python API, next to the old bindings

**Read first:** `00-background.md`. It has the context, the constraints, the Python binding
conventions and the common verification and commit rules.

**Depends on:** step 2 (`shamalgs::impl_registry`). It can run in parallel with step 4.

## Scope

Add the 6 name-keyed functions to `shamalgs_module` in `src/shampylib/src/pyShamalgs.cpp`
(module created at line 40). Each one forwards to `shamalgs::impl_registry`:

| Python | C++ |
|---|---|
| `get_registered_algs() -> list[str]` | `impl_registry::get_registered_algs()` |
| `get_default_impl_list(alg) -> list[str]` | `impl_registry::get_default_impl_list(alg)` |
| `get_current_impl(alg) -> str` | `impl_registry::get_current_impl(alg)` (`"null"` when unset) |
| `is_impl_set(alg) -> bool` | `impl_registry::is_impl_set(alg)` |
| `set_impl(alg, impl)` | `impl_registry::set_impl(alg, impl)` |
| `autoselect_impl(alg)` | `impl_registry::autoselect_impl(alg, shamsys::instance::get_compute_scheduler_ptr())` |

- Use `py::arg("alg")` / `py::arg("impl")`, and an `R"pbdoc(...)pbdoc"` docstring as the last
  argument. The docstring gives the JSON string format of an implementation,
  `{"implementation": ..., "parameters": ...}`.
- **Only `autoselect_impl` fetches the compute scheduler.** The other functions must work before
  `shamrock.sys.init()`. Before init, `autoselect_impl` raises cleanly, because the registry
  null-checks the scheduler.
- Put the block in its own `{ // implementation registry ... }` scope, next to the existing
  per-algorithm blocks.

## Out of scope for this step

- The old per-algorithm bindings in `pyShamalgs.cpp` and `pyShamtree.cpp`. They stay (step 6).
- Benchmark scripts and docs (step 5).
- `pyShamtree.cpp`. The DTT is reachable through `shamrock.algs.*("clbvh_dual_tree_traversal")`.
- The legacy `impl_param` binding (`pyShamalgs.cpp:44-67`).

## Done when

A scratch script run through `--rscript`, and not committed, shows that:
- `shamrock.algs.get_registered_algs()` contains all 8 names;
- for each algorithm, `autoselect_impl` makes `is_impl_set` true, and a
  `set_impl`/`get_current_impl` round trip over `get_default_impl_list` works;
- `shamrock.algs.set_impl("nope", "{}")` raises a Python exception;
- the old per-algorithm bindings still work, e.g. `shamrock.algs.get_current_impl_reduction()`.

If you can run in lib mode (`import shamrock` without `sys.init()`), also check that
`autoselect_impl("reduction")` raises instead of crashing.

## Verification

From "Common verification" in the background:
1. **Build:** `shammake shampylib`, then the full build.
2. **Python scripts:** the scratch script above.
3. **Lint:** pre-commit, and `clang-tidy-check.py` on `pyShamalgs.cpp`.
4. **Commit.**
