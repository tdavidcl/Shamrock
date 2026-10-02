# Implementation registry: step plan

> **Status: completed.** All 6 steps landed on upstream `main` (last one: #2506, `b1194a0`).
> These files are kept as a record; their line numbers refer to `adb21e5` and are historical.
> For the current API read `src/shamalgs/include/shamalgs/impl_registry.hpp` and
> `doc/sphinx/source/dev_doc/implementation_selection.md`. The next work (config export/import,
> env var/CLI, tuning, autotune) is planned in `.claude/plans/impl-variant-registry.md`.

Goal: replace the 5 hand-written per-algorithm implementation-control functions
(`get_default_impl_list_X`, `get_current_impl_X`, `is_impl_set_X`, `set_impl_X`,
`autoselect_impl_X`) of the 8 algorithms built on `shamalgs::ImplVariantGlobal` with a single
registry keyed by algorithm name. On the Python side, a name-keyed `shamrock.algs` API
(`set_impl("reduction", impl)`, ...) replaces the per-algorithm bindings.

The work is split into 6 steps. Each step is its own PR, and each leaves the build and the tests
green. Until step 6, every step only **adds** code or moves callers over; step 6 does all the
deletions.

## How to hand a step to an agent

Give the agent exactly two files:
- `00-background.md`, the shared, self-contained context: current code, target design,
  constraints, call-site inventory, common verification and commit rules;
- one `step-N-*.md` file.

The agent implements only that step, on its own branch following CLAUDE.md's
`claude/<type>/<short-kebab-description>` rule, and does not open a PR unless asked. When the
step is merged, update its status below.

## Steps

| Step | File | Scope | Depends on | Status |
|---|---|---|---|---|
| 1 | `step-1-compute-histogram-global.md` | Move `compute_histogram_impl` from an `inline` header global into a `.cpp` | - | done, #2470 `8ed6895` |
| 2 | `step-2-registry-core.md` | `impl_registry.hpp/.cpp`, `ImplRegistrar` for all 8 algorithms, deleted copy/move on `ImplVariantGlobal`, registry unit test | 1 | done, #2476 `d8e14f1` |
| 3 | `step-3-python-api.md` | Name-keyed `shamrock.algs` API, added next to the old bindings | 2 | done, #2480 `e6e33e6` |
| 4 | `step-4-cpp-callers.md` | Dispatch sites and C++ tests go through the registry | 2 | done, #2491 `e6b6681` |
| 5 | `step-5-python-callers-and-user-docs.md` | Benchmark scripts and the user-side docs use the new Python API | 3 | done, #2502 `c8a571d` |
| 6 | `step-6-remove-per-alg-api.md` | Delete the per-algorithm functions and bindings, and update the developer docs | 4, 5 | done, #2506 `b1194a0` |

Steps 3 and 4 both depend only on step 2, so they can run in parallel.

Follow-up: #2503 (`e4c7728`) makes the registry's `set_impl` / `autoselect_impl` log only on
rank 0.

## Deviations from the plan

What merged differs from these files in a few places:
- **Registration** uses the macro `SHAMALGS_REGISTER_IMPL(name, impl)` from `impl_registry.hpp`
  (a `shambase::call_lambda` static at namespace scope), not an `ImplRegistrar` struct.
- **Logging** in `set_impl` / `autoselect_impl` is guarded by `shamcomm::world_rank() == 0`.
- **Both optional `ImplVariantGlobal` changes were taken:** copy and move are deleted, and the
  constructor throws on an empty `AutoselectFn`.
- **New alternative:** `sort_by_keys` and `sort_by_key_pow2_len` gained `lsd_radix_sort_basic`
  (#2464), which landed alongside this work.

## After step 6

The later features (whole-config JSON export/import, `SHAMROCK_IMPL_CONFIG` / `--impl-config`,
hardware tuning, the autotune hook) are described in `.claude/plans/impl-variant-registry.md`
and summarized in `00-background.md` ("Out of scope ... but stay compatible with it").
