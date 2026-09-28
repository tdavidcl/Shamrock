# Step 4: route the C++ dispatch sites and tests through the registry

**Read first:** `00-background.md`. It has the context, the constraints, the "The 8 algorithms"
table, the call-site inventory and the common verification and commit rules.

**Depends on:** step 2, which provides `shamalgs::impl_registry` and the `<algo>_impl_name`
constants. It can run in parallel with step 3.

## Why

Once dispatch goes through `impl_registry::autoselect_impl`, lazy default selection has a single
entry point. That gives one uniform "defaulting ..." log line, and it is where the later
hardware-tuning feature will plug in (see "Out of scope ... but stay compatible with it" in the
background).

## Scope

1. **Dispatch sites.** At each site in the "Dispatch site(s)" column of "The 8 algorithms" table,
   replace `impl::autoselect_impl_X(<sched>)` with
   `shamalgs::impl_registry::autoselect_impl(impl::X_impl_name, <sched>)`, and keep `<sched>` as
   it is. The sites are:
   - the 3 sites of `reduction.cpp`;
   - `is_all_true.cpp`, `scan_exclusive_sum_in_place.cpp`, `segmented_sort_in_place.cpp`,
     `sort_by_key_pow2_len.cpp` and `sort_by_keys.cpp`;
   - `compute_histogram.hpp:374`, which needs `#include "shamalgs/impl_registry.hpp"`;
   - `CLBVHDualTreeTraversal.cpp:97`.

   Keep the `if (!impl::X_impl.is_set())` guard, and the direct `impl::X_impl.get()` that feeds
   `std::visit`.
2. **C++ tests.** Migrate the 8 test files listed under "C++ tests" in the call-site inventory:
   `ns::impl::<fn>_<alg>(...)` becomes `shamalgs::impl_registry::<fn>("<alg>", ...)`.
   - Keep each loop's shape: autoselect if unset, save the current implementation, loop over the
     list calling set, then restore.
   - Keep `compute_histogram_tests`' reliance on `"reference"` being first in the list.
   - Include `shamalgs/impl_registry.hpp` where needed.
   - In `algorithmTests.cpp`, include `shamsys/NodeInstance.hpp` directly; today it only arrives
     through `sortTests.hpp`.

## Out of scope for this step

- Deleting the per-algorithm functions. They stay, still work, and are still used by the old
  Python bindings (step 6).
- Python bindings, benchmark scripts and docs (steps 3, 5 and 6).
- The `sycl::buffer` entry points that bypass the selector (`is_all_true.cpp:268`,
  `sort_by_key_pow2_len.cpp:79`). Leave them alone.

## Done when

- `git grep -nE 'impl::autoselect_impl_[a-z]' src/shamalgs src/shamtree` only hits the
  per-algorithm functions' own definitions and declarations, not dispatch sites.
- `git grep -nE '(get_default_impl_list|get_current_impl|is_impl_set|set_impl|autoselect_impl)_[a-z]' src/tests`
  returns nothing.
- The full unit-test suite passes.

## Verification

From "Common verification" in the background:
1. **Build:** `shammake shamalgs shamtree`, then the full build.
2. **Unit tests:** the full `--unittest` run.
3. **Lint:** pre-commit on the changed files, and `clang-tidy-check.py` on one edited algorithm
   `.cpp` and one edited test file.
4. **Commit.**
