# Implementation selection

Several performance-critical algorithms in Shamrock ship with more than one concrete
implementation (e.g. a portable fallback vs. a backend-specific accelerated kernel, or several
kernel strategies tuned for different problem sizes). Implementation selection is the mechanism
that lets you list the implementations available for such an algorithm, query the one currently
active, and switch it at runtime, from Python, without recompiling.

This page documents both sides of it: how to use it as a user (e.g. for benchmarking, or to work
around a bad implementation on a given platform), and how to wire a new algorithm into it as a
developer.

## Current status

Two flavors of implementation selection currently coexist in the codebase:

- **Legacy pattern**: a hand-rolled `enum class` + a global variable + manual
  `name <-> enum` string mapping functions. Still used by:

  | Algorithm | Files |
  |-----------|-------|
  | `shamalgs::primitives::sum` / `min` / `max` (reduction) | `shamalgs/primitives/reduction.{hpp,cpp}` |
  | `shamalgs::primitives::is_all_true` | `shamalgs/primitives/is_all_true.{hpp,cpp}` |
  | `shamtree::clbvh_dual_tree_traversal` | `shamtree/CLBVHDualTreeTraversal.{hpp,cpp}` |

- **Generic implementation selector**: `shamalgs::ImplVariantGlobal`
  (`shamalgs/include/shamalgs/ImplVariant.hpp`), a `std::variant`-based selector that replaces the
  legacy pattern's enum/global-variable/mapping boilerplate with one-liners. Currently used by:

  | Algorithm | Files |
  |-----------|-------|
  | `shamalgs::primitives::segmented_sort_in_place` | `shamalgs/primitives/segmented_sort_in_place.{hpp,cpp}` |
  | `shamalgs::primitives::scan_exclusive_sum_in_place` | `shamalgs/primitives/scan_exclusive_sum_in_place.{hpp,cpp}` |

`ImplVariantGlobal` is the pattern to use for any **new** algorithm that needs implementation
selection, and the target when migrating an existing algorithm off the legacy pattern.

Both flavors expose the same three-function shape per algorithm
(`get_default_impl_list_<algo>`, `get_current_impl_<algo>`, `set_impl_<algo>`), but differ in the
exact ABI, described below.

:::{note}
A third, unrelated mechanism, `shamalgs::primitives::ImplControl`
(`shamalgs/include/shamalgs/ImplControl.hpp`), also exists and is used by `compute_histogram`. It
picks a config per `DeviceScheduler` (with an optional autotuning hook) rather than a single
process-wide implementation, and is out of scope for this page.
:::

## User side (Python)

For every algorithm that supports it, three functions are exposed on the Python bindings, under
`shamrock.algs` for `shamalgs` primitives and `shamrock.tree` for `shamtree`'s dual tree
traversal:

- `get_default_impl_list_<algo>()` — the list of available implementations.
- `get_current_impl_<algo>()` — the implementation currently selected.
- `set_impl_<algo>(...)` — select an implementation.

The exact shape of what these pass around depends on which ABI flavor the algorithm uses.

### New ABI (`ImplVariantGlobal`-based algorithms)

Implementations are plain JSON config strings of the form
`{"implementation": "<name>", "parameters": {...}}`. `set_impl_<algo>` takes that whole string
back.

```python
import json
import shamrock

current = shamrock.algs.get_current_impl_scan_exclusive_sum_in_place()
print(current)
# {"implementation":"decoupled_lookback_512","parameters":{}}

for impl in shamrock.algs.get_default_impl_list_scan_exclusive_sum_in_place():
    shamrock.algs.set_impl_scan_exclusive_sum_in_place(impl)
    name = json.loads(impl)["implementation"]
    print(f"running with {name}")
    # ...
```

### Legacy ABI (`enum`-based algorithms)

Implementations are `shamrock.algs.impl_param` objects with two plain string fields,
`impl_name` and `params` (the latter is unused by every algorithm still on this pattern — it is
always an empty string). `set_impl_<algo>` takes the two as separate arguments.

```python
import shamrock

current = shamrock.algs.get_current_impl_reduction()
print(current.impl_name, current.params)
# group_reduction128 ""

for impl in shamrock.algs.get_default_impl_list_reduction():
    shamrock.algs.set_impl_reduction(impl.impl_name, impl.params)
    print(f"running with {impl.impl_name}")
    # ...
```

### Where this is used in practice

The benchmark scripts under `examples/benchmarks/` sweep over every available implementation of
an algorithm this way to compare their performance:

- `run_segmented_sort_in_place_performance.py`, `run_exclusive_scan_in_place.py` (new ABI)
- `run_reduction_performance.py`, `run_is_all_true_performance.py`, `run_dtt_performance.py`
  (legacy ABI)

The C++ unit tests for these algorithms follow the same loop, to run every implementation against
the same reference data (see e.g. `tests/shamalgs/primitives/scan_exclusive_sum_in_placeTests.cpp`).

Selection is process-wide and does not persist: it resets to the algorithm's default every time
the process restarts.

## Developer side (C++)

### Adding implementation selection to a new algorithm

Use `shamalgs::ImplVariantGlobal`, documented in detail in
`shamalgs/include/shamalgs/ImplVariant.hpp`. Each implementation is a small tag struct exposing a
`variant_type_name`; the selector is a `std::variant` of those, and dispatch is a plain
`std::visit`. Skeleton, following `scan_exclusive_sum_in_place.cpp` as a reference:

```cpp
#include "shamalgs/ImplVariant.hpp"
#include "shambase/overloaded.hpp"

namespace shamalgs::primitives {

    namespace impl {

        /// One-line doc per alternative: what it does / when it's a good fit
        struct AltA {
            static constexpr std::string_view variant_type_name = "alt_a";
        };
        struct AltB {
            static constexpr std::string_view variant_type_name = "alt_b";
        };

        shamalgs::ImplVariantGlobal<AltA, AltB> my_algo_impl;

        std::vector<std::string> get_default_impl_list_my_algo() {
            return decltype(my_algo_impl)::get_default_config_list();
        }

        std::string get_current_impl_my_algo() { return my_algo_impl.get_current_config(); }

        bool is_impl_set_my_algo() { return my_algo_impl.is_set(); }

        void set_impl_my_algo(const std::string &impl) { my_algo_impl.set(impl); }

        /// Called lazily on first use if no implementation was selected yet
        void autoselect_impl_my_algo() { my_algo_impl.set(AltA{}); }

    } // namespace impl

    void my_algo(...) {
        if (!impl::my_algo_impl.is_set()) {
            impl::autoselect_impl_my_algo();
        }

        std::visit(
            shambase::overloaded{
                [&](impl::AltA) { /* ... */ },
                [&](impl::AltB) { /* ... */ },
            },
            impl::my_algo_impl.get());
    }

} // namespace shamalgs::primitives
```

`ImplVariantGlobal` has no notion of a default at construction — `is_set()` starts `false`. Some
algorithms (e.g. `reduction`, `is_all_true`) instead pick a default eagerly, right where the
selector is declared, when there is no reason to defer it; either is fine, pick whichever reads
more naturally for the algorithm at hand.

An alternative with tunable fields (not currently used by any real algorithm, but supported)
specializes `shamalgs::ImplVariantParams<Alt>` to control how those fields serialize to/from the
`"parameters"` JSON — see the doc comment at the top of `ImplVariant.hpp` for a worked example
(a `group_size` field).

Once the selector and dispatch are in place, wire it up end to end:

1. Header: declare `get_default_impl_list_<algo>`, `get_current_impl_<algo>`,
   `set_impl_<algo>`, and (if using the lazy-default pattern) `is_impl_set_<algo>` and
   `autoselect_impl_<algo>` in the algorithm's `impl` namespace.
2. Python bindings (`shampylib/src/pyShamalgs.cpp` or `pyShamtree.cpp`): expose the three
   user-facing functions under the relevant submodule.
3. Unit test: loop over `get_default_impl_list_<algo>()`, calling `set_impl_<algo>` before each
   run, then restore the implementation that was active before the loop.
4. Benchmark script (`examples/benchmarks/`, if one exists for the algorithm): same loop,
   extracting the implementation's display name with `json.loads(impl)["implementation"]`.

### Legacy pattern

The algorithms not yet migrated (`reduction`, `is_all_true`, `clbvh_dual_tree_traversal`) still
follow the older shape:

```cpp
enum class MY_ALGO_IMPL : u32 { ALT_A, ALT_B };
MY_ALGO_IMPL my_algo_impl = MY_ALGO_IMPL::ALT_A;

inline MY_ALGO_IMPL my_algo_impl_from_params(const std::string &impl) {
    if (impl == "alt_a") return MY_ALGO_IMPL::ALT_A;
    if (impl == "alt_b") return MY_ALGO_IMPL::ALT_B;
    throw shambase::make_except_with_loc<std::invalid_argument>(/* ... */);
}

inline shamalgs::impl_param my_algo_impl_to_params(const MY_ALGO_IMPL &impl) {
    if (impl == MY_ALGO_IMPL::ALT_A) return {.impl_name = "alt_a", .params = ""};
    if (impl == MY_ALGO_IMPL::ALT_B) return {.impl_name = "alt_b", .params = ""};
    throw shambase::make_except_with_loc<std::invalid_argument>(/* ... */);
}

// dispatch: switch (my_algo_impl) { case MY_ALGO_IMPL::ALT_A: ...; case MY_ALGO_IMPL::ALT_B: ...; }
```

Do not use this pattern for new code — use `ImplVariantGlobal` instead. When touching one of the
three algorithms still on it, migrating it to `ImplVariantGlobal` first is welcome; the diff that
migrated `scan_exclusive_sum_in_place` is a good reference for what that migration looks like end
to end (header, implementation, Python bindings, unit test, benchmark script).

## Related files

- `shamalgs/include/shamalgs/ImplVariant.hpp` — authoritative reference for
  `ImplVariantGlobal`'s API.
- `shamalgs/include/shamalgs/impl_utils.hpp` — the legacy ABI's `impl_param` struct.
