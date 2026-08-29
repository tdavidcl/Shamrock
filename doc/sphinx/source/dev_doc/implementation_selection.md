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

Implementation selection is built on `shamalgs::ImplVariantGlobal`
(`shamalgs/include/shamalgs/ImplVariant.hpp`), a `std::variant`-based selector. This is the
pattern to use for any algorithm that needs implementation selection, whether new or being added
to an existing one.

Every such selector registers itself under its algorithm's name in the process-wide
`shamalgs::ImplVariantRegistry` (`shamalgs/include/shamalgs/ImplVariantRegistry.hpp`), which
hands it back through the non-templated `shamalgs::IImplVariant` base interface. That is what
makes it possible to query or configure *any* algorithm by name, without knowing its
alternatives at compile time. The registry is a side channel only: algorithms keep dispatching
on their own concrete selector object, so nothing on their hot path goes through it.

## User side (Python)

For every algorithm that supports it, three functions are exposed on the Python bindings, under
`shamrock.algs` for `shamalgs` primitives and `shamrock.tree` for `shamtree`'s dual tree
traversal:

- `get_default_impl_list_<algo>()` — the list of available implementations.
- `get_current_impl_<algo>()` — the implementation currently selected.
- `set_impl_<algo>(...)` — select an implementation.
- `is_impl_set_<algo>()` — whether an implementation has been selected yet.
- `autoselect_impl_<algo>()` — select the algorithm's default implementation.

Implementations are plain JSON config strings of the form
`{"implementation": "<name>", "parameters": {...}}`. `set_impl_<algo>` takes that whole string
back.

`is_impl_set_<algo>` / `autoselect_impl_<algo>` matter because `ImplVariantGlobal` has no notion
of a default until something picks one: some algorithms only pick their default the first time
they actually run, so `get_current_impl_<algo>()` returns `"null"` until then, unless you call
`autoselect_impl_<algo>()` yourself first.

```python
import shamrock

current = shamrock.algs.get_current_impl_scan_exclusive_sum_in_place()
print(current)
# null (nothing selected yet, and the algorithm hasn't run)

# two ways of selecting an implementation manually:

# 1. pick a specific one
shamrock.algs.set_impl_scan_exclusive_sum_in_place(
    '{"implementation":"std_scan","parameters":{}}'
)

# 2. or fall back to the algorithm's own default
if not shamrock.algs.is_impl_set_scan_exclusive_sum_in_place():
    shamrock.algs.autoselect_impl_scan_exclusive_sum_in_place()
```

If you want to test something against every available implementation, do:

```python
import json
import shamrock

for impl in shamrock.algs.get_default_impl_list_scan_exclusive_sum_in_place():
    shamrock.algs.set_impl_scan_exclusive_sum_in_place(impl)
    name = json.loads(impl)["implementation"]
    print(f"running with {name}")
    # ...
```

### Querying every algorithm at once

The same state is reachable by algorithm name through `shamrock.impl`, which does not need the
algorithm to be known in advance:

- `shamrock.impl.list_keys()` — the name of every algorithm supporting implementation selection.
- `shamrock.impl.is_set(key)` / `get_current(key)` / `get_default_list(key)` / `set(key, config)`
  / `autoselect(key)` — the per-algorithm functions above, taken by name.

```python
import shamrock

for key in shamrock.impl.list_keys():
    print(key, "->", shamrock.impl.get_current(key))

shamrock.impl.set("is_all_true", '{"implementation":"sum_reduction","parameters":{}}')

# same state as the per-algorithm function
print(shamrock.algs.get_current_impl_is_all_true())
```

`list_keys()` only reports the algorithms whose translation unit is linked into the running
binary, so it is a view of what this process can select, not of everything Shamrock implements.

### Where this is used in practice

The benchmark scripts under `examples/benchmarks/` sweep over every available implementation of
an algorithm this way to compare their performance: `run_segmented_sort_in_place_performance.py`,
`run_exclusive_scan_in_place.py`.

The C++ unit tests for these algorithms follow the same loop, to run every implementation against
the same reference data (see e.g. `tests/shamalgs/primitives/scan_exclusive_sum_in_placeTests.cpp`).

Selection is process-wide and does not persist: it resets to the algorithm's default every time
the process restarts.

## Developer side (C++)

### Adding implementation selection to a new algorithm

Use `shamalgs::ImplVariantGlobal`, documented in detail in
`shamalgs/include/shamalgs/ImplVariant.hpp`. Each implementation is a small tag struct exposing a
`variant_type_name`; the selector is a `std::variant` of those, and dispatch is a plain
`std::visit`. Constructing the selector takes two things: the algorithm's name, which is the key
it registers under, and a provider returning the algorithm's default implementation, which backs
`autoselect`. Skeleton, following `is_all_true.cpp` as a reference:

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

        /// Currently selected my_algo implementation
        shamalgs::ImplVariantGlobal<AltA, AltB> my_algo_impl{
            "my_algo", [](const sham::DeviceScheduler_ptr &) {
                return AltA{};
            }};

        std::vector<std::string> get_default_impl_list_my_algo() {
            return my_algo_impl.get_default_config_list();
        }

        std::string get_current_impl_my_algo() { return my_algo_impl.get_current_config(); }

        bool is_impl_set_my_algo() { return my_algo_impl.is_set(); }

        void set_impl_my_algo(const std::string &impl) { my_algo_impl.set(impl); }

        /// Called lazily on first use if no implementation was selected yet
        void autoselect_impl_my_algo(const sham::DeviceScheduler_ptr &sched) {
            my_algo_impl.autoselect(sched);
        }

    } // namespace impl

    void my_algo(sham::DeviceBuffer<T> &buf, ...) {
        if (!impl::my_algo_impl.is_set()) {
            impl::autoselect_impl_my_algo(buf.get_dev_scheduler_ptr());
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

The name passed to the constructor is the key the algorithm is known by from
`shamrock.impl` — use the same suffix as the `set_impl_<algo>` functions. Registering two
selectors under one name throws, so a typo is caught at startup rather than silently shadowing
another algorithm.

Logging of every selection happens inside `ImplVariantGlobal`, under the `impl` log domain, so
the forwarders above do not log themselves — that way a selection made through
`shamrock.impl.set` is logged too.

`ImplVariantGlobal` has no notion of a default at construction — `is_set()` starts `false`. It is
up to each call site to decide what to do when unset: the lazy-default pattern above (check
`is_set()`, autoselect right before dispatching) is what `segmented_sort_in_place` and
`scan_exclusive_sum_in_place` do, but picking a default eagerly, right where the selector is
declared, is just as valid when there is no reason to defer it.

The default provider always takes the `sham::DeviceScheduler_ptr` the algorithm will run on,
because some defaults depend on the device: `compute_histogram` picks a different implementation
on a GPU than on a CPU. Most defaults do not, and simply leave the parameter unnamed, as in the
skeleton above. Defaults that only depend on compile-time information (a `#ifdef`
backend/platform check, e.g. `scan_exclusive_sum_in_place`'s) do the same.

When the provider has more than one `return`, the lambda needs an explicit return type. Give the
selector type a name and use its `Variant` alias, as `compute_histogram` does:

```cpp
using ComputeHistogramImplVariant
    = shamalgs::ImplVariantGlobal<Reference, NaiveGpu, GpuTeamFetching, GpuOversubscribe>;

inline ComputeHistogramImplVariant compute_histogram_impl{
    "compute_histogram",
    [](const sham::DeviceScheduler_ptr &dev_sched) -> ComputeHistogramImplVariant::Variant {
        if (dev_sched->ctx->device->prop.type == sham::DeviceType::GPU) {
            return GpuOversubscribe{};
        }
        return NaiveGpu{};
    }};
```

The dispatching function already has a scheduler on hand, either as one of its own parameters or
via `buf.get_dev_scheduler_ptr()`, so threading it into `autoselect_impl_<algo>` costs nothing.
The Python binding supplies the current one, which keeps `autoselect_impl_<algo>()` argument-free
on the Python side:

```cpp
shamalgs_module.def("autoselect_impl_my_algo", []() {
    shamalgs::primitives::impl::autoselect_impl_my_algo(
        shamsys::instance::get_compute_scheduler_ptr());
});
```

An alternative with tunable fields (not currently used by any real algorithm, but supported)
specializes `shamalgs::ImplVariantParams<Alt>` to control how those fields serialize to/from the
`"parameters"` JSON — see the doc comment at the top of `ImplVariant.hpp` for a worked example
(a `group_size` field).

Once the selector and dispatch are in place, wire it up end to end:

1. Header: declare `get_default_impl_list_<algo>`, `get_current_impl_<algo>`,
   `set_impl_<algo>`, and (if using the lazy-default pattern) `is_impl_set_<algo>` and
   `autoselect_impl_<algo>` in the algorithm's `impl` namespace.
2. Python bindings (`shampylib/src/pyShamalgs.cpp` or `pyShamtree.cpp`): expose all the
   user-facing functions declared in the header under the relevant submodule.
3. Unit test: loop over `get_default_impl_list_<algo>()`, calling `set_impl_<algo>` before each
   run, then restore the implementation that was active before the loop.
4. Benchmark script (`examples/benchmarks/`, if one exists for the algorithm): same loop,
   extracting the implementation's display name with `json.loads(impl)["implementation"]`; if
   using the lazy-default pattern, call `autoselect_impl_<algo>()` first when
   `is_impl_set_<algo>()` is `False`.

## Related files

- `shamalgs/include/shamalgs/ImplVariant.hpp` — authoritative reference for `IImplVariant` and
  `ImplVariantGlobal`'s API.
- `shamalgs/include/shamalgs/ImplVariantRegistry.hpp` — the by-name registry of every algorithm's
  control variable.
- `shampylib/src/pyImplVariantRegistry.cpp` — the `shamrock.impl` bindings.
