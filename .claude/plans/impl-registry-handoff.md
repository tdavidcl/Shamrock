# Handoff: name-keyed registry for implementation selection

You are implementing **one** piece of a larger design: a registry that lets code and Python select
an algorithm's implementation by name. Read `AGENTS.md` and `CLAUDE.md` first: build, test, commit
authorship and the no-session-link rules all apply.

This document is self-contained: everything you need about the current code, the target design,
and the later work this must stay compatible with is below. Line numbers were taken at commit
`2922d08` on `main`. Re-check them with `grep` before editing, because they may have drifted.

Repository facts:
- Project: SHAMROCK, a C++20 / SYCL / MPI / pybind11 hydrodynamics code.
- Upstream: `Shamrock-code/Shamrock`, base branch `main`.
- Work on the branch you were assigned. Do not open a PR unless asked.

## Goal

Today each of 8 algorithms hand-writes the same 5 free functions around its
`shamalgs::ImplVariantGlobal<...>` global, plus 5 matching Python bindings. For `reduction`:

- `get_default_impl_list_reduction()`
- `get_current_impl_reduction()`
- `is_impl_set_reduction()`
- `set_impl_reduction(impl)`
- `autoselect_impl_reduction()`

Replace all of them with one registry keyed by algorithm name:

```python
shamrock.algs.get_registered_algs()
shamrock.algs.get_default_impl_list("reduction")
shamrock.algs.autoselect_impl("reduction")
shamrock.algs.is_impl_set("reduction")
shamrock.algs.set_impl("reduction", impl)
shamrock.algs.get_current_impl("reduction")
```

Each call site that used a per-algorithm function (tests, Python bindings, benchmark scripts, docs)
switches to the registry. The per-algorithm functions are then deleted.

The registry works entirely through `IImplVariant`. To make that possible, `IImplVariant` gains
two virtuals, `is_set()` and `autoselect(sched)`, so the registry never needs extra
per-algorithm callbacks or JSON parsing.

## Hard constraints

1. **`IImplVariant` gains exactly two pure virtuals and nothing else**
   (`src/shamalgs/include/shamalgs/ImplVariant.hpp:208-221`):
   - `virtual bool is_set() const = 0;`
   - `virtual void autoselect(const sham::DeviceScheduler_ptr &sched) = 0;`

   Its existing members (`get_current_config`, `get_default_config_list`, `set(std::string_view)`)
   stay unchanged. The registry works purely through `IImplVariant`, so it never casts to a
   concrete type.
2. **`ImplVariantGlobal` changes only as needed to implement those two virtuals** (see "Changes
   to `ImplVariant.hpp`" below). No name in the constructor, no `get_or_autoselect`, no
   removal of `set(Variant)`.
3. The Python value types stay the same: one implementation is a JSON **string**
   (`{"implementation": ..., "parameters": ...}`), and the default list is a `list[str]`.

## Out of scope (do NOT implement), but stay compatible with it

These features are planned as later, separate changes on top of this registry. Do not implement
any of them, but do not make design choices that would block them either.

| Later feature | What it will do | What this change must keep possible |
|---|---|---|
| Whole-config JSON export and import | `get_impl_config()` returns `{"device": {...}, "sycl": {...}, "config": {"<alg>": <impl config>, ...}}`, leaving out algorithms that are not set. `set_impl_config(json)` applies `config` entry by entry through the registry's `set_impl`, warning on and skipping unknown algorithms or implementations. | The registry can enumerate every algorithm (`get_registered_algs`) and read and write each one by name. **Every change of selection goes through `impl_registry::set_impl`, or through `IImplVariant::set`, which it calls.** |
| Env var `SHAMROCK_IMPL_CONFIG` and CLI `--impl-config <file>` | Load such a JSON file at the end of `shamsys::instance::init_sycl_mpi` (env var) and in `shamsys::instance::init(argc, argv)` (CLI). | The registry lives in shamalgs, which shamsys already links. It is fully populated by static initialization, before `main` or before the Python module import completes. |
| Hardware tuning | User-supplied `"tunings": [{"match": {"device": {...}, "sycl": {...}}, "config": {...}}]` entries. When an algorithm autoselects, the most specific entry that matches the scheduler's device wins; an unusable entry falls back to the next one, and then to the hard-coded default. | Autoselect must always receive the **scheduler**, so that it can see the device (`sched->ctx->device->prop`). That is why `IImplVariant::autoselect` takes a `sham::DeviceScheduler_ptr`. |
| Autotune hook | An optional per-algorithm autotuner on `IImplVariant`, with "none" as the default, rolled out one algorithm at a time. | Nothing extra; just do not add it now. |
| Compiler-id move | Move the generated `shamrock_compiler_id_string` from shamlib down into shambackends, for the export's `sycl` block. | Nothing. |

Also out of scope: any change to `ImplVariant.hpp` beyond the two virtuals and what implementing
them requires.

## Background: how implementation selection works today

### `src/shamalgs/include/shamalgs/ImplVariant.hpp`

This header holds a `std::variant`-based selector.

**Alternatives.** Each implementation of an algorithm is a small struct with a
`static constexpr std::string_view variant_type_name`. For example:
`struct Fallback { static constexpr std::string_view variant_type_name = "fallback"; };`.

**Config string format.** An implementation serializes to a single JSON string:
`{"implementation": "<variant_type_name>", "parameters": {...}}`. The free functions that
handle it are:
- `variant_to_config_string(v)`;
- `variant_from_config_string<Variant>(s)`, which throws `std::invalid_argument` for an unknown
  name and nlohmann exceptions for malformed JSON;
- `variant_default_type_names<Variant>()`.

**`ImplVariantParams<Alt>`** (lines 80-86) is a trait for alternatives with tunable fields.
Specializing it gives the alternative's own `to_json`/`from_json` for `"parameters"`. The default
serializes to `{}`. Example: `GroupReduction{u32 group_size}` in
`src/shamalgs/src/primitives/reduction.cpp:57-71`.

**`HasCustomDefaults`** (lines 107-110): an alternative may define
`static std::vector<Alt> variant_custom_defaults()`. It is then listed once per returned instance
in the default list, e.g. `GroupReduction{16}, {128}, {256}`.

**`IImplVariant`** (lines 208-221) is the non-template interface:
- `get_current_config()`, which returns the string `"null"` when unset;
- `get_default_config_list()`;
- `set(std::string_view config_json)`.

Today nothing collects instances of it.

**`ImplVariantGlobal<Alts...>`** (lines 244-281) implements `IImplVariant` and holds
`std::optional<std::variant<Alts...>> current`. It starts unset, has no notion of a default, and
provides:
- `is_set()`, which is **not** virtual today;
- `get()`, which requires `is_set()`;
- `set(Variant)`;
- `set(std::string_view)`, which parses first and assigns only on success;
- `get_current_config()` and `get_default_config_list()`.

### The current per-algorithm pattern (what you are removing)

From `src/shamalgs/src/primitives/reduction.cpp:73-144`:

```cpp
namespace impl {
    shamalgs::ImplVariantGlobal<Fallback
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        , GroupReduction
#endif
        > reduction_impl;

    std::vector<std::string> get_default_impl_list_reduction() { return reduction_impl.get_default_config_list(); }
    std::string get_current_impl_reduction() { return reduction_impl.get_current_config(); }
    bool is_impl_set_reduction() { return reduction_impl.is_set(); }
    void set_impl_reduction(const std::string &impl) {
        shamlog_info_ln("algs", "setting reduction implementation to impl :", impl);
        reduction_impl.set(impl);
    }
    void autoselect_impl_reduction() {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        reduction_impl.set(GroupReduction{});
#else
        reduction_impl.set(Fallback{});
#endif
        shamlog_info_ln("algs", "defaulting reduction implementation to impl :", get_current_impl_reduction());
    }
}

template<class T>
T sum(const sham::DeviceScheduler_ptr &sched, const sham::DeviceBuffer<T> &buf1, u32 start_id, u32 end_id) {
    if (!impl::reduction_impl.is_set()) {
        impl::autoselect_impl_reduction();   // lazy default on first use
    }
    return std::visit(shambase::overloaded{
            [&](impl::Fallback) { return sum_usm_fallback(sched, buf1, start_id, end_id); },
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
            [&](impl::GroupReduction cfg) { return sum_usm_group(sched, buf1, start_id, end_id, cfg.group_size); },
#endif
        }, impl::reduction_impl.get());
}
```

- The 5 functions are declared in the header's `namespace impl` block (e.g. `reduction.hpp:140-157`).
- Every algorithm's dispatch site does the lazy `if (!is_set()) autoselect` before `std::visit`.
- Some entry points taking a `sycl::buffer` bypass the selector entirely, e.g.
  `is_all_true.cpp:264-279` and `sort_by_key_pow2_len.cpp:77-92`. Leave them as they are.

### Current default selections (these move into the constructor lambdas)

| Algorithm | Alternatives (`variant_type_name`) | Current `autoselect` body |
|---|---|---|
| `reduction` | `fallback`; `group_reduction` (`group_size`, defaults `{16,128,256}`, only under `#ifdef SYCL2020_FEATURE_GROUP_REDUCTION`) | `GroupReduction{}` if `SYCL2020_FEATURE_GROUP_REDUCTION`, else `Fallback{}` (`reduction.cpp:105-115`) |
| `is_all_true` | `host`, `sum_reduction`, `atomic_early_exit` (`group_size`, defaults `{64,256}`) | `Host{}` (`is_all_true.cpp:228-234`) |
| `scan_exclusive_sum_in_place` | `std_scan`; `std_scan_single_task_acpp` (`#ifdef __ACPP__`); `decoupled_lookback_512` (`#ifdef SYCL2020_FEATURE_GROUP_REDUCTION`); `acpp_alg` (`#ifdef ACPP_ALG_AVAILABLE`) | Nested `#ifdef __MACH__` / `__ACPP__` / `SYCL2020_FEATURE_GROUP_REDUCTION` choice (`scan_exclusive_sum_in_place.cpp:179-197`). Keep it verbatim. |
| `segmented_sort_in_place` | `local_insertion_sort`, `multi_std_sort` | `MultiStdSort{}` (`segmented_sort_in_place.cpp:144-150`) |
| `sort_by_key_pow2_len` | `bitonic_sort` (`stencil_size`, defaults `{16,32}`), `std_sort` | `BitonicSort{}` (`sort_by_key_pow2_len.cpp:120-126`) |
| `sort_by_keys` | `std_sort`, `batcher_odd_even_host_serial`, `batcher_odd_even` | `StdSort{}` (`sort_by_keys.cpp:95-101`) |
| `compute_histogram` | `reference`, `naive_gpu`, `gpu_team_fetching`, `gpu_oversubscribe` | Depends on the device at runtime: `GpuOversubscribe{}` if `dev_sched->ctx->device->prop.type == sham::DeviceType::GPU`, else `NaiveGpu{}` (`compute_histogram.hpp:79-89`). It is the only one that already takes a scheduler. |
| `clbvh_dual_tree_traversal` | `reference`, `parallel_select`, `scan_multipass` | `ScanMultipass{}` (`CLBVHDualTreeTraversal.cpp:66-72`); it logs under the `"tree"` tag |

`shamalgs::primitives::impl::StdSort` is defined identically in both `sort_by_keys.cpp:58-60` and
`sort_by_key_pow2_len.cpp:54-56`. That is pre-existing; leave it unless it causes a real problem.

### Build, linking and static-initialization facts relevant to self-registration

- **Library kinds.** Every library is either `SHARED` (the default, `SHAMROCK_USE_SHARED_LIB=On`
  in `cmake/ShamrockBuildOptions.cmake:18`) or `OBJECT` (`Off`, forced on Apple and in coverage
  builds). There are **no** `STATIC` archives, so no translation unit holding a registrar can be
  dropped by the linker.
  - OBJECT libraries are linked directly into each final binary (`shamrock`, `shamrock_test`, the
    `pyshamrock` Python module), once each.
- **Dependencies.**
  - shamalgs links only shambackends (`src/shamalgs/CMakeLists.txt:60`).
  - shamtree links shamalgs, shammath and shamsys.
  - shamsys links shamalgs.
  - So **shamalgs must not call shamsys**. Take the scheduler as a parameter instead.
- **The codebase already relies on static-init registration inside libraries:**
  - `ON_PYTHON_INIT` in `src/shambindings/include/shambindings/pybindaliases.hpp:45-71`;
  - the shamsolvergraph JSON registry, a Meyers singleton,
    `src/shamsolvergraph/include/shamsolvergraph/JsonSerializable.hpp:140-206`.
- **Initialization order.** Within one translation unit, namespace-scope objects are initialized
  in definition order, so a registrar defined after its global sees a constructed global.
  - The registry singleton is created during the first registration, so it is destroyed after
    every global (a single atexit chain, even across shared libraries).

### Getting a scheduler

- **Type.** `sham::DeviceScheduler_ptr` is `std::shared_ptr<sham::DeviceScheduler>`
  (`src/shambackends/include/shambackends/DeviceScheduler.hpp:73`).
- **Device properties** are at `sched->ctx->device->prop`, a `sham::DeviceProperties` in
  `src/shambackends/include/shambackends/Device.hpp:84-138`.
- **From a buffer:** `sham::DeviceBuffer<T>::get_dev_scheduler_ptr()` (`DeviceBuffer.hpp:439-449`,
  with both const and non-const overloads).
- **Process-wide compute scheduler:** `shamsys::instance::get_compute_scheduler_ptr()`
  (`src/shamsys/include/shamsys/NodeInstance.hpp:137-141`). It is usable from shampylib and the
  tests, but **not** from shamalgs. It returns a null pointer until `shamrock.sys.init(...)`
  (lib mode) or `--sycl-cfg` (executable) has initialized the devices.

### Python binding conventions

- **Module layout.** The compiled module is `pyshamrock`, re-exported as `shamrock`.
  - `src/shampylib/src/pyShamalgs.cpp:40` creates the `algs` submodule inside an `ON_PYTHON_INIT`
    block: `py::module shamalgs_module = m.def_submodule("algs", ...)`.
  - `src/pylib/shamrock/algs/__init__.py` re-exports everything automatically, so no Python-side
    change is needed.
- **Existing bindings.** Today every implementation binding is a bare lambda with no docstring
  and no `py::arg`.
- **Style for new bindings:** use `py::arg("name")`, and an `R"pbdoc(...)pbdoc"` docstring as the
  last argument (see `src/shampylib/include/shampylib/pyNodeInstance.hpp`).
- **Types.** `std::vector<std::string>` converts to `list[str]` through `pybind11/stl.h`, which is
  already included through `shambindings/pybind11_stl.hpp`. C++ exceptions surface as Python
  exceptions.
- **Includes.** `pyShamalgs.cpp` already includes `shamsys/NodeInstance.hpp`, and shampylib links
  shamsys.

### Test framework conventions (`src/shamtest/shamtest.hpp`)

- **Declaring a test.** `NEW_TEST(Unittest, "name", 1) { ... }`, where the last argument is the
  MPI node count. See `src/tests/shamalgs/primitives/reductionTests.cpp:21` and copy its style.
- **Assertions.** `REQUIRE(cond)`, `REQUIRE_EQUAL(a, b)`, and
  `REQUIRE_EXCEPTION_THROW(expr, ExceptionType)`. The last one is a macro, so wrap any
  expression containing commas in a lambda.
- **Discovery.** `src/tests/CMakeLists.txt` collects `*.cpp` with `GLOB_RECURSE CONFIGURE_DEPENDS`,
  so new test files need no CMake edit.
- **Scheduler.** Tests get it with `shamsys::instance::get_compute_scheduler_ptr()`.

## The 8 algorithms

| Registry name | Global (file:line) | Scheduler at the dispatch site |
|---|---|---|
| `reduction` | `reduction_impl`, `src/shamalgs/src/primitives/reduction.cpp:78` | the `sched` argument |
| `is_all_true` | `is_all_true_impl`, `src/shamalgs/src/primitives/is_all_true.cpp:208` | `buf.get_dev_scheduler_ptr()` |
| `scan_exclusive_sum_in_place` | `scan_exclusive_sum_in_place_impl`, `src/shamalgs/src/primitives/scan_exclusive_sum_in_place.cpp:139` | `buf1.get_dev_scheduler_ptr()` |
| `segmented_sort_in_place` | `segmented_sort_in_place_impl`, `src/shamalgs/src/primitives/segmented_sort_in_place.cpp:121` | `buf.get_dev_scheduler_ptr()` |
| `sort_by_key_pow2_len` | `sort_by_key_pow2_len_impl`, `src/shamalgs/src/primitives/sort_by_key_pow2_len.cpp:97` | the `sched` argument |
| `sort_by_keys` | `sort_by_keys_impl`, `src/shamalgs/src/primitives/sort_by_keys.cpp:72` | `buf_key.get_dev_scheduler_ptr()` |
| `compute_histogram` | `compute_histogram_impl`, an `inline` global at `src/shamalgs/include/shamalgs/primitives/compute_histogram.hpp:56` | the `dev_sched` argument |
| `clbvh_dual_tree_traversal` | `dtt_impl`, `src/shamtree/src/CLBVHDualTreeTraversal.cpp:44` | the `dev_sched` argument |

Two naming notes:
- The DTT getter today is `get_current_impl_clbvh_dual_tree_traversal_impl`, with a stray `_impl`
  suffix. The registry name is `clbvh_dual_tree_traversal`.
- Each algorithm's per-algorithm functions are declared in its header's `namespace impl` block
  (e.g. `reduction.hpp:140-157`).

## Design

### Changes to `ImplVariant.hpp`

**`IImplVariant`** (lines 208-221) gains two pure virtuals:

```cpp
/// Whether an implementation has been selected yet
virtual bool is_set() const = 0;

/// Select the algorithm's default implementation for the device behind `sched`
virtual void autoselect(const sham::DeviceScheduler_ptr &sched) = 0;
```

- Add `#include "shambackends/DeviceScheduler.hpp"` for `sham::DeviceScheduler_ptr`.
  - There is no include cycle: shambackends never includes shamalgs, and shamalgs already links
    shambackends.
  - Every current includer of `ImplVariant.hpp` already pulls in SYCL, so the include adds no
    real cost.
- Update the file-level doc comment and the `IImplVariant` doc comment to mention the two new
  members and the registry.

**`ImplVariantGlobal<Alts...>`** (lines 244-281) implements them. The default selection moves
from a per-algorithm free function into a callback given at construction:

```cpp
/// Returns the default implementation to use on the device behind `sched`
using DefaultSelector = std::function<Variant(const sham::DeviceScheduler_ptr &)>;

explicit ImplVariantGlobal(DefaultSelector default_impl)
    : default_impl(std::move(default_impl)) {
    if (!this->default_impl) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "ImplVariantGlobal needs a default implementation selector");
    }
}

inline bool is_set() const override { return current.has_value(); }

inline void autoselect(const sham::DeviceScheduler_ptr &sched) override {
    set(default_impl(sched));
}

private:
DefaultSelector default_impl;
```

Rules for these changes:
- **`is_set()` becomes the `override`.** Its body is unchanged; it simply was not virtual before.
- **Mark every override `override`.** Otherwise clang warns `-Winconsistent-missing-override`.
- **No default constructor.** This forces every algorithm to supply its default.
- **Everything else in the class stays as it is**, including the public `set(Variant)`, which
  `autoselect` uses.
- **Copying and moving:** the registry stores an `IImplVariant *` to each global, so delete copy
  and move.

### New `src/shamalgs/include/shamalgs/impl_registry.hpp` + `src/shamalgs/src/impl_registry.cpp`

- The file name is lower_case, because it holds free functions (see the file-naming rule in
  AGENTS.md).
- Add `src/impl_registry.cpp` to the **explicit** `Sources` list in
  `src/shamalgs/CMakeLists.txt:12-48`, which has no glob.
- Everything goes in namespace `shamalgs::impl_registry`.

```cpp
/// Throws std::invalid_argument if `name` is already registered (nothing is stored then)
void register_impl(std::string name, IImplVariant &impl);

/// Lets a registration sit at namespace scope right after the global it registers
struct ImplRegistrar {
    ImplRegistrar(std::string name, IImplVariant &impl);
};

std::vector<std::string> get_registered_algs();                       // sorted
std::vector<std::string> get_default_impl_list(std::string_view alg); // impl.get_default_config_list()
std::string get_current_impl(std::string_view alg);                   // impl.get_current_config(), "null" when unset
bool is_impl_set(std::string_view alg);                               // impl.is_set()
void set_impl(std::string_view alg, std::string_view impl);           // logs, then impl.set(impl)
void autoselect_impl(std::string_view alg, const sham::DeviceScheduler_ptr &sched); // impl.autoselect(sched), then logs
```

**Storage**
- A function-local static singleton, accessed only from the `.cpp`. The Meyers-singleton pattern
  is already used at `src/shamsolvergraph/include/shamsolvergraph/JsonSerializable.hpp:140`.
- It holds `std::map<std::string, IImplVariant *, std::less<>>`, so lookups work by
  `string_view`.
- The singleton is constructed during the first registration, so it outlives every registered
  global. No unregister is needed.

**Errors**
- An unknown `alg` throws `std::invalid_argument` built with
  `shambase::make_except_with_loc`. The message lists the registered names.
- `autoselect_impl` checks the scheduler with `shambase::get_check_ref(sched)` before it calls
  `impl.autoselect(sched)`. This matters for Python, where the scheduler is null before
  `shamrock.sys.init()`.

**Logging**

The registry is the only place that logs selections. The per-algorithm log lines go away.
- `set_impl` logs `shamlog_info_ln("algs", "setting", alg, "implementation to impl :", impl)`.
- `autoselect_impl` logs `shamlog_info_ln("algs", "defaulting", alg, "implementation to impl :", impl.get_current_config())`.

**Lint**
- `std::move` any by-value parameters: `performance-unnecessary-value-param` is enabled.
- Every new file needs the license header and a doxygen `@file` block (pre-commit's
  `doxygen_header` hook).
- No non-ASCII characters in C++ files (pre-commit's `check_no_utf8` hook).

### Per-algorithm migration pattern (the `.cpp` algorithms)

Taking `reduction.cpp` as the example:

```cpp
namespace shamalgs::primitives::impl {

    /// Registry name, shared by the registrar and the dispatch site
    constexpr std::string_view reduction_impl_name = "reduction";

    using ReductionImpl = shamalgs::ImplVariantGlobal<
        Fallback
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        ,
        GroupReduction
#endif
        >;

    ReductionImpl reduction_impl{
        [](const sham::DeviceScheduler_ptr & /*sched*/) -> ReductionImpl::Variant {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
            return GroupReduction{};
#else
            return Fallback{};
#endif
        }};

    namespace {
        // Must come after reduction_impl: same TU, so it is initialized after the global
        shamalgs::impl_registry::ImplRegistrar reduction_registrar{
            std::string(reduction_impl_name), reduction_impl};
    } // namespace
} // namespace shamalgs::primitives::impl

// dispatch site: goes through the registry so that the "defaulting ..." log line is uniform
if (!impl::reduction_impl.is_set()) {
    shamalgs::impl_registry::autoselect_impl(impl::reduction_impl_name, sched);
}
```

The explicit `-> ReductionImpl::Variant` return type is needed, because the lambda returns
different alternatives on different `#ifdef` branches.

Steps for each algorithm:
1. **Header:** delete the 5 per-algorithm declarations from its `namespace impl` block. Delete
   the whole block if it ends up empty.
2. **`.cpp`:**
   - Delete all 5 per-algorithm functions.
   - Move the old `autoselect_impl_X` body, the `#ifdef`s and the compile-time choices unchanged
     into the constructor lambda. Change each `X_impl.set(Alt{})` into `return Alt{};`, and drop
     its log line, since the registry logs instead.
   - Add the name constant and the `ImplRegistrar` after the global.
3. **Dispatch site:**
   - Keep the direct `is_set()` / `get()` access on the typed global, which `std::visit` needs.
   - Call `shamalgs::impl_registry::autoselect_impl(<name constant>, <scheduler>)`, with the
     scheduler from the table above.
   - The registry lookup only happens on first use.

### `compute_histogram` (header-only dispatch): special case

Today the `inline` global in the header has a copy in every includer (`pyShamalgs.cpp`,
`shammodels/common/src/pyCommonUtils.cpp`, the test), and the dynamic linker merges them. If
self-registration ran from that header, every unmerged copy would register again, and the
duplicate would throw.

Changes:
- **Header:**
  - Keep the alternative structs.
  - Add `using ComputeHistogramImpl = shamalgs::ImplVariantGlobal<Reference, NaiveGpu, GpuTeamFetching, GpuOversubscribe>;`,
    `extern ComputeHistogramImpl compute_histogram_impl;`, and the
    `constexpr std::string_view compute_histogram_impl_name = "compute_histogram";`.
  - Delete the 5 inline functions.
- **Dispatch** (`compute_histogram.hpp:371-373`):

  ```cpp
  if (!impl::compute_histogram_impl.is_set()) {
      shamalgs::impl_registry::autoselect_impl(impl::compute_histogram_impl_name, dev_sched);
  }
  ```

- **New `src/shamalgs/src/primitives/compute_histogram.cpp`:**
  - Includes the header.
  - Defines `compute_histogram_impl` with a lambda holding the old autoselect logic:

    ```cpp
    return (sched->ctx->device->prop.type == sham::DeviceType::GPU)
               ? ComputeHistogramImpl::Variant{GpuOversubscribe{}}
               : ComputeHistogramImpl::Variant{NaiveGpu{}};
    ```

    The registry has already null-checked `sched`.
  - Holds the `ImplRegistrar`.
  - Add it to the `Sources` list in `src/shamalgs/CMakeLists.txt`.

### `clbvh_dual_tree_traversal` (shamtree)

Same pattern as the `.cpp` algorithms, in `src/shamtree/src/CLBVHDualTreeTraversal.cpp`.
- Delete the declarations at `src/shamtree/include/shamtree/CLBVHDualTreeTraversal.hpp:65-82`.
- shamtree already links shamalgs.

## Call sites to migrate (these must all go through the registry)

### Python bindings

In `src/shampylib/src/pyShamalgs.cpp`, delete the per-algorithm blocks:

| Algorithm | Lines | Note |
|---|---|---|
| `is_all_true` | 125-143 | |
| `reduction` | 172-190 | |
| `scan_exclusive_sum_in_place` | 211-229 | |
| `segmented_sort_in_place` | 259-269 | only binds 3 of the 5 functions today |
| `sort_by_keys` | 299-317 | |
| `sort_by_key_pow2_len` | 352-370 | |
| `compute_histogram` | 375-394 | |

In `src/shampylib/src/pyShamtree.cpp`, delete the DTT block at lines 93-111.

Add to `shamalgs_module` (`pyShamalgs.cpp:40`):
- `get_registered_algs()`
- `get_default_impl_list(alg)`
- `get_current_impl(alg)`
- `is_impl_set(alg)`
- `set_impl(alg, impl)`
- `autoselect_impl(alg)`

For each of them:
- Use `py::arg("alg")` / `py::arg("impl")` and `R"pbdoc(...)pbdoc"` docstrings.
- Only `autoselect_impl` fetches `shamsys::instance::get_compute_scheduler_ptr()`; the others must
  work before `sys.init()`.

Leave the unrelated legacy `impl_param` binding (`pyShamalgs.cpp:44-67`) alone.

### C++ tests

Replace `ns::impl::<fn>_<alg>(...)` with `shamalgs::impl_registry::<fn>("<alg>", ...)`. Autoselect
takes `shamsys::instance::get_compute_scheduler_ptr()`. The shape stays the same everywhere:
autoselect if unset, save, loop and set, then restore.

- `src/tests/shamalgs/primitives/reductionTests.cpp` (4 copies of the loop: 155-167, 303-315, 451-463, 656-668)
- `src/tests/shamalgs/primitives/is_all_trueTests.cpp:140-153`
- `src/tests/shamalgs/primitives/scan_exclusive_sum_in_placeTests.cpp:77-90`
- `src/tests/shamalgs/primitives/segmented_sort_in_placeTests.cpp:184-197`
- `src/tests/shamalgs/primitives/sort_by_keysTests.cpp:183-196`
- `src/tests/shamalgs/primitives/compute_histogram_tests.cpp` (autoselect at around 255-276; `set_impl_compute_histogram` at 82, 153, 227). Keep its reliance on `"reference"` being first in the list.
- `src/tests/shamalgs/algorithm/algorithmTests.cpp:25-38`. Include `shamsys/NodeInstance.hpp` directly; today it only arrives through `sortTests.hpp`.
- `src/tests/shamtree/DTTTesting_tests.cpp:410-442, 452-473`

### New test `src/tests/shamalgs/impl_registryTests.cpp`

`src/tests/CMakeLists.txt` uses a `GLOB_RECURSE`, so the file is picked up automatically. Add
`NEW_TEST(Unittest, "shamalgs/impl_registry", 1)`, checking that:
- `get_registered_algs()` **contains** all 8 names. Use a contains check, not an equality check.
- For every registered algorithm:
  - autoselect with the compute scheduler, then check `is_impl_set`;
  - save `get_current_impl`;
  - `set_impl` each entry of `get_default_impl_list` and read it back;
  - restore the saved value.

  Autoselect first, because a saved `"null"` cannot be restored.
- An unknown algorithm name throws `std::invalid_argument` from every function.
- Registering an existing name (`"reduction"`) throws. Build the dummy as a function-local
  `ImplVariantGlobal<A>` with a lambda default, where `A` is a file-scope tag struct with a
  `variant_type_name`. Wrap the call in a lambda, because `REQUIRE_EXCEPTION_THROW` is a macro and
  the commas would split its arguments:
  `REQUIRE_EXCEPTION_THROW(([&]{ shamalgs::impl_registry::register_impl("reduction", dummy); })(), std::invalid_argument)`.
- The `ImplVariantGlobal` constructor throws when given an empty `DefaultSelector`.
  The duplicate check must throw *before* anything is stored, so no dangling pointer is left.
- **Never register a test-local object under a new name.** There is no unregister, so the entry
  would dangle once the object goes out of scope.

### Benchmark scripts (`examples/benchmarks/`)

sphinx-gallery executes every `run_*.py` (`doc/sphinx/source/conf.py:71,80`), so a missed rename
breaks the docs CI. The change is mechanical: `shamrock.algs.<fn>_<alg>(x)` becomes
`shamrock.algs.<fn>("<alg>", x)`, and `shamrock.tree.*` DTT calls become `shamrock.algs.*`.

- `run_reduction_performance.py:100-117`
- `run_is_all_true_performance.py:107-125`
- `run_sort_by_keys_performance.py:106-125, 193-212`
- `run_exclusive_scan_in_place.py:85-103`
- `run_compute_histogram.py:34-83`
- `run_dtt_performance.py:233-253`
- `run_segmented_sort_in_place_performance.py:95-110`: also add the missing
  `if not is_impl_set(...): autoselect_impl(...)` before it reads the current implementation.

### Docs

Update `doc/sphinx/source/dev_doc/implementation_selection.md`:
- **Python section:** show the name-keyed API.
- **C++ skeleton:** the name constant, the global constructed with its default-selector lambda,
  an `ImplRegistrar` after it, and the dispatch site calling `impl_registry::autoselect_impl`.
  Also document the two new `IImplVariant` virtuals (`is_set`, `autoselect`).
- **"Wire it up end to end" list:** the header declares nothing now, and there are no Python
  bindings to add per algorithm.
- **Stale statements to fix:**
  - "three functions" at line 22;
  - the `AtomicEarlyExit{128, 256}` example, which is `{64, 256}` in the code;
  - lines 141-145, which say only 2 algorithms use the lazy default. All 8 do.

## Verification

1. **Build.** `cd build && ./shamenv_do shamconfigure`. The first run builds AdaptiveCpp and takes
   a few minutes. Then run `./shamenv_do shammake shamalgs shamtree shampylib && echo DONE`, and
   before testing a full `./shamenv_do shammake && echo DONE`. Check that `./shamrock` and
   `./shamrock_test` exist.
2. **Unit tests.**
   - Run `test -d reference-files || ./shamenv_do pull_reffiles`, then `./shamenv_do ./shamrock --smi`.
   - Show the device table and **ask the user which device to use** (only once).
   - Run `./shamenv_do ./shamrock_test --sycl-cfg X:X --loglevel 1 --unittest`.
   - The new `shamalgs/impl_registry` test and every migrated test must pass.
3. **Python.**
   - Run each of the 7 benchmark scripts with `./shamenv_do ./shamrock --sycl-cfg X:X --rscript <script>`.
     Temporarily shrink the sizes if they are slow.
   - A scratch script must show that:
     - `shamrock.algs.get_registered_algs()` lists all 8;
     - a set/get round-trip works;
     - `shamrock.algs.set_impl("nope", "{}")` raises a Python exception.
4. **Leftovers.** `git grep -nE '(get_default_impl_list|get_current_impl|is_impl_set|set_impl|autoselect_impl)_[a-z]'`
   must return nothing outside the removed code. Check `src/`, `examples/` and `doc/`.
5. **Lint.**
   - Run `SETUPTOOLS_USE_DISTUTILS=stdlib pre-commit run --files <changed files>`.
   - Run `.claude/tools/clang-tidy-check.py` on `impl_registry.cpp`, `compute_histogram.cpp` and
     one migrated algorithm `.cpp`.
6. **Commit.**
   - Follow AGENTS.md "Commit authorship": the author is the human, the only trailer is
     `Assisted-by: <agent>`, there are no model names, no `Co-authored-by` and no session link.
     Amend with `--no-verify`.
   - Push to the designated branch. Do not open a PR unless asked.
