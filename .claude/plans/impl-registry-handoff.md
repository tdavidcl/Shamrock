# Handoff: name-keyed registry for implementation selection

You are implementing **one** piece of a larger design: a registry that lets code and Python select
an algorithm's implementation by name. Read `AGENTS.md` and `CLAUDE.md` first: build, test, commit
authorship, branch naming (`claude/<type>/<short-kebab-description>`) and the no-session-link rules
all apply.

This document is self-contained: it covers the current code, the target design, and the later
work this change must stay compatible with. All line numbers refer to upstream `main` at
**`ca16e0e`**. Re-check them with `grep` before editing, because they may have drifted.

About the repository:
- Project: SHAMROCK, a C++20 / SYCL / MPI / pybind11 hydrodynamics code.
- Upstream: `Shamrock-code/Shamrock`, base branch `main`.
- Do not open a PR unless asked.

## Status: what is already merged upstream

Two preparatory PRs already landed. **Do not redo them.**

- **#2442 (`c7232a4`)**: every `autoselect_impl_<algo>` free function takes
  `const sham::DeviceScheduler_ptr &dev_sched`.
  - Dispatch sites forward their scheduler (or `buf.get_dev_scheduler_ptr()`).
  - The Python bindings and unit tests pass `shamsys::instance::get_compute_scheduler_ptr()`.
- **#2451 (`ce73d7b`)**: `IImplVariant` gains the pure virtuals `bool is_set() const` and
  `void autoselect(const sham::DeviceScheduler_ptr &)`.
  - `ImplVariantGlobal` now takes its default-selection rule at construction and runs it in
    `autoselect()`.
  - All 8 algorithms declare their default in that constructor lambda. The defaults themselves
    are unchanged.
  - See "Background" for the exact API.

What is **not** done yet, and is the scope of this handoff:
- There is no registry (`impl_registry.hpp/.cpp`, `ImplRegistrar`).
- The 5 per-algorithm free functions still exist in the headers, the `.cpp` files, the Python
  bindings, the tests, the benchmark scripts and the docs.
- There is no name-keyed Python API.
- `compute_histogram_impl` is still an `inline` global in a header.

## Goal

Each of the 8 algorithms still hand-writes the same 5 free functions around its
`shamalgs::ImplVariantGlobal<...>` global, plus 5 matching Python bindings. For `reduction`:

- `get_default_impl_list_reduction()`
- `get_current_impl_reduction()`
- `is_impl_set_reduction()`
- `set_impl_reduction(impl)`
- `autoselect_impl_reduction(dev_sched)`

Replace all of them with one registry keyed by algorithm name:

```python
shamrock.algs.get_registered_algs()
shamrock.algs.get_default_impl_list("reduction")
shamrock.algs.autoselect_impl("reduction")
shamrock.algs.is_impl_set("reduction")
shamrock.algs.set_impl("reduction", impl)
shamrock.algs.get_current_impl("reduction")
```

Every call site that uses a per-algorithm function switches to the registry: tests, Python
bindings, benchmark scripts and docs. The per-algorithm functions are then deleted. The registry
works purely through `IImplVariant`, which already has everything it needs. It never casts to a
concrete type, never parses JSON to find out whether an implementation is set, and needs no
per-algorithm callbacks.

## Hard constraints

1. **Do not modify `IImplVariant`** (`src/shamalgs/include/shamalgs/ImplVariant.hpp:211-230`).
   It is complete for this work.
2. **`ImplVariantGlobal` changes are limited to two things:**
   - delete its copy and move constructors and assignment operators, because the registry stores
     an `IImplVariant *` to each global;
   - optionally, throw `std::invalid_argument` in the constructor when the `AutoselectFn` is
     empty.

   Keep everything else as merged, including the `AutoselectFn(sched, self)` signature and the
   public `set(Variant)` the lambdas use.
3. The Python value types stay the same. One implementation is a JSON **string**
   (`{"implementation": ..., "parameters": ...}`), and the default list is a `list[str]`.

## Out of scope (do NOT implement), but stay compatible with it

These features are planned as later, separate changes on top of this registry. Do not implement
any of them, but do not make design choices that would block them.

| Later feature | What it will do | What this change must keep possible |
|---|---|---|
| Whole-config JSON export/import | `get_impl_config()` returns `{"device": {...}, "sycl": {...}, "config": {"<alg>": <impl config>, ...}}`, leaving out unset algorithms. `set_impl_config(json)` applies `config` entry by entry through `set_impl`, warning on and skipping unknown algorithms or implementations. | The registry can enumerate every algorithm and read and write each one by name. **Every change of selection from outside an algorithm goes through `impl_registry::set_impl`.** |
| Env var `SHAMROCK_IMPL_CONFIG` and CLI option `--impl-config <file>` | Load such a JSON file at the end of `shamsys::instance::init_sycl_mpi` (env var) and in `shamsys::instance::init(argc, argv)` (CLI). | The registry lives in shamalgs, which shamsys already links. It is fully populated by static initialization. |
| Hardware tuning | User-supplied `"tunings": [{"match": {"device": {...}, "sycl": {...}}, "config": {...}}]`. When an algorithm autoselects, the most specific entry matching the scheduler's device wins; an unusable entry falls back to the next one, and then to the hard-coded default. | This will be implemented **inside `impl_registry::autoselect_impl`**: try the tuning candidates through `set_impl`, else fall back to `impl.autoselect(sched)`. That is why **dispatch sites must call `impl_registry::autoselect_impl`**, not `X_impl.autoselect` directly. |
| Autotune hook | An optional per-algorithm autotuner, with "none" as the default, rolled out one algorithm at a time. | Nothing; just do not add it now. |
| Compiler-id move | Move the generated `shamrock_compiler_id_string` from shamlib down to shambackends, for the export's `sycl` block. | Nothing. |

## Background: current code (at `ca16e0e`)

### `src/shamalgs/include/shamalgs/ImplVariant.hpp`

This is a `std::variant`-based implementation selector.

- **Alternatives:** small structs with a `static constexpr std::string_view variant_type_name`,
  e.g. `struct Fallback { static constexpr std::string_view variant_type_name = "fallback"; };`.
- **Config string:** `{"implementation": "<variant_type_name>", "parameters": {...}}`.
  - `variant_to_config_string(v)` produces it.
  - `variant_from_config_string<Variant>(s)` parses it. It throws `std::invalid_argument` on an
    unknown name and nlohmann exceptions on malformed JSON.
- **`ImplVariantParams<Alt>`** (line 84): a trait an alternative with tunable fields specializes
  for its `"parameters"` `to_json`/`from_json`. The default serializes to `{}`.
  - Example: `GroupReduction{u32 group_size}` in `src/shamalgs/src/primitives/reduction.cpp:57-71`.
- **`HasCustomDefaults`** (line 111): an alternative may define
  `static std::vector<Alt> variant_custom_defaults()`, and is then listed once per returned
  instance in the default list, e.g. `GroupReduction{16}, {128}, {256}`.
- **`IImplVariant`** (lines 211-230):
  - `get_current_config()` returns the string `"null"` when unset;
  - `get_default_config_list()`;
  - `set(std::string_view)`;
  - `is_set()`;
  - `autoselect(const sham::DeviceScheduler_ptr &)`.
- **`ImplVariantGlobal<Alts...>`** (lines 265-314):

  ```cpp
  using Variant = std::variant<Alts...>;
  using AutoselectFn = std::function<void(const sham::DeviceScheduler_ptr &, ImplVariantGlobal &)>;
  explicit ImplVariantGlobal(AutoselectFn fn);                     // starts unset
  bool is_set() const override;
  void autoselect(const sham::DeviceScheduler_ptr &sched) override; // runs fn(sched, *this)
  const Variant &get() const;                                      // requires is_set()
  std::string get_current_config() const override;
  std::vector<std::string> get_default_config_list() const override;
  void set(Variant v);                                              // used by the lambdas
  void set(std::string_view config_json) override;                  // parses, then assigns
  ```

  - It already includes `shambackends/DeviceScheduler.hpp` and `<functional>`.
  - Copy and move are currently **not** deleted.

### The current per-algorithm pattern (what you are removing)

From `src/shamalgs/src/primitives/reduction.cpp:73-132`:

```cpp
namespace impl {
    shamalgs::ImplVariantGlobal<Fallback
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        , GroupReduction
#endif
        > reduction_impl{[](const sham::DeviceScheduler_ptr &, auto &self) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
            self.set(GroupReduction{});
#else
            self.set(Fallback{});
#endif
        }};

    std::vector<std::string> get_default_impl_list_reduction() { return reduction_impl.get_default_config_list(); }
    std::string get_current_impl_reduction() { return reduction_impl.get_current_config(); }
    bool is_impl_set_reduction() { return reduction_impl.is_set(); }
    void set_impl_reduction(const std::string &impl) {
        shamlog_info_ln("algs", "setting reduction implementation to impl :", impl);
        reduction_impl.set(impl);
    }
    void autoselect_impl_reduction(const sham::DeviceScheduler_ptr &dev_sched) {
        reduction_impl.autoselect(dev_sched);
        shamlog_info_ln("algs", "defaulting reduction implementation to impl :", get_current_impl_reduction());
    }
}

template<class T>
T sum(const sham::DeviceScheduler_ptr &sched, const sham::DeviceBuffer<T> &buf1, u32 start_id, u32 end_id) {
    if (!impl::reduction_impl.is_set()) {
        impl::autoselect_impl_reduction(sched);   // lazy default on first use
    }
    return std::visit(shambase::overloaded{ /* one lambda per alternative */ }, impl::reduction_impl.get());
}
```

- The 5 functions are declared in each header's `namespace impl` block.
- Some `sycl::buffer` entry points bypass the selector entirely: `is_all_true.cpp:268` and
  `sort_by_key_pow2_len.cpp:79`. Leave them alone.

### The 8 algorithms

| Registry name | Global and default lambda | Per-algorithm functions (`.cpp` / header) | Dispatch site(s) and scheduler |
|---|---|---|---|
| `reduction` | `reduction_impl`, `src/shamalgs/src/primitives/reduction.cpp:78-91`: `GroupReduction{}` if `SYCL2020_FEATURE_GROUP_REDUCTION`, else `Fallback{}` | `.cpp` 93-117, `reduction.hpp:140-157` | `:131, :158, :185`, `sched` |
| `is_all_true` | `is_all_true_impl`, `is_all_true.cpp:208-211`: `Host{}` | `.cpp` 213-237, `is_all_true.hpp:106-123` | `:245`, `buf.get_dev_scheduler_ptr()` |
| `scan_exclusive_sum_in_place` | `scan_exclusive_sum_in_place_impl`, `scan_exclusive_sum_in_place.cpp:139-168`: nested `#ifdef __MACH__` / `__ACPP__` / `SYCL2020_FEATURE_GROUP_REDUCTION` choice | `.cpp` 170-200, `scan_exclusive_sum_in_place.hpp:78-98` | `:220`, `buf1.get_dev_scheduler_ptr()` |
| `segmented_sort_in_place` | `segmented_sort_in_place_impl`, `segmented_sort_in_place.cpp:121-124`: `MultiStdSort{}` | `.cpp` 126-153, `segmented_sort_in_place.hpp:31-48` | `:170`, `buf.get_dev_scheduler_ptr()` |
| `sort_by_key_pow2_len` | `sort_by_key_pow2_len_impl`, `sort_by_key_pow2_len.cpp:97-100`: `BitonicSort{}` | `.cpp` 102-129, `sort_by_key_pow2_len.hpp:99-116` | `:182`, `sched` |
| `sort_by_keys` | `sort_by_keys_impl`, `sort_by_keys.cpp:72-75`: `StdSort{}` | `.cpp` 77-103, `sort_by_keys.hpp:54-71` | `:112`, `buf_key.get_dev_scheduler_ptr()` |
| `compute_histogram` | `compute_histogram_impl`, an **`inline` global** at `src/shamalgs/include/shamalgs/primitives/compute_histogram.hpp:56-63`: `GpuOversubscribe{}` if `dev_sched->ctx->device->prop.type == sham::DeviceType::GPU`, else `NaiveGpu{}`. It dereferences the scheduler with **no null check**. | inline, `compute_histogram.hpp:65-91` | `compute_histogram.hpp:374`, `dev_sched` |
| `clbvh_dual_tree_traversal` | `dtt_impl`, `src/shamtree/src/CLBVHDualTreeTraversal.cpp:44-47`: `ScanMultipass{}` | `.cpp` 49-75, `src/shamtree/include/shamtree/CLBVHDualTreeTraversal.hpp:66-83` | `:97`, `dev_sched` |

All the `primitives/*.cpp` files above are in `src/shamalgs/src/primitives/`, and their headers
are in `src/shamalgs/include/shamalgs/primitives/`.

Notes:
- The DTT getter is `get_current_impl_clbvh_dual_tree_traversal_impl`, with a stray `_impl`
  suffix. Its registry name is `clbvh_dual_tree_traversal`.
- `shamalgs::primitives::impl::StdSort` is defined identically in both `sort_by_keys.cpp` and
  `sort_by_key_pow2_len.cpp`. That predates this change; leave it.

### Build, linking and static-initialization facts relevant to self-registration

- **No static archives.** Every library is either `SHARED` (the default,
  `SHAMROCK_USE_SHARED_LIB=On`, `cmake/ShamrockBuildOptions.cmake:18`) or `OBJECT` (`Off`,
  forced on Apple and in coverage builds). There are no `STATIC` archives, so the linker never
  drops a translation unit holding a registrar. OBJECT libraries are linked directly into each
  final binary (`shamrock`, `shamrock_test`, the `pyshamrock` module), once each.
- **Link directions.**
  - shamalgs links only shambackends (`src/shamalgs/CMakeLists.txt`).
  - shamtree links shamalgs, shammath and shamsys.
  - shamsys links shamalgs.
  - So shamalgs **must not call shamsys**. Take the scheduler as a parameter instead.
- **Existing static-init registrations to model on:**
  - `ON_PYTHON_INIT` in `src/shambindings/include/shambindings/pybindaliases.hpp`;
  - the shamsolvergraph JSON registry, a Meyers singleton, in
    `src/shamsolvergraph/include/shamsolvergraph/JsonSerializable.hpp`.
- **Initialization order.**
  - Within one translation unit, namespace-scope objects are initialized in definition order.
    A registrar defined after its global therefore sees the global already constructed.
  - The registry singleton is created during the first registration, so it is destroyed after
    every global. There is a single atexit chain, even across shared libraries.
- **Why `compute_histogram` must move.** Its `inline` global has a copy in each includer:
  `src/shampylib/src/pyShamalgs.cpp`, `src/shammodels/common/src/pyCommonUtils.cpp` and the
  test. Today only the dynamic linker merges them. A self-registering copy that was not merged
  would register again and throw "duplicate".

### Getting a scheduler

- **Type:** `sham::DeviceScheduler_ptr` is `std::shared_ptr<sham::DeviceScheduler>`
  (`src/shambackends/include/shambackends/DeviceScheduler.hpp`).
- **Device properties:** `sched->ctx->device->prop`, a `sham::DeviceProperties`
  (`src/shambackends/include/shambackends/Device.hpp`).
- **From a buffer:** `sham::DeviceBuffer<T>::get_dev_scheduler_ptr()`, with both const and
  non-const overloads.
- **Process-wide compute scheduler:** `shamsys::instance::get_compute_scheduler_ptr()`, usable
  from shampylib and the tests but **not** from shamalgs. It returns a **null pointer** until
  `shamrock.sys.init(...)` (lib mode) or `--sycl-cfg` (executable) has initialized the devices.

### Python binding conventions

- **Module layout:**
  - The compiled module is `pyshamrock`, re-exported as `shamrock`.
  - `src/shampylib/src/pyShamalgs.cpp:40` creates the `algs` submodule inside an
    `ON_PYTHON_INIT` block, as `py::module shamalgs_module = m.def_submodule("algs", ...)`.
  - `src/pylib/shamrock/algs/__init__.py` re-exports everything automatically.
- **Style:**
  - Today's implementation bindings are bare lambdas.
  - New bindings should use `py::arg("name")` and an `R"pbdoc(...)pbdoc"` docstring as the last
    argument; see `src/shampylib/include/shampylib/pyNodeInstance.hpp`.
- **Type conversions:**
  - `std::vector<std::string>` converts to `list[str]` through `pybind11/stl.h`, which is
    already included via `shambindings/pybind11_stl.hpp`.
  - C++ exceptions surface as Python exceptions.
- **Dependencies:** `pyShamalgs.cpp` already includes `shamsys/NodeInstance.hpp`, and shampylib
  links shamsys.

### Test framework conventions (`src/shamtest/shamtest.hpp`)

- **Declaring a test:** `NEW_TEST(Unittest, "name", 1) { ... }`, where the last argument is the
  MPI node count. See `src/tests/shamalgs/primitives/reductionTests.cpp:21`.
- **Assertions:** `REQUIRE(cond)`, `REQUIRE_EQUAL(a, b)` and
  `REQUIRE_EXCEPTION_THROW(expr, ExceptionType)`. The last one is a macro, so wrap any
  expression that contains commas in a lambda.
- **Discovery:** `src/tests/CMakeLists.txt` collects `*.cpp` with a `GLOB_RECURSE`, so a new
  test file needs no CMake edit.

## Design

### `ImplVariant.hpp`: the one small change

In `ImplVariantGlobal` (line 265):

```cpp
ImplVariantGlobal(const ImplVariantGlobal &)            = delete;
ImplVariantGlobal &operator=(const ImplVariantGlobal &) = delete;
ImplVariantGlobal(ImplVariantGlobal &&)                 = delete;
ImplVariantGlobal &operator=(ImplVariantGlobal &&)      = delete;
```

Optionally, also make the constructor throw
`shambase::make_except_with_loc<std::invalid_argument>` when `!fn`. Then add a sentence to the
class doc comment saying instances are registered by name, and are therefore neither copyable
nor movable.

### New `src/shamalgs/include/shamalgs/impl_registry.hpp` + `src/shamalgs/src/impl_registry.cpp`

- **File name:** lower_case, because it holds free functions (AGENTS.md file-naming rule).
- **CMake:** add `src/impl_registry.cpp` to the **explicit** `Sources` list in
  `src/shamalgs/CMakeLists.txt`, which has no glob.
- **Namespace:** `shamalgs::impl_registry`.

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
void set_impl(std::string_view alg, std::string_view impl);           // impl.set(impl), then logs
void autoselect_impl(std::string_view alg, const sham::DeviceScheduler_ptr &sched); // impl.autoselect(sched), then logs
```

**Storage**
- A function-local static singleton, accessed only from the `.cpp`.
- It holds `std::map<std::string, IImplVariant *, std::less<>>`, so lookups by `string_view`
  work.
- No unregister is needed.

**Errors**
- An unknown `alg` throws `std::invalid_argument` built with
  `shambase::make_except_with_loc`, and the message lists the registered names.
- `autoselect_impl` checks the scheduler with `shambase::get_check_ref(sched)` before calling
  `impl.autoselect(sched)`. This matters for Python, where the scheduler is null before
  `shamrock.sys.init()`, and for `compute_histogram`, whose lambda dereferences the scheduler.

**Logging.** The registry becomes the only place that logs selections; the per-algorithm log
lines go away.
- `set_impl`: `shamlog_info_ln("algs", "setting", alg, "implementation to impl :", impl)`.
- `autoselect_impl`: `shamlog_info_ln("algs", "defaulting", alg, "implementation to impl :", impl.get_current_config())`.

**Lint**
- `std::move` by-value parameters, because `performance-unnecessary-value-param` is enabled.
- Every new file needs the license header and a doxygen `@file` block, for pre-commit's
  `doxygen_header` hook.
- No non-ASCII characters in C++ files, for pre-commit's `check_no_utf8` hook.

### Per-algorithm migration pattern (the `.cpp` algorithms)

Taking `reduction.cpp` as the example:

```cpp
namespace shamalgs::primitives::impl {

    /// Registry name, shared by the registrar and the dispatch sites
    constexpr std::string_view reduction_impl_name = "reduction";

    shamalgs::ImplVariantGlobal<
        Fallback
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        ,
        GroupReduction
#endif
        >
        reduction_impl{[](const sham::DeviceScheduler_ptr &, auto &self) {
            // merged default rule, unchanged
        }};

    namespace {
        // Must come after reduction_impl: same TU, so it is initialized after the global
        shamalgs::impl_registry::ImplRegistrar reduction_registrar{
            std::string(reduction_impl_name), reduction_impl};
    } // namespace
} // namespace shamalgs::primitives::impl

// dispatch site
if (!impl::reduction_impl.is_set()) {
    shamalgs::impl_registry::autoselect_impl(impl::reduction_impl_name, sched);
}
```

For each algorithm:
1. **Header:** delete the 5 per-algorithm declarations from its `namespace impl` block, and delete
   the block if it ends up empty.
2. **`.cpp`:**
   - Keep the global and its lambda exactly as merged.
   - Delete the 5 per-algorithm functions.
   - Add the name constant and the `ImplRegistrar` after the global.
3. **Dispatch site:**
   - Keep the direct `is_set()` / `get()` access on the typed global, which `std::visit` needs.
   - Replace `impl::autoselect_impl_X(s)` with
     `shamalgs::impl_registry::autoselect_impl(impl::X_impl_name, s)`.
   - The registry lookup only happens on first use.

### `compute_histogram` (header-only dispatch): special case

- **Header (`compute_histogram.hpp`):**
  - Keep the alternative structs.
  - Replace the `inline` global with:
    - `using ComputeHistogramImpl = shamalgs::ImplVariantGlobal<Reference, NaiveGpu, GpuTeamFetching, GpuOversubscribe>;`
    - `extern ComputeHistogramImpl compute_histogram_impl;`
    - `constexpr std::string_view compute_histogram_impl_name = "compute_histogram";`
  - Delete the 5 inline functions at lines 65-91.
- **Dispatch (line 374):** call
  `shamalgs::impl_registry::autoselect_impl(impl::compute_histogram_impl_name, dev_sched);`.
- **New `src/shamalgs/src/primitives/compute_histogram.cpp`:**
  - include the header;
  - define `ComputeHistogramImpl compute_histogram_impl{<the merged lambda, moved verbatim>};`;
  - add the `ImplRegistrar`;
  - add the file to the `Sources` list in `src/shamalgs/CMakeLists.txt`.

### `clbvh_dual_tree_traversal` (shamtree)

Use the same pattern as the `.cpp` algorithms, in `src/shamtree/src/CLBVHDualTreeTraversal.cpp`,
and delete the declarations at `CLBVHDualTreeTraversal.hpp:66-83`. shamtree already links
shamalgs.

## Call sites to migrate (these must all go through the registry)

### Python bindings

**`src/shampylib/src/pyShamalgs.cpp`**: delete the per-algorithm blocks. Every
`autoselect_impl_*` lambda among them currently passes
`shamsys::instance::get_compute_scheduler_ptr()`.

| Algorithm | Lines | Note |
|---|---|---|
| `is_all_true` | 125-144 | |
| `reduction` | 173-192 | |
| `scan_exclusive_sum_in_place` | 213-232 | |
| `segmented_sort_in_place` | 262-272 | Only 3 functions are bound today; `is_impl_set` and `autoselect_impl` are missing. |
| `sort_by_keys` | 302-321 | |
| `sort_by_key_pow2_len` | 356-375 | |
| `compute_histogram` | 380-399 | |

**`src/shampylib/src/pyShamtree.cpp`**: delete the DTT block at lines 93-112.

**Add to `shamalgs_module`** (`pyShamalgs.cpp:40`):
- `get_registered_algs()`
- `get_default_impl_list(alg)`
- `get_current_impl(alg)`
- `is_impl_set(alg)`
- `set_impl(alg, impl)`
- `autoselect_impl(alg)`

Give each one `py::arg("alg")` / `py::arg("impl")` and an `R"pbdoc(...)pbdoc"` docstring.
Only `autoselect_impl` fetches `shamsys::instance::get_compute_scheduler_ptr()`; the others must
work before `sys.init()`.

Leave the unrelated legacy `impl_param` binding (`pyShamalgs.cpp:44-67`) alone.

### C++ tests

Replace `ns::impl::<fn>_<alg>(...)` with `shamalgs::impl_registry::<fn>("<alg>", ...)`. The tests
already pass `shamsys::instance::get_compute_scheduler_ptr()` to autoselect. Keep the loop shape:
autoselect if unset, save the current implementation, loop over the list calling set, then
restore.

- `src/tests/shamalgs/primitives/reductionTests.cpp`: 4 copies of the loop, at 155-168,
  304-317, 453-466 and 659-672
- `src/tests/shamalgs/primitives/is_all_trueTests.cpp:140-154`
- `src/tests/shamalgs/primitives/scan_exclusive_sum_in_placeTests.cpp:77-91`
- `src/tests/shamalgs/primitives/segmented_sort_in_placeTests.cpp:184-198`
- `src/tests/shamalgs/primitives/sort_by_keysTests.cpp:183-197`
- `src/tests/shamalgs/primitives/compute_histogram_tests.cpp`:
  - `set_impl_compute_histogram` at 82, 153 and 227;
  - autoselect and the list at 261-275;
  - keep its reliance on `"reference"` being first in the list.
- `src/tests/shamalgs/algorithm/algorithmTests.cpp:26-40`
- `src/tests/shamtree/DTTTesting_tests.cpp:410-443, 453-475`

### New test `src/tests/shamalgs/impl_registryTests.cpp`

`NEW_TEST(Unittest, "shamalgs/impl_registry", 1)` covers:
- **Registration:** `get_registered_algs()` **contains** all 8 names. Check containment, not
  equality.
- **Round trip:** for every registered algorithm,
  1. autoselect with the compute scheduler, then check `is_impl_set`;
  2. save `get_current_impl`;
  3. `set_impl` each entry of `get_default_impl_list` and read it back;
  4. restore the saved value.

  Autoselect first, because a saved `"null"` cannot be restored.
- **Unknown names:** an unknown algorithm name throws `std::invalid_argument` from every
  function.
- **Null scheduler:** `autoselect_impl("reduction", nullptr)` throws.
- **Duplicate names:** registering an existing name (`"reduction"`) throws.
  - Build the dummy as a function-local `shamalgs::ImplVariantGlobal<A>` whose lambda is
    `self.set(A{})`, where `A` is a file-scope tag struct with a `variant_type_name`.
  - Wrap the call in a lambda so the macro's commas don't split it:
    `REQUIRE_EXCEPTION_THROW(([&]{ shamalgs::impl_registry::register_impl("reduction", dummy); })(), std::invalid_argument)`.
  - The duplicate check must throw *before* anything is stored, so no dangling pointer is left.
- **Never register a test-local object under a new name.** There is no unregister, so its entry
  would dangle once the object goes out of scope.

### Benchmark scripts (`examples/benchmarks/`)

sphinx-gallery executes every `run_*.py` (`doc/sphinx/source/conf.py`), so a missed rename breaks
the docs CI. The change is mechanical:
- `shamrock.algs.<fn>_<alg>(x)` becomes `shamrock.algs.<fn>("<alg>", x)`;
- `shamrock.tree.*` DTT calls become `shamrock.algs.*`.

Files to update:
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
- **"User side (Python)" (lines 20-82):** show the name-keyed API. Fix "three functions" at
  line 22, since the list shows five.
- **"Developer side (C++)" (lines 84-228):**
  - Keep the merged explanation of the `AutoselectFn` constructor lambda and the scheduler.
  - Add the name constant, the `ImplRegistrar`, and the dispatch through
    `impl_registry::autoselect_impl`.
- **"Wire it up end to end" list (around lines 215-228):** the header now declares nothing, and
  no Python binding is needed per algorithm.
- **`ImplVariant.hpp` doc comment:** it still says the class makes "get_default_impl_list_X /
  get_current_impl_X / set_impl_X free functions become one-liners". Replace that with a mention
  of the registry.

## Verification

1. **Build.**
   - `cd build && ./shamenv_do shamconfigure`. The first run builds AdaptiveCpp, which takes a
     few minutes.
   - `./shamenv_do shammake shamalgs shamtree shampylib && echo DONE`.
   - Before testing, the full `./shamenv_do shammake && echo DONE`, then check that `./shamrock`
     and `./shamrock_test` exist.
2. **Unit tests.**
   - `test -d reference-files || ./shamenv_do pull_reffiles`, then `./shamenv_do ./shamrock --smi`.
   - Show the device table and **ask the user which device to use** (once only).
   - `./shamenv_do ./shamrock_test --sycl-cfg X:X --loglevel 1 --unittest`.
   - The new `shamalgs/impl_registry` test and every migrated test must pass.
3. **Python.**
   - Run each of the 7 benchmark scripts with
     `./shamenv_do ./shamrock --sycl-cfg X:X --rscript <script>`. Shrink the sizes temporarily
     if they are slow.
   - Run a scratch script that:
     - checks `shamrock.algs.get_registered_algs()` lists all 8;
     - does a set/get round trip;
     - checks that `shamrock.algs.set_impl("nope", "{}")` raises a Python exception.
4. **Leftovers.** This must return nothing in `src/`, `examples/` or `doc/`:
   `git grep -nE '(get_default_impl_list|get_current_impl|is_impl_set|set_impl|autoselect_impl)_[a-z]'`.
5. **Lint.**
   - `SETUPTOOLS_USE_DISTUTILS=stdlib pre-commit run --files <changed files>`.
   - `.claude/tools/clang-tidy-check.py` on `impl_registry.cpp`, `compute_histogram.cpp` and one
     migrated algorithm `.cpp`.
6. **Commit.**
   - Follow AGENTS.md "Commit authorship": the author is the human, the only trailer is
     `Assisted-by: <agent>`, with no model names, no `Co-authored-by` and no session link.
     Amend with `--no-verify`.
   - Follow CLAUDE.md's branch-naming rule. Do not open a PR unless asked.
