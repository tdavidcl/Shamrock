# Handoff: name-keyed registry for implementation selection

You are implementing **one** piece of a larger design: a registry that lets code and Python select
an algorithm's implementation by name. Read `AGENTS.md` and `CLAUDE.md` first: build, test, commit
authorship and the no-session-link rules all apply.

The full long-term design is in `.claude/plans/impl-variant-registry.md`. Read it for context
only: **most of it is out of scope here** (see "Out of scope" below).

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

## Hard constraints

1. **Do not modify `IImplVariant`** (`src/shamalgs/include/shamalgs/ImplVariant.hpp:208-221`).
   Its interface stays exactly:
   - `get_current_config() const`: the config JSON string, or the string `"null"` when unset;
   - `get_default_config_list() const`;
   - `set(std::string_view)`.

   Do not modify `ImplVariantGlobal` either; the whole of `ImplVariant.hpp` stays unchanged.
2. The registry must work through that interface alone. What `IImplVariant` lacks is filled in
   as follows:
   - **is_set:** derive it as `!nlohmann::json::parse(impl.get_current_config()).is_null()`.
   - **autoselect:** there is no virtual for it, so each algorithm registers its autoselect
     function next to its `IImplVariant &`.
3. The Python value types stay the same: one implementation is a JSON **string**
   (`{"implementation": ..., "parameters": ...}`), and the default list is a `list[str]`.

## Out of scope (do NOT implement)

- JSON export or import of the whole config (`get_impl_config` / `set_impl_config`).
- Hardware tuning entries and device matching.
- The autotune hook.
- The `SHAMROCK_IMPL_CONFIG` env var and the `--impl-config` CLI option.
- Moving `shamrock_compiler_id_string` into shambackends.
- Any change to `ImplVariant.hpp`, including `get_or_autoselect` and a name in the constructor.

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

### New `src/shamalgs/include/shamalgs/impl_registry.hpp` + `src/shamalgs/src/impl_registry.cpp`

- The file name is lower_case, because it holds free functions (see the file-naming rule in
  AGENTS.md).
- Add `src/impl_registry.cpp` to the **explicit** `Sources` list in
  `src/shamalgs/CMakeLists.txt:12-48`, which has no glob.
- Everything goes in namespace `shamalgs::impl_registry`.

```cpp
using autoselect_fct = std::function<void(const sham::DeviceScheduler_ptr &)>;

/// Throws std::invalid_argument if `name` is already registered
void register_impl(std::string name, IImplVariant &impl, autoselect_fct autoselect);

/// RAII-free helper so a registration can sit at namespace scope right after the global
struct ImplRegistrar {
    ImplRegistrar(std::string name, IImplVariant &impl, autoselect_fct autoselect);
};

std::vector<std::string> get_registered_algs();                       // sorted
std::vector<std::string> get_default_impl_list(std::string_view alg);
std::string get_current_impl(std::string_view alg);                    // "null" when unset
bool is_impl_set(std::string_view alg);
void set_impl(std::string_view alg, std::string_view impl);            // logs, then impl.set()
void autoselect_impl(std::string_view alg, const sham::DeviceScheduler_ptr &sched);
```

**Storage**
- A function-local static singleton, accessed only from the `.cpp`. The Meyers-singleton pattern
  is already used at `src/shamsolvergraph/include/shamsolvergraph/JsonSerializable.hpp:140`.
- It holds `std::map<std::string, Entry, std::less<>>`, where
  `struct Entry { IImplVariant *impl; autoselect_fct autoselect; };`.
- The singleton is constructed during the first registration, so it outlives every registered
  global. No unregister is needed.

**Errors**
- An unknown `alg` throws `std::invalid_argument` built with
  `shambase::make_except_with_loc`. The message lists the registered names.
- `autoselect_impl` checks the scheduler with `shambase::get_check_ref(sched)` before it calls the
  function. This matters for Python, where the scheduler is null before `shamrock.sys.init()`.

**Logging**
- `set_impl` logs `shamlog_info_ln("algs", "setting", alg, "implementation to impl :", impl)`,
  replacing the per-algorithm log lines.
- The per-algorithm autoselect functions keep their existing "defaulting ..." log lines.

**Lint**
- Use designated initializers when building an `Entry`: clang-tidy enables
  `modernize-use-designated-initializers`.
- `std::move` any by-value parameters: `performance-unnecessary-value-param` is enabled.
- Every new file needs the license header and a doxygen `@file` block (pre-commit's
  `doxygen_header` hook).
- No non-ASCII characters in C++ files (pre-commit's `check_no_utf8` hook).

### Per-algorithm migration pattern (the `.cpp` algorithms)

Taking `reduction.cpp` as the example:

```cpp
namespace shamalgs::primitives::impl {
    shamalgs::ImplVariantGlobal<Fallback /*, GroupReduction under #ifdef*/> reduction_impl;

    namespace {
        /// Select the default implementation for reduction
        void autoselect_impl_reduction(const sham::DeviceScheduler_ptr & /*sched*/) {
            // body unchanged (the #ifdef choice + the "defaulting ..." log line)
        }

        // Must come after reduction_impl: same TU, so it is initialized after the global
        shamalgs::impl_registry::ImplRegistrar reduction_registrar{
            "reduction", reduction_impl, autoselect_impl_reduction};
    } // namespace
}

// dispatch site (unchanged shape, just passes the scheduler):
if (!impl::reduction_impl.is_set()) {
    impl::autoselect_impl_reduction(sched);
}
```

Steps for each algorithm:
1. **Header:** delete the 5 per-algorithm declarations from its `namespace impl` block. Delete
   the whole block if it ends up empty.
2. **`.cpp`:**
   - Delete `get_default_impl_list_X`, `get_current_impl_X`, `is_impl_set_X` and `set_impl_X`.
   - Move `autoselect_impl_X` into an anonymous namespace with the signature
     `(const sham::DeviceScheduler_ptr &)`. Keep its body unchanged: the `#ifdef`s, the
     compile-time choices and the log line.
   - Add the `ImplRegistrar` after the global.
3. **Dispatch site:** keep the direct `is_set()` / `get()` access on the typed global, which
   `std::visit` needs. Pass the scheduler from the table above to the local autoselect function.

### `compute_histogram` (header-only dispatch): special case

Today the `inline` global in the header has a copy in every includer (`pyShamalgs.cpp`,
`shammodels/common/src/pyCommonUtils.cpp`, the test), and the dynamic linker merges them. If
self-registration ran from that header, every unmerged copy would register again, and the
duplicate would throw.

Changes:
- **Header:**
  - Keep the alternative structs.
  - Add `using ComputeHistogramImpl = shamalgs::ImplVariantGlobal<Reference, NaiveGpu, GpuTeamFetching, GpuOversubscribe>;`
    and `extern ComputeHistogramImpl compute_histogram_impl;`.
  - Delete the 5 inline functions.
- **Dispatch** (`compute_histogram.hpp:371-373`):

  ```cpp
  if (!impl::compute_histogram_impl.is_set()) {
      shamalgs::impl_registry::autoselect_impl("compute_histogram", dev_sched);
  }
  ```

- **New `src/shamalgs/src/primitives/compute_histogram.cpp`:**
  - Includes the header.
  - Defines `compute_histogram_impl`.
  - Holds the autoselect function in an anonymous namespace, with its body unchanged (its
    `prop.type == GPU` check now reads the lambda argument).
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
- Registering an existing name (`"reduction"`) with a dummy throws. Wrap the call in a lambda,
  because `REQUIRE_EXCEPTION_THROW` is a macro and the commas would split its arguments:
  `REQUIRE_EXCEPTION_THROW(([&]{ ...register_impl("reduction", dummy, f); })(), std::invalid_argument)`.
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
- **C++ skeleton:** the global, the anonymous-namespace autoselect taking the scheduler, and an
  `ImplRegistrar` after the global.
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
