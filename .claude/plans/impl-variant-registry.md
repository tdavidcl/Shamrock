# Plan: registry for implementation choices, with JSON config/tuning and autotune hooks

## Context

Each of the 8 algorithms with several implementations wraps a `shamalgs::ImplVariantGlobal<Alts...>`, and each one hand-writes the same 5 free functions (`get_default_impl_list_X`, `get_current_impl_X`, `is_impl_set_X`, `set_impl_X`, `autoselect_impl_X`). Each also gets 5 matching Python bindings and a test loop that calls these functions by name.

The 8 algorithms are `reduction`, `is_all_true`, `scan_exclusive_sum_in_place`, `segmented_sort_in_place`, `sort_by_key_pow2_len`, `sort_by_keys`, `compute_histogram` and `clbvh_dual_tree_traversal`.

`IImplVariant` already exists, but nothing collects the instances. This change adds:

- **A registry** that every `ImplVariantGlobal` joins itself, under a name.
- **A single name-keyed API** in C++ and in Python (`shamrock.algs.set_impl("reduction", impl)`, ...), which replaces all the per-algorithm functions.
- **Export and import** of the whole implementation config as JSON: from Python, from `SHAMROCK_IMPL_CONFIG`, and from `--impl-config`.
- **Hardware-specific tuning**, supplied by the user and matched against the device at autoselect.
- **An autotune hook** on `IImplVariant`. It defaults to "no autotuner", so autotuners can be added one algorithm at a time later.

Decisions made with the user:

- **Old API:** remove the per-algorithm functions and bindings, and migrate the tests, benchmarks and docs.
- **Python location:** `shamrock.algs`.
- **Python types:** one implementation is a JSON string; the full config is a dict.
- **Every mutation goes through `set_impl`**, i.e. `IImplVariant::set(string)`.
- **Export** leaves out unset algorithms. It includes a `device` block and a `sycl` block (the implementation name and the compiler id).
- **Tuning** is supplied by the user only. It is a list of device-matched entries, applied at the **autoselect** step.
- **Match priority** is most specific first. An unknown algorithm or implementation gives a warning, is skipped, and the next matching entry is tried. If nothing usable matches, the hard-coded default is used.
- **One file schema** is shared by `SHAMROCK_IMPL_CONFIG`, `--impl-config` and Python: `config` and/or `tunings`.
- **Autotune:** the hook is on `IImplVariant`, and the default is none.

## JSON schema (the same everywhere)

```json
{
  "device":  {"backend": "CUDA", "type": "GPU", "name": "NVIDIA A100-SXM4-40GB", "platform": "..."},
  "sycl":    {"implementation": "AdaptiveCpp", "compiler_id": "<--version output>"},
  "config":  {"reduction": {"implementation": "group_reduction", "parameters": {"group_size": 256}}},
  "tunings": [
    {"name": "optional label",
     "match":  {"device": {"backend": "CUDA", "name": "A100"}, "sycl": {"implementation": "AdaptiveCpp"}},
     "config": {"reduction": {...}, "sort_by_keys": {...}}}
  ]
}
```

- **Every top-level key is optional.**
  - The export writes `device`, `sycl` and `config`.
  - Import ignores `device` and `sycl`, applies `config` immediately through `set_impl`, and adds `tunings` to the tuning DB. The tunings are added first.
- **Match semantics** use the same nesting as the export, so an exported `device`/`sycl` block can be pasted into `match` and trimmed.
  - `device.backend`, `device.type` and `sycl.implementation` must match exactly.
  - `device.name`, `device.platform` and `sycl.compiler_id` match as substrings.
  - An unknown match key throws when the tuning is loaded, so typos can't silently fail to match.
  - Specificity is the number of leaf keys; `{}` matches every device.
- **Priority at autoselect**, per algorithm: matching entries that contain the algorithm, ordered by specificity (highest first), then by load order (later first). The first config that `set()` accepts wins. If `set()` rejects one (invalid implementation or JSON), log a warning and try the next. If none is accepted, use the hard-coded default.
- **`vendor` is left out**: `Device.cpp:354` always reports it as `UNKNOWN`.

## C++ design

### `src/shamalgs/include/shamalgs/ImplVariant.hpp` (edit)

**New includes:** `shambackends/DeviceScheduler.hpp` and `shamcomm/logs.hpp`.
- There is no cycle: shambackends never includes shamalgs.
- Every current includer already pulls in SYCL.

**Hooks declared here, not in `impl_registry.hpp`.** Right after `IImplVariant`, declare the few out-of-line functions the template needs, in `namespace shamalgs::impl_registry`. This breaks the circular dependency.
- `register_impl(IImplVariant &)` and `unregister_impl(IImplVariant &)`.
- `std::vector<std::pair<std::string, std::string>> get_tuning_candidates(std::string_view alg, const sham::DeviceScheduler_ptr &)`, returning `{entry name, config json}` pairs in priority order.
  - The device read and the null check (`shambase::get_check_ref`) happen inside `impl_registry.cpp`.
  - The template therefore never dereferences the scheduler.

**`IImplVariant` gains** these pure virtuals, and every override in `ImplVariantGlobal` is marked `override` (this includes `is_set`, which is not virtual today):
- `std::string_view get_name() const`
- `bool is_set() const`
- `void autoselect(const sham::DeviceScheduler_ptr &)`
- `bool has_autotune() const`
- `void autotune(const sham::DeviceScheduler_ptr &)`

It keeps `get_current_config`, `get_default_config_list` and `set(string_view)`.

**`ImplVariantGlobal<Alts...>`:**
- **Constructor:** `ImplVariantGlobal(std::string name, DefaultSelector default_impl, Autotuner autotuner = {})`. Both callbacks are `std::function<Variant(const sham::DeviceScheduler_ptr &)>`, and the parameters are moved into the members.
  - Throw if `default_impl` is empty.
  - Call `register_impl(*this)` as the **last** statement of the body. Use `this->name` or `get_name()` there, never the moved-from parameter.
  - The destructor calls `unregister_impl(*this)`. It only erases the entry if the entry points to `this`.
  - Copy and move are deleted, because the registry holds a pointer.
- **`set(string_view)` is the only mutator.**
  - Order: parse (`variant_from_config_string`), then log `"setting <name> implementation to impl : ..."`, then assign.
  - Parsing first means a rejected candidate never prints a misleading line, and the old value is kept.
  - The public `set(Variant)` is removed; nothing outside the 8 algorithm files calls it.
  - The default selector and the autotuner go through a private `set_variant(const Variant &)`, which serializes with `variant_to_config_string` and then calls `set(string)`. Every write therefore goes through `set_impl`.
- **`autoselect(sched)`**
  - Tries each `get_tuning_candidates(name, sched)` entry in turn with `set()`, inside `try { ... } catch (const std::exception &e)`. It must catch `std::exception`, because nlohmann throws `parse_error`, `type_error` and `out_of_range`, not only `invalid_argument`. On a failure it warns and continues.
  - If no candidate is accepted, it calls `set_variant(default_impl(sched))`.
  - It logs where the choice came from: a tuning entry's name, or the default.
- **`autotune(sched)`** throws if there is no autotuner. The registry function checks `has_autotune()` first.
- **New dispatch helper:** `const Variant &get_or_autoselect(const sham::DeviceScheduler_ptr &sched)`, which autoselects when unset. It replaces the `if (!is_set()) autoselect...; get()` boilerplate at every dispatch site. `get()` stays.

### New file `src/shamalgs/include/shamalgs/impl_registry.hpp` + `src/shamalgs/src/impl_registry.cpp`

The header is named lower_case because it is a bag of free functions. It includes `ImplVariant.hpp` and declares the rest of `namespace shamalgs::impl_registry`. Add `src/impl_registry.cpp` to the explicit `Sources` list in `src/shamalgs/CMakeLists.txt:12-48`.

**Storage:** a function-local static singleton (the Meyers singleton pattern, as in `shamsolvergraph/JsonSerializable.hpp:140`). Its accessor stays out-of-line, in the `.cpp`. It holds:
- `std::map<std::string, IImplVariant*, std::less<>>`, so lookups work by `string_view`;
- `std::vector<TuningEntry>`, where `TuningEntry` has the fields `name`, `match`, `config` and `load_index`. Construct it with designated initializers, because clang-tidy enables `modernize-use-designated-initializers`.

The singleton finishes construction inside the first global's constructor. It is therefore destroyed after every global, across all shared libraries.

**Registration:**
- `register_impl` throws on a duplicate name.
- `unregister_impl` is the counterpart.

**Scheduler access:** every function that takes a scheduler checks it with `shambase::get_check_ref`. A null scheduler, e.g. before `shamrock.sys.init()`, then raises a clear error instead of a segfault.

**Per-algorithm functions** (an unknown `alg` throws `std::invalid_argument` listing the registered names):
- `get_registered_algs()`
- `get_default_impl_list(alg)`
- `get_current_impl(alg)`
- `is_impl_set(alg)`
- `set_impl(alg, impl)`
- `autoselect_impl(alg, sched)`
- `has_autotune(alg)`
- `autotune_impl(alg, sched)`: returns `false` and logs that no autotuner is implemented when there is none.

**Whole-registry functions:**
- `nlohmann::json get_impl_config(const sham::DeviceScheduler_ptr &sched)`: the `device` block, the `sycl` block, and `config` restricted to algorithms that are set.
- `void set_impl_config(const nlohmann::json &)`: the unified import described above.
  - `config`: an unknown algorithm, or any `std::exception` from `set`, gives a warning and a skip; the rest is still applied.
  - `tunings`: validated in full before any entry is added, so a schema error (an unknown match key, or the wrong types) throws with nothing loaded. An unknown algorithm name only gives a warning when loaded.
- `void load_impl_config_file(const std::string &path)`: reads the file, then calls `set_impl_config`.
- `nlohmann::json get_impl_tunings()` and `void clear_impl_tunings()`.
- `get_tuning_candidates`: declared in `ImplVariant.hpp` and defined here.

**Exact strings to document for matching:** `backend_name` returns `"CUDA"`, `"ROCm"`, `"OpenMP"` or `"Unknown"`. `device_type_name` returns `"CPU"`, `"GPU"` or `"UNKNOWN"` (`Device.hpp:54-76`).

**Where the `sycl` block comes from:**
- The implementation name comes from `sycl_implementation` in `shambackends/sycl.hpp:24-34`, mapped by a local helper to `"AdaptiveCpp"`, `"DPC++"` or `"Unknown"`.
- `compiler_id` comes from `shamrock_compiler_id_string`, which moves down into shambackends (next section).
- The `device` block uses `sham::backend_name`, `device_type_name`, `prop.name` and `prop.platform` (`shambackends/Device.hpp`).

### Move the compiler id string down to shambackends

- `shamrock_compiler_id_string` is generated today by `src/shamrock/CMakeLists.txt:37-54,64,70`, in shamlib, which shamalgs cannot link.
- Move that block, including the `FATAL_ERROR` check, into `src/shambackends/CMakeLists.txt`, and add the generated `compiler_id.cpp` to its sources.
- Remove `compiler_id.cpp` from shamlib's `add_library` calls (lines 64 and 70).
- `SHAMROCK_COMPILER_ID_STRING` is already set at top-level scope (`ShamConfigureSYCL.cmake:39-44`) before `add_subdirectory(src)`.
- **Avoid rebuilds on every configure.** `file(WRITE)` rewrites the file on each configure, which would force a relink of `libshambackends` and everything downstream. Write to `compiler_id.cpp.tmp`, then `configure_file(... COPYONLY)`, which copies only when the content changed.
- Add `shambackends/include/shambackends/sycl_compiler_id.hpp`, declaring `extern const char *shamrock_compiler_id_string;`.
- The generated `.cpp` `#include`s that header, so the declaration and the definition are type-checked against each other.
- Make `src/shamrock/include/shamrock/version.hpp:35-36` include that header instead of re-declaring the symbol. Its only user (`pyShamsys.cpp:85`) is unaffected.

### Migrate the 8 algorithms (same pattern each time)

The pattern, using `reduction.cpp` as the example:

```cpp
using ReductionImpl = shamalgs::ImplVariantGlobal<Fallback /*, GroupReduction under #ifdef*/>;
ReductionImpl reduction_impl{"reduction", [](const sham::DeviceScheduler_ptr &) -> ReductionImpl::Variant {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
    return GroupReduction{};
#else
    return Fallback{};
#endif
}};
// dispatch: std::visit(overloaded{...}, impl::reduction_impl.get_or_autoselect(sched));
```

- **Delete the 5 free functions** from every header's `namespace impl` and from every `.cpp`.
- **Move each `autoselect_impl_X` body into its default-selector lambda unchanged**, `#ifdef`s included. `compute_histogram` keeps its `prop.type == GPU` check, now reading it from the lambda's `sched` argument.
- **Get the scheduler at the dispatch sites** from the argument the site already has. For `is_all_true` (`is_all_true.cpp:239`) and the other buffer-only entry points, use `buf.get_dev_scheduler_ptr()`.
- **Algorithm names:** keep the current names. The DTT one is `"clbvh_dual_tree_traversal"`, which drops the stray `_impl` suffix from `get_current_impl_clbvh_dual_tree_traversal_impl`.

The files, grouped by action:

| Action | Files |
|---|---|
| Edit the header and `.cpp` | `reduction`, `is_all_true`, `scan_exclusive_sum_in_place`, `segmented_sort_in_place`, `sort_by_key_pow2_len`, `sort_by_keys` (headers in `src/shamalgs/include/shamalgs/primitives/`, sources in `src/shamalgs/src/primitives/`); `src/shamtree/include/shamtree/CLBVHDualTreeTraversal.hpp`, `src/shamtree/src/CLBVHDualTreeTraversal.cpp` |
| Change the `inline` global to a type alias plus an `extern` declaration | `compute_histogram.hpp` |
| Create, to define the `compute_histogram` global | new `src/shamalgs/src/primitives/compute_histogram.cpp`, which includes the header and is added to the `Sources` list in `src/shamalgs/CMakeLists.txt` |

`compute_histogram` moves to a `.cpp` so that it registers exactly once. Today its `inline` global has copies in `libshampylib`, `libshammodels_common` and `shamrock_test` that are merged only by the dynamic linker. With self-registration, any copy that did not merge would throw "duplicate".

Where each dispatch site gets its scheduler (all 8 checked):

| Algorithm | Scheduler source |
|---|---|
| `reduction`, `sort_by_key_pow2_len` | the `sched` argument |
| `compute_histogram`, `clbvh_dual_tree_traversal` | the `dev_sched` argument |
| `is_all_true`, `scan_exclusive_sum_in_place`, `segmented_sort_in_place`, `sort_by_keys` | `buf.get_dev_scheduler_ptr()` |

### Env var and CLI (in shamsys, which already links shamalgs and shamcmdopt)

- **`SHAMROCK_IMPL_CONFIG`:**
  - At namespace scope in `src/shamsys/src/NodeInstance.cpp`, register only its documentation, so it appears in `--help`.
  - Read its value with `shamcmdopt::getenv_str` *inside* `init_sycl_mpi` (`NodeInstance.cpp:300-314`), then call `load_impl_config_file`. Reading it at library load time would miss an `os.environ[...]` set after `import shamrock`.
  - This one path covers the executable, the tests and `shamrock.sys.init()` in lib mode.
  - Nothing inside `init_sycl_mpi` runs one of the algorithms, so no autoselect can happen before the tunings are loaded.
  - Document that it overrides any `set_impl` done in Python before `sys.init()`.
- **`--impl-config (filepath)`:**
  - Registered in `src/main.cpp` (around lines 54-75) and in `src/main_test.cpp` (around lines 47-77).
  - Applied in `shamsys::instance::init(argc, argv)` (`NodeInstance.cpp:316`) after `init_sycl_mpi`, so it runs after the env var and wins over it. It cannot go in `init_sycl_mpi`, because cmdopt is uninitialized in lib mode.
  - Throw if the path is empty: `get_option` returns `""` when the flag is the last argument.
  - `init(argc, argv)` only runs when `--sycl-cfg` is given (`main.cpp:114`, `main_test.cpp:124`). Warn in `main` when `--impl-config` is given without `--sycl-cfg`.
- Both paths go through `set_impl_config`, and so through `set_impl`.

## Python (`src/shampylib/src/pyShamalgs.cpp`, `pyShamtree.cpp`)

**Remove:**
- the 7 per-algorithm blocks in `pyShamalgs.cpp` (lines 125-143, 172-190, 211-229, 259-269, 299-317, 352-370, 375-394);
- the DTT block in `pyShamtree.cpp:93-111`.

**Add to `shamrock.algs`**, with `R"pbdoc(...)pbdoc"` docstrings and `py::arg` names.
- **Only** `autoselect_impl`, `autotune_impl` and `get_impl_config` fetch `shamsys::instance::get_compute_scheduler_ptr()`. That pointer is null before `sys.init()`, and the registry's `get_check_ref` turns a null into a clean error.
- Every other function works without a device, so it can be called before init.

Per algorithm:
- `get_registered_algs() -> list[str]`
- `get_default_impl_list(alg) -> list[str]`
- `get_current_impl(alg) -> str`
- `is_impl_set(alg) -> bool`
- `set_impl(alg, impl: str)`
- `autoselect_impl(alg)`
- `has_autotune(alg) -> bool`
- `autotune_impl(alg) -> bool`

Whole registry:
- `get_impl_config() -> dict`
- `set_impl_config(cfg: dict)`
- `load_impl_config(path: str)`
- `get_impl_tunings() -> list[dict]`
- `clear_impl_tunings()`

**Dict conversion** uses `json.loads`/`json.dumps` through `py::module_::import("json")`, the same pattern as `shammodels/common/.../shamrock_json_to_py_json.hpp:25-54`, written inline as in `pyUnits.cpp:83-97`.

The unused legacy `impl_param` binding (`pyShamalgs.cpp:44-67`) is left alone.

## Tests, scripts, docs

- **Migrate the C++ tests to `shamalgs::impl_registry::*("<alg>")`**, with autoselect taking `shamsys::instance::get_compute_scheduler_ptr()`. The save/loop/restore shape stays the same. Files, in `src/tests/shamalgs/primitives/`:
  - `reductionTests.cpp` (4 copies of the loop)
  - `is_all_trueTests.cpp`
  - `scan_exclusive_sum_in_placeTests.cpp`
  - `segmented_sort_in_placeTests.cpp`
  - `sort_by_keysTests.cpp`
  - `compute_histogram_tests.cpp`

  Also `src/tests/shamalgs/algorithm/algorithmTests.cpp` and `src/tests/shamtree/DTTTesting_tests.cpp`.
- **`algorithmTests.cpp`:** include `shamsys/NodeInstance.hpp` directly. Today it only arrives through `sortTests.hpp`.
- **New `src/tests/shamalgs/impl_registryTests.cpp`** (`NEW_TEST(Unittest, "shamalgs/impl_registry", 1)`). `src/tests/CMakeLists.txt` uses a `GLOB_RECURSE`, so the file is picked up automatically. It checks that:
  - All 8 names are registered. This is a *contains* check, not an equality check.
  - `set_impl`/`get_current_impl` round-trip for each algorithm. Each algorithm is autoselected before its current value is saved, because a saved `"null"` cannot be restored.
  - An unknown algorithm throws.
  - A duplicate registration throws. Wrap the call in a lambda so the macro sees one argument: `REQUIRE_EXCEPTION_THROW(([&]{ Dummy d("reduction", f); })(), std::invalid_argument)`.

  It also constructs a test-local `ImplVariantGlobal<A, B, C{param}>{"test_impl_registry_dummy", ...}`, whose scope takes care of registering and unregistering it. The alternatives and the `ImplVariantParams<C>` specialization are at file scope. The test checks that:
  - The export leaves out unset algorithms.
  - `set_impl_config` warns and skips an unknown algorithm or implementation while still applying the rest.
  - Specificity layering works: entries matched on the real device's backend and name, where the most specific one holds an invalid implementation, fall through to the next.
  - No match leads to the default.
  - An unknown match key throws.
  - `has_autotune` is `false` and `autotune_impl` returns `false`; with an autotuner, its result is applied.

  The tuning DB is saved and restored with `get_impl_tunings()`/`clear_impl_tunings()`.
- **Benchmark scripts** in `examples/benchmarks/`:
  - `run_reduction_performance.py`
  - `run_is_all_true_performance.py`
  - `run_sort_by_keys_performance.py` (both loops)
  - `run_exclusive_scan_in_place.py`
  - `run_compute_histogram.py`
  - `run_dtt_performance.py`
  - `run_segmented_sort_in_place_performance.py`, which also gains the missing autoselect.

  The change is a mechanical rename to `shamrock.algs.<fn>("<alg>", ...)`. sphinx-gallery runs every `run_*.py` (`doc/sphinx/source/conf.py:71,80`), so a missed rename breaks the docs CI.
- **Docs:** rewrite `doc/sphinx/source/dev_doc/implementation_selection.md`. The new version covers the Python API, the JSON schema, config export/import, the env var and CLI, tuning (match rules and priority), the autotune hook, and the C++ skeleton (constructor with name, default lambda, `get_or_autoselect`). It also fixes the stale points: "three functions", the `{128,256}` example, and the list of lazy-default algorithms.
- **`ImplVariant.hpp` file-level doc comment:** update it to describe the registry and the removal of the per-algorithm functions.

## Verification

1. **Build.** In `build/`, run `./shamenv_do shamconfigure` (the first run builds AdaptiveCpp), then `./shamenv_do shammake shamalgs shamtree shamsys`. Before running tests, do a full `./shamenv_do shammake && echo DONE`, and check that `./shamrock` and `./shamrock_test` exist.
2. **Unit tests.** Run `test -d reference-files || ./shamenv_do pull_reffiles`, then `./shamenv_do ./shamrock --smi`. Ask the user once which device to use. Then run `./shamenv_do ./shamrock_test --sycl-cfg X:X --loglevel 1 --unittest`, which includes the new `shamalgs/impl_registry` test and the migrated tests.
3. **Python end to end.** A scratch script run with `./shamenv_do ./shamrock --sycl-cfg X:X --rscript script.py`. It will:
   - list the algorithms;
   - loop over `set_impl`/`get_current_impl`;
   - call `get_impl_config()`, then `json.dump` the result to a file;
   - reload the file with `SHAMROCK_IMPL_CONFIG=file` and with `--impl-config file`, and check that `get_current_impl` matches;
   - add a tuning entry that matches the device, and check it after `autoselect_impl`.
4. **Benchmarks.** Run each of the 7 migrated benchmark scripts through `--rscript` (the docs CI runs them all). Also check that calling `shamrock.algs.autoselect_impl(...)` before `sys.init()` raises a clean error, not a crash.
5. **Lint.** `SETUPTOOLS_USE_DISTUTILS=stdlib pre-commit run --files <changed>` and `.claude/tools/clang-tidy-check.py` on the new and edited `.cpp` files.
   - The pre-commit hooks `check_no_utf8` and `doxygen_header` apply: no em-dashes or arrows in C++ files, and an `@file` header on new files.
   - Watch for `bugprone-use-after-move`, `performance-unnecessary-value-param`, missing `override`, and designated initializers.
6. **Commit** following AGENTS.md: author = the user, `--no-verify` amend, no session link. Then push to `claude/zealous-sagan-rb4bn7`.
