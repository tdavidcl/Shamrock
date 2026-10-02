# Plan: implementation-selection registry, with JSON config/tuning and autotune hooks

## Status (upstream `main` at `b1194a0`)

| Step | State |
|---|---|
| Scheduler passed to every `autoselect_impl_<algo>` | **Merged**, #2442 (`c7232a4`) |
| `IImplVariant::is_set` / `autoselect` virtuals, default rule given to the `ImplVariantGlobal` constructor | **Merged**, #2451 (`ce73d7b`) |
| Name-keyed registry `shamalgs::impl_registry`, name-keyed `shamrock.algs` API, removal of the per-algorithm functions | **Merged**, #2470, #2476, #2480, #2491, #2502, #2506 and the follow-up #2503 (record: `.claude/plans/impl-registry/`) |
| 1. Compiler-id move to shambackends | **Next**. Small, and a prerequisite for the export's `sycl` block |
| 2. Config JSON export/import, `SHAMROCK_IMPL_CONFIG`, `--impl-config` | Later, after 1 |
| 3. Hardware tuning at autoselect | Later, after 2 (same file schema) |
| 4. Autotune hook | Later, independent of 1-3 |

## Context

Every algorithm with several implementations wraps a `shamalgs::ImplVariantGlobal<Alts...>` and
registers it by name in `shamalgs::impl_registry`. C++ and Python (`shamrock.algs`) select
implementations through that name-keyed registry.

The 8 registered algorithms are `reduction`, `is_all_true`, `scan_exclusive_sum_in_place`,
`segmented_sort_in_place`, `sort_by_key_pow2_len`, `sort_by_keys`, `compute_histogram` and
`clbvh_dual_tree_traversal`.

The remaining features build on top of the registry:

- **Export and import** of the whole implementation config as JSON: from Python, from
  `SHAMROCK_IMPL_CONFIG`, and from `--impl-config`.
- **Hardware-specific tuning**, supplied by the user and matched against the device at
  autoselect.
- **An autotune hook** on `IImplVariant`. It defaults to "no autotuner", so autotuners can be
  added one algorithm at a time later.

Decisions made with the user (unchanged):

- **Python location:** `shamrock.algs`.
- **Python types:** one implementation is a JSON string; the full config is a dict.
- **Every change of selection goes through `impl_registry::set_impl`**, i.e. through
  `IImplVariant::set(string)`.
- **Export** leaves out unset algorithms. It includes a `device` block and a `sycl` block (the
  implementation name and the compiler id).
- **Tuning** is supplied by the user only. It is a list of device-matched entries, applied at the
  **autoselect** step.
- **Match priority** is most specific first. An unknown algorithm or implementation gives a
  warning, is skipped, and the next matching entry is tried. If nothing usable matches, the
  hard-coded default is used.
- **One file schema** is shared by `SHAMROCK_IMPL_CONFIG`, `--impl-config` and Python: `config`
  and/or `tunings`.
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
  - Import ignores `device` and `sycl`, applies `config` immediately through `set_impl`, and adds
    `tunings` to the tuning DB. The tunings are added first.
- **Match semantics** use the same nesting as the export, so an exported `device`/`sycl` block
  can be pasted into `match` and trimmed.
  - `device.backend`, `device.type` and `sycl.implementation` must match exactly.
  - `device.name`, `device.platform` and `sycl.compiler_id` match as substrings.
  - An unknown match key throws when the tuning is loaded, so typos can't silently fail to match.
  - Specificity is the number of leaf keys; `{}` matches every device.
- **Priority at autoselect**, per algorithm: matching entries that contain the algorithm, ordered
  by specificity (highest first), then by load order (later first). The first config that
  `set()` accepts wins. If `set()` rejects one (invalid implementation or JSON), log a warning
  and try the next. If none is accepted, use the hard-coded default.
- **`vendor` is left out**: `Device.cpp:354` always reports it as `UNKNOWN`.

## What exists today (at `b1194a0`)

### `src/shamalgs/include/shamalgs/ImplVariant.hpp`

- **`IImplVariant`** has five virtuals:
  - `get_current_config()`, which returns the string `"null"` when unset;
  - `get_default_config_list()`;
  - `set(std::string_view)`;
  - `bool is_set() const`;
  - `void autoselect(const sham::DeviceScheduler_ptr &)`.
- **`ImplVariantGlobal<Alts...>`:**
  - It has `using AutoselectFn = std::function<void(const sham::DeviceScheduler_ptr &, ImplVariantGlobal &)>;`
    and `explicit ImplVariantGlobal(AutoselectFn fn)`, which throws `std::invalid_argument` if
    `fn` is empty.
  - `autoselect(sched)` runs `fn(sched, *this)`, and the lambda picks the default by calling the
    public `set(Variant)`.
  - Copy and move are deleted, because the registry stores each instance's address.

### `src/shamalgs/include/shamalgs/impl_registry.hpp` + `src/shamalgs/src/impl_registry.cpp`

**API** (namespace `shamalgs::impl_registry`):
- `register_impl(std::string name, IImplVariant &)`
- `get_registered_algs()`, sorted
- `get_default_impl_list(alg)`
- `get_current_impl(alg)`
- `is_impl_set(alg)`
- `set_impl(alg, impl)`
- `autoselect_impl(alg, sched)`

**Registration:** the `SHAMALGS_REGISTER_IMPL(name, impl)` macro, a `shambase::call_lambda`
static placed at namespace scope right after the global, in the same translation unit. Each
algorithm defines a `constexpr std::string_view <algo>_impl_name`.

**Behavior:**
- **Storage:** a Meyers-singleton `std::map<std::string, IImplVariant *, std::less<>>`. There is
  no unregister.
- **Errors:**
  - a duplicate name throws `std::invalid_argument` before storing anything;
  - an unknown name throws `std::invalid_argument` listing the registered names;
  - `autoselect_impl` calls `shambase::get_check_ref(sched)`, which throws `std::runtime_error`
    on a null scheduler.
- **Logging:** `set_impl` logs "setting ..." after `impl.set()` succeeds, and `autoselect_impl`
  logs "defaulting ...". Both log on rank 0 only (#2503).
- **Dispatch sites** call `impl_registry::autoselect_impl(<algo>_impl_name, sched)` when
  `!X_impl.is_set()`. So anything added to `autoselect_impl`, like tuning, applies to both lazy
  and explicit autoselect.

### Python (`src/shampylib/src/pyShamalgs.cpp`)

- The `{ // implementation registry }` block (around lines 112-190) binds `get_registered_algs`,
  `get_default_impl_list`, `get_current_impl`, `is_impl_set`, `set_impl` and `autoselect_impl`.
- Only `autoselect_impl` fetches `shamsys::instance::get_compute_scheduler_ptr()`.
- The legacy `impl_param` binding (around lines 47-70) is unrelated; leave it alone.

### Tests and docs

- `src/tests/shamalgs/impl_registryTests.cpp` covers:
  - the registered names;
  - a per-algorithm round trip;
  - unknown names;
  - a null scheduler (`std::runtime_error`);
  - duplicate registration.
- `doc/sphinx/source/dev_doc/implementation_selection.md` documents the name-keyed Python API
  ("User side") and the `SHAMALGS_REGISTER_IMPL` pattern ("Developer side").

## Remaining design

### Convention for all additions

- **Logging:** informational logs are rank 0 only, like #2503. So are warnings about skipped
  config or tuning entries, because every rank loads the same file.
- **Mutations:** every change of selection goes through `impl_registry::set_impl`.
- **`std::exception`:** catch it wherever a user-supplied config is applied. nlohmann throws
  `parse_error`, `type_error` and `out_of_range`, not only `invalid_argument`.

### 1. Move the compiler id string down to shambackends

- `shamrock_compiler_id_string` is generated by `src/shamrock/CMakeLists.txt:37-54` and compiled
  into shamlib (lines 64 and 70), which shamalgs cannot link.
- `SHAMROCK_COMPILER_ID_STRING` is set at top-level scope by `cmake/SYCLAdaptACppDirect.cmake:77`
  or `cmake/SYCLAdaptIntelLLVM.cmake:56` (via `cmake/ShamConfigureSYCL.cmake`), before
  `add_subdirectory(src)`.
- **Move:** move the block, including the `FATAL_ERROR` check, into
  `src/shambackends/CMakeLists.txt`, add the generated `compiler_id.cpp` to its sources, and
  remove it from shamlib's `add_library` calls.
- **Avoid rebuilds on every configure.** `file(WRITE)` rewrites the file on each configure, which
  would force a relink of `libshambackends` and everything downstream. Write to
  `compiler_id.cpp.tmp`, then `configure_file(... COPYONLY)`, which copies only when the content
  changed.
- **New header:** add `shambackends/include/shambackends/sycl_compiler_id.hpp`, declaring
  `extern const char *shamrock_compiler_id_string;`.
  - The generated `.cpp` `#include`s it, so the declaration and the definition are type-checked
    against each other.
  - `src/shamrock/include/shamrock/version.hpp:35-36` includes it instead of re-declaring the
    symbol.
  - Its only user, `pyShamsys.cpp:83-88`, is unaffected.

### 2. Config export/import, `SHAMROCK_IMPL_CONFIG`, `--impl-config`

**C++ additions to `impl_registry`:**
- `nlohmann::json get_impl_config(const sham::DeviceScheduler_ptr &sched)`: the `device` block,
  the `sycl` block, and `config` restricted to algorithms that are set (`is_impl_set`).
- `void set_impl_config(const nlohmann::json &)`: the unified import described above.
  - `config`: an unknown algorithm, or any `std::exception` from `set_impl`, gives a warning and
    a skip; the rest is still applied.
  - `tunings`: once step 3 exists. Until then, a `tunings` key gives one warning that tunings are
    not supported yet.
- `void load_impl_config_file(const std::string &path)`: reads the file, then calls
  `set_impl_config`.

**Where the `sycl` and `device` blocks come from:**
- `implementation`: `sycl_implementation` in `shambackends/sycl.hpp:24-32`, mapped by a local
  helper to `"AdaptiveCpp"`, `"DPC++"` or `"Unknown"`.
- `compiler_id`: `shamrock_compiler_id_string`, after step 1.
- The `device` block: `sham::backend_name`, `sham::device_type_name`, `prop.name` and
  `prop.platform` (`shambackends/Device.hpp:54`, `:70`).
- Document the exact strings for matching: `backend_name` returns `"CUDA"`, `"ROCm"`,
  `"OpenMP"` or `"Unknown"`, and `device_type_name` returns `"CPU"`, `"GPU"` or `"UNKNOWN"`.

**Env var and CLI** (in shamsys, which already links shamalgs and shamcmdopt):
- **`SHAMROCK_IMPL_CONFIG`:**
  - At namespace scope in `src/shamsys/src/NodeInstance.cpp`, register only its documentation, so
    it appears in `--help`. See the registration pattern at `:295`.
  - Read its value with `shamcmdopt::getenv_str` *inside* `init_sycl_mpi` (`:300`), then call
    `load_impl_config_file`. Reading it at library load time would miss an `os.environ[...]` set
    after `import shamrock`.
  - This one path covers the executable, the tests and `shamrock.sys.init()` in lib mode.
  - Nothing inside `init_sycl_mpi` runs one of the algorithms, so no autoselect can happen before
    the config is loaded.
  - Document that it overrides any `set_impl` done in Python before `sys.init()`.
- **`--impl-config (filepath)`:**
  - Registered in `src/main.cpp` (options at lines 54-75, `opts::init` at `:86`) and in
    `src/main_test.cpp` (options at 47-77, `opts::init` at `:90`).
  - Applied in `shamsys::instance::init(argc, argv)` (`NodeInstance.cpp:316`) after
    `init_sycl_mpi`, so it runs after the env var and wins over it. It cannot go in
    `init_sycl_mpi`, because cmdopt is uninitialized in lib mode.
  - Throw if the path is empty: `get_option` returns `""` when the flag is the last argument.
  - `instance::init(argc, argv)` only runs when `--sycl-cfg` is given (`main.cpp:116`,
    `main_test.cpp:125`). Warn in `main` when `--impl-config` is given without `--sycl-cfg`.
- Both paths go through `set_impl_config`, and so through `set_impl`.

**Python additions to `shamrock.algs`:**
- `get_impl_config() -> dict` (uses the compute scheduler)
- `set_impl_config(cfg: dict)`
- `load_impl_config(path: str)`

Dict conversion uses `json.loads`/`json.dumps` through `py::module_::import("json")`, the same
pattern as `shammodels/common/include/shammodels/common/shamrock_json_to_py_json.hpp`
(`to_py_json`/`from_py_json`), written inline as `pyUnits.cpp` does.

### 3. Hardware tuning at autoselect

**Storage:** `std::vector<TuningEntry>` next to the registry map, where `TuningEntry` has the
fields `name`, `match`, `config` and `load_index`. Construct it with designated initializers,
because clang-tidy enables `modernize-use-designated-initializers`.

**Loading:** `set_impl_config` handles `tunings`.
- The whole list is validated before any entry is added, so a schema error (an unknown match
  key, or the wrong types) throws with nothing loaded.
- An unknown algorithm name in an entry gives a warning when loaded.

**`autoselect_impl(alg, sched)`** first tries each tuning candidate for `alg` that matches
`sched`'s device, in priority order, through `set_impl`.
- A failed candidate gives a warning, and the next one is tried.
- If no candidate is accepted, it calls `impl.autoselect(sched)`.
- It logs which source won: a tuning entry's name, or the default.

**Additions:** `nlohmann::json get_impl_tunings()` and `void clear_impl_tunings()`, plus the
Python `get_impl_tunings() -> list[dict]` and `clear_impl_tunings()`.

### 4. Autotune hook

- **`IImplVariant`** gains `bool has_autotune() const` and
  `void autotune(const sham::DeviceScheduler_ptr &)`.
- **`ImplVariantGlobal`** takes an optional second constructor argument, an `AutotuneFn` with the
  same `(sched, self)` signature, empty by default. `autotune` throws if it is empty.
- **Registry:**
  - `has_autotune(alg)`;
  - `autotune_impl(alg, sched)`, which returns `false` and logs on rank 0 that no autotuner is
    implemented when there is none. Otherwise it runs `impl.autotune(sched)`, logs the result,
    and returns `true`.
- **Python:** `has_autotune(alg) -> bool` and `autotune_impl(alg) -> bool` (uses the compute
  scheduler).
- Autotuners are then rolled out one algorithm at a time, as separate changes.

## Tests and docs for the remaining features

- **Extend `src/tests/shamalgs/impl_registryTests.cpp`** (or add sibling test files):
  - The export leaves out unset algorithms, and a `get_impl_config` / `set_impl_config` round
    trip restores the selections.
  - `set_impl_config` warns on and skips an unknown algorithm or implementation, while still
    applying the rest.
  - **Tuning:**
    - specificity layering: entries matched on the real device's backend and name, where the most
      specific one holds an invalid implementation, fall through to the next;
    - no match leads to the default;
    - an unknown match key throws.

    Save and restore the tuning DB with `get_impl_tunings()` / `clear_impl_tunings()`.
  - **Autotune:** `has_autotune` is `false` and `autotune_impl` returns `false` for the existing
    algorithms. Also test a selector with an autotuner.
  - **Never register a test-local object.** There is no unregister, so its entry would dangle.
    For a dummy selector, use a file-scope global in the test translation unit, registered once
    with `SHAMALGS_REGISTER_IMPL` under a clearly test-only name. Otherwise test against a real
    algorithm and save and restore its selection, autoselecting first, since `"null"` cannot be
    restored.
- **Docs:** extend `doc/sphinx/source/dev_doc/implementation_selection.md` with sections on
  config export/import, the env var and CLI, tuning (match rules and priority), and the autotune
  hook. Keep the existing sections.

## Verification

1. **Build.** In `build/`, run `./shamenv_do shamconfigure` (the first run builds AdaptiveCpp),
   then `./shamenv_do shammake shambackends shamalgs shamsys`. Before running tests, do a full
   `./shamenv_do shammake && echo DONE`, and check that `./shamrock` and `./shamrock_test`
   exist.
2. **Unit tests.**
   - `test -d reference-files || ./shamenv_do pull_reffiles`, then `./shamenv_do ./shamrock --smi`.
   - Ask the user once which device to use.
   - `./shamenv_do ./shamrock_test --sycl-cfg X:X --loglevel 1 --unittest`.
3. **Python end to end.** A scratch script run with
   `./shamenv_do ./shamrock --sycl-cfg X:X --rscript script.py` that:
   - calls `get_impl_config()` and `json.dump`s the result to a file;
   - reloads the file with `SHAMROCK_IMPL_CONFIG=file` and with `--impl-config file`, and checks
     that `get_current_impl` matches;
   - adds a tuning entry that matches the device, and checks it after `autoselect_impl`.

   Also run one benchmark script (e.g. `examples/benchmarks/run_reduction_performance.py`) to
   check that nothing regressed.
4. **Lint.**
   - `SETUPTOOLS_USE_DISTUTILS=stdlib pre-commit run --files <changed>`, and
     `.claude/tools/clang-tidy-check.py` on the new and edited `.cpp` files.
   - The pre-commit hooks `check_no_utf8` and `doxygen_header` apply: no em-dashes or arrows in
     C++ files, and an `@file` header on new files.
5. **Commit** following AGENTS.md: author = the user, `--no-verify` amend, no session link, and
   the CLAUDE.md branch-naming rule.
