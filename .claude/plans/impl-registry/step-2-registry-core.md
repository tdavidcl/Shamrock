# Step 2: registry core, with the old API untouched

**Read first:** `00-background.md`. It has the context, the constraints, the code background and
the common verification and commit rules. The API and semantics you implement are in its
"Target design" section.

**Depends on:** step 1, so that `compute_histogram_impl` is defined in
`src/shamalgs/src/primitives/compute_histogram.cpp`.

## Scope

1. **`src/shamalgs/include/shamalgs/ImplVariant.hpp`:** follow "`ImplVariant.hpp`: the one small
   change" in the background.
   - Delete the copy and move constructors and assignment operators of `ImplVariantGlobal`.
   - Optionally, throw on an empty `AutoselectFn`.
   - Add one sentence to the class doc comment: instances register by name, so they are neither
     copyable nor movable.
   - Do **not** touch `IImplVariant`.
2. **New `src/shamalgs/include/shamalgs/impl_registry.hpp` + `src/shamalgs/src/impl_registry.cpp`,**
   exactly as in "New `impl_registry.hpp` + `impl_registry.cpp`" in the background.
   - The API: `register_impl`, `ImplRegistrar`, `get_registered_algs`, `get_default_impl_list`,
     `get_current_impl`, `is_impl_set`, `set_impl`, `autoselect_impl`.
   - A Meyers-singleton `std::map<std::string, IImplVariant *, std::less<>>`.
   - A duplicate name throws **before** anything is stored.
   - An unknown name throws, listing the registered names.
   - `autoselect_impl` calls `shambase::get_check_ref(sched)` before `impl.autoselect(sched)`.
   - `set_impl` and `autoselect_impl` log.
   - Add `src/impl_registry.cpp` to `Sources` in `src/shamalgs/CMakeLists.txt`.
3. **Register all 8 algorithms.** Next to each global (the table "The 8 algorithms" in the
   background), add:
   - `constexpr std::string_view <algo>_impl_name = "<registry name>";`
   - an anonymous-namespace `shamalgs::impl_registry::ImplRegistrar <algo>_registrar{std::string(<algo>_impl_name), <algo>_impl};`,
     defined **after** the global in the same translation unit.

   The registry names are `reduction`, `is_all_true`, `scan_exclusive_sum_in_place`,
   `segmented_sort_in_place`, `sort_by_key_pow2_len`, `sort_by_keys`, `compute_histogram` and
   `clbvh_dual_tree_traversal`.

   For `compute_histogram`, the name constant goes in `compute_histogram.hpp` (step 4's dispatch
   site in the header needs it), and the registrar goes in `compute_histogram.cpp`.
4. **New test `src/tests/shamalgs/impl_registryTests.cpp`**, declared as
   `NEW_TEST(Unittest, "shamalgs/impl_registry", 1)`. It needs no CMake edit. It checks:
   - `get_registered_algs()` **contains** all 8 names. Check containment, not equality.
   - A round trip for every registered algorithm:
     1. `autoselect_impl(alg, shamsys::instance::get_compute_scheduler_ptr())`, then check that
        `is_impl_set`;
     2. save `get_current_impl`;
     3. `set_impl` each entry of `get_default_impl_list` and read it back;
     4. restore the saved value.

     Autoselect first, because a saved `"null"` cannot be restored.
   - An unknown algorithm name throws `std::invalid_argument` from every function.
   - `autoselect_impl("reduction", nullptr)` throws.
   - Registering `"reduction"` again throws.
     - Build the dummy as a function-local `shamalgs::ImplVariantGlobal<A>` whose lambda is
       `self.set(A{})`, where `A` is a file-scope tag struct with a `variant_type_name`.
     - Wrap the call for the macro:
       `REQUIRE_EXCEPTION_THROW(([&]{ shamalgs::impl_registry::register_impl("reduction", dummy); })(), std::invalid_argument)`.
   - **Never** register a test-local object under a new name. There is no unregister, so its
     entry would dangle once the object goes out of scope.

## Out of scope for this step

- Dispatch sites: they keep calling `impl::autoselect_impl_X` (step 4).
- The per-algorithm functions and bindings, which stay and keep working (step 6).
- Python (step 3).
- Migrating the existing tests (step 4).
- Docs (steps 5 and 6).

## Done when

- The registry lists all 8 algorithms at startup, both in `shamrock_test` and in the Python
  module.
- The new `shamalgs/impl_registry` test passes, and every existing test still passes, unchanged.
- `ImplVariantGlobal` can no longer be copied or moved, and every existing use still compiles.

## Verification

From "Common verification" in the background:
1. **Build:** `shammake shamalgs shamtree`, then the full build.
2. **Unit tests:** the full `--unittest` run.
3. **Lint:** pre-commit, and `clang-tidy-check.py` on `impl_registry.cpp`,
   `compute_histogram.cpp` and one algorithm `.cpp` you added a registrar to.
4. **Commit.**
