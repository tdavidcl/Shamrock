# Compiler probes

Two standalone programs. Both are self-contained, `-std=c++20`, no project
headers, and paste straight into godbolt.

## `nttp_probe.cpp` — which NTTP spelling is available? **(answered)**

The plan wants a unit as a template argument so that every dimension exponent is
a constant expression and `pow_constexpr_fast_inv<0>` expands to nothing. Two
spellings exist, and they are not interchangeable:

| Spelling | Takes a prvalue? | Needs floating-point NTTP? |
| --- | --- | --- |
| `template<Unit u>` (by value) | yes — `get<upow(metre,2)>()` | **yes**, `Unit` holds a `double` |
| `template<const Unit &u>` | no — must name the unit first | no, reference types are structural |

### How to run

```bash
g++     -std=c++20 -O2 nttp_probe.cpp -o probe   # default: by-value NTTP
clang++ -std=c++20 -O2 nttp_probe.cpp -o probe

# fallback variant
clang++ -std=c++20 -O2 -DUSE_REF_NTTP=1 nttp_probe.cpp -o probe
```

If the default build compiles, the by-value form is available. If it fails, note
which probe errored:

- **PROBE 1** (`template<double D>`) — no floating-point NTTP at all.
- **PROBE 2** (`template<Unit U>`) — no structural class NTTP with a `double`.
- **PROBE 3** (the `static_assert`s on `si.get<upow(...)>()`) — class NTTP works
  but not from a prvalue.

`-DSHOW_REF_REJECTS_PRVALUE=1` together with `-DUSE_REF_NTTP=1` is **expected to
fail**; it only demonstrates why the reference form needs a named variable.

### Results

| Compiler | `template<double D>` | `template<Unit u>` | `template<const Unit &u>` |
| --- | --- | --- | --- |
| clang < 18 (incl. the clang-15 CI leg) | **rejected** | **rejected** | **compiles** |
| clang ≥ 18 | compiles | compiles | compiles |

Conclusion: ship `template<const Unit &u>`, which works everywhere the project
currently builds. Keep the by-value overload commented out and enable it when
clang < 18 is dropped — it is purely additive.

### Known flaw in this probe

PROBE 1 is **unconditional**, so `-DUSE_REF_NTTP=1` still aborts the translation
unit on the floating-point check before the reference variant is reached. The
first run of this probe was therefore misread as "the fallback fails too", when
in fact it had never been exercised. Guard each feature check behind its own
`#if` before reusing this file.

## `codegen_probe.cpp` — what does `get` actually emit? **(measured)**

Implements enough of the planned design for one realistic function to compile,
so the assembly can be read. Compiles and runs clean on gcc 13 and clang 20.

```bash
g++     -std=c++20 -O2 codegen_probe.cpp -S -o -   # and -O3, and clang++
```

Symbols to compare in the asm pane:

| Symbol | Why |
| --- | --- |
| `test_func(UnitSystem<double>)` | the realistic case — `si` is a runtime by-value parameter |
| `test_func_local` | should be identical; confirms block-scope `static constexpr` costs nothing |
| `test_func_const` | same math with a `constexpr` input — expect a single constant load |
| `test_func_runtime` | value-argument overload, runtime exponents — the contrast case |
| `probe_hertz` | simplest case: one dimension, exponent −1 |
| `probe_hertz_byval` | same unit through the value-argument overload, argument statically known |

### Results

Body sizes from `nm --print-size --size-sort -C`, `-O2` (identical at `-O3`):

| Symbol | gcc 13 | clang 20 | what it emits |
| --- | --- | --- | --- |
| `probe_hertz` (NTTP) | 10 B | 6 B | **one `movsd`** — loads `s_inv`, zero arithmetic. The six zero-exponent dimensions and their `*1.0` factors fold out entirely. |
| `test_func` | 43 B | 39 B | identical 7-instruction bodies: load `si.m`, `mul` by au, **one** `divsd`, square, `mul` by au². |
| `test_func_local` | 43 B | 39 B | byte-identical to `test_func` — block-scope `static constexpr` costs nothing. |
| `test_func_const` | 13 B | 9 B | a single constant load. |
| `probe_hertz_byval` | 10 B | 52 B | **the divergence**: gcc folds it to the NTTP's single `movsd`, clang spills the `Unit` and emits a real `call`. |
| `test_func_runtime` | 525 B | 57 B | the honest worst case; no `divsd` in either, so the stored `_inv` members do their job. |

Two conclusions:

1. **The elimination works.** `test_func` keeps exactly one division out of the
   seven the constructor writes, and the two `get<>` calls feeding `unit_time`
   and `unit_mass` vanish with it. The NTTP rows are what `addget` emits today.
2. **The value-argument form is not a substitute, on clang.** The plan had
   assumed `fmul x, 1.0 → x` would fold on both compilers without fast-math.
   gcc does; clang does not, at `-O2` or `-O3`, even with a literal argument.
   So the NTTP form is the one to use wherever the unit is statically known —
   not merely the tidier option.

Still to add for a complete answer: the current `addget` implementation
side-by-side, so the NTTP form can be checked as byte-identical for the units
actually used in `ComputeEos.cpp` and `shamphys/*`.

## Provenance

Both files compile and run clean on gcc 13 and clang 20 (`-std=c++20 -O2 -Wall`,
and `-O3`) in the session container. `nttp_probe.cpp` was additionally run by
the user on clang 15 and clang 18 — that is where the NTTP result above comes
from, since this container has only clang 20.

`codegen_probe.cpp` also doubles as a numeric check: it prints `G` in
(Myr, au, M☉) as `3.947813e+13`, which is exactly the current library's
`3.94781e+25` divided by 10¹² — an independent confirmation of bug 4 to the
digit.
