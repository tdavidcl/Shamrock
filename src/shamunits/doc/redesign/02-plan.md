# Redesign `shamunits` around C++20 values

## Context

`src/shamunits/` is a compile-time unit-conversion library whose data is spread
across six hand-maintained tables that must be kept in sync by hand:

| What | Restated in |
| --- | --- |
| 38 units | `XMAC_UNITS` enum (`Names.hpp`) · 38 `addget(...)` bodies · 38 `case` labels in `getter_1` · dimension only in a comment · `doc/godbolt.cpp` |
| 13 prefixes | `XMAC_UNIT_PREFIX` · 13 `case` labels in `getter_2` |
| 16 conversion factors | `ConversionConstants.hpp` + a 47-line doxygen block |
| 35 constants | `UNITS_CONSTANTS` (dimension) · `Constants<T>::Si` (value, 50 lines away) · ~210 lines of `\fn` doxygen · `pyUnits.cpp` |

Adding a unit means editing four places; adding a constant, four or five.
Nothing checks that the dimension column agrees with the value column, and the
duplication has already produced five defects:

1. `Constants.hpp:48,79` — `h` and `hbar` registered as **J·s⁻¹**; they are J·s.
2. `Constants.hpp:76` — `guiness_density` registered as **kg·m⁻¹**; its value is kg·m⁻³.
3. `UnitSystem.hpp:233` — `to<pref,u,power>()` forwards to `get<u,-power>()`, which
   resolves to the no-prefix overload and **silently drops the prefix**.
4. **The prefix is applied once per nesting level, not once per unit.**
   `addget(years)` is `PREF * Uget(s,1) * Cget(yr_to_s,1)`, and `Uget(s,1)`
   re-applies `PREF` — so `get<mega, units::years>()` yields `1e12·yr_to_s`
   instead of `1e6·yr_to_s`. `Joule` nests three deep: `get<kilo, Joule>()` is
   off by 10⁶. **Verified against the README's own output**: it prints
   `to<units::second>() = 3.15576e+19` (1 Myr is `3.15576e13` s) and
   `G = 3.94781e+25` (correct value `3.947813e13`, measured) — off by 10⁶ and by 10⁶ squared,
   G carrying s⁻². This reaches production:
   `src/pylib/shamrock/utils/analysis/UnitHelper.py` uses `to("yr", pref="M")`
   and `pref="G"` for Myr/Gyr axis scaling.
5. `doc/godbolt.cpp` is a stale hand-copied fork of the whole library.

### The compile-time property, stated correctly

The current library is **not** compile-time evaluated in the language sense —
`UnitSystem`'s constructor is not `constexpr` (`UnitSystem.hpp:141`), so
`constexpr UnitSystem<double> si{};` does not compile today. The folding seen on
godbolt is optimizer work: everything inlines and `pow_constexpr_fast_inv<power>`
is an `if constexpr` ladder that collapses before the optimizer runs.

Making a unit a *value* would, on its own, move the dimension exponents from
template arguments to runtime ints and turn that ladder into a loop. The
redesign avoids paying that: the NTTP `get<u>()` overload (see "Guaranteed
elimination" below) keeps every exponent a constant expression and reuses
`pow_constexpr_fast_inv` verbatim, so a named unit emits exactly what it emits
today. On top of that, the property becomes **language-guaranteed and
CI-checked** rather than optimizer-dependent, which the current design cannot
offer at all:

```cpp
constexpr UnitSystem<double> si{};                        // constexpr ctor
inline constexpr Unit au_sq = upow(units::astronomical_unit, 2);
constexpr double au2 = si.get<au_sq>();                   // or it won't compile
static_assert(close(astro_units.get<au_sq>(), 1.0, 1e-12));   // never ==, see below
```

For a genuinely runtime unit system (one loaded from JSON) the folding argument
never applied anyway — the base units aren't constants. All such call sites
(`shamphys/{BlackHoles,Planets,collapse,orbits}.hpp`, the `SolverConfig`
getters) are host-side setup, none per-particle inside a kernel.

### Decisions taken with the user

- **No macros at all** — `inline constexpr Unit` declarations + one
  `constexpr std::array` registry per category.
- **There is no `power` parameter — the power lives in the unit.** `upow(metre,2)`
  *is* a `Unit`, so `get` takes one argument and nothing else. This is both
  simpler and strictly more expressive than a scalar `power`, which can only
  scale every exponent uniformly: `m·s⁻²` is unreachable as `u^p` but trivial as
  `metre / upow(second, 2)`. It also removes the division question outright —
  `upow` folds the `si_factor` exponentiation at compile time, so `get` itself
  never divides. Three call shapes:
  - `get<unit, Tret>()` — unit as a `const Unit&` NTTP. Every exponent is a
    constant expression, so **unused dimensions emit no code at all**.
  - `get<Tret>(unit)` — unit as an ordinary argument, for string-resolved units.
  - `get<Tret>(unit, power)` — convenience, exactly `get(upow(unit, power))`;
    the only shape with a runtime exponent, and Python's path.
- **The NTTP is by reference, so a composed unit must be named first**
  (`inline constexpr Unit au_sq = upow(units::astronomical_unit, 2);`). Measured:
  `template<Unit u>` by value would allow `get<upow(metre,2)>()` inline, but it
  needs floating-point NTTP support that clang < 18 lacks, and CI still builds
  on clang 15. That variant stays in the header commented out, to enable once
  clang < 18 is dropped. See "Measured on clang 15 and 18" below.
- **The scalar type is never pinned to `double`.** The table's `si_factor` is
  `double` (a compile-time constant, exact for every entry); it is narrowed once
  at the boundary and all arithmetic runs in the system's `T`. Both overloads
  additionally take `Tret = T` so a caller can pick the result type.
- Named C++ accessors `.G()`, `.sol_mass()` are **replaced** by
  `get<constants::G>()`. Python keeps them all.
- Python API and JSON keys frozen; C++ call sites (~16) may change.
- Fix all four value/behaviour bugs, and add dimension `static_assert`s.

---

## Design

**One type, one operation, three tables.** A unit, a prefix and a physical
constant are all the same thing: a dimension plus a magnitude in SI.

### `Unit.hpp` (new; no STL, device-safe)

```cpp
namespace shamunits {
    /// Exponents of the seven SI base dimensions
    struct Dimension {
        int second{}, metre{}, kilogram{}, ampere{}, kelvin{}, mole{}, candela{};
    };

    /// A unit: its dimension, and its magnitude in SI base units
    struct Unit {
        Dimension dim{};
        double si_factor = 1;
    };

    constexpr Unit operator*(Unit a, Unit b);    // factors multiply, exponents add
    constexpr Unit operator/(Unit a, Unit b);
    constexpr Unit operator*(double s, Unit u);  // 149597870700.0 * metre
    constexpr Unit operator*(Unit u, double s);
    constexpr Unit operator/(Unit u, double s);
    constexpr Unit operator/(double s, Unit u);
    constexpr Unit upow(Unit u, int n);          // named upow: avoids ADL fights
                                                 // with std::pow / sycl::pow
    constexpr bool operator==(Unit, Unit) = default;
}
```

`si_factor` is `double`, not `T`, so **one** shared table serves every `T`
(templating `Unit` on `T` would re-fragment the tables per scalar type for no
gain, and would rule out the reference-NTTP form). Every factor in the library
is exactly representable in `double` — `ly_to_m` is `147823913634075 × 2⁶`, well
inside 2⁵³ — so the table is lossless. It is *not* a computation type: it is
narrowed to `Tret` once at the boundary of `get`, and no arithmetic anywhere
happens in `double` unless `Tret` is `double`. For `f32` this is also strictly
more accurate than today, where `ConversionConstants<f32>::au_to_m` rounds to
`149597872128.0f` before any arithmetic happens.

### `unit_table.hpp` — **the** unit table

Two adjacent lines per unit, one file, one order:

```cpp
namespace shamunits::units {
    inline constexpr Unit metre  = base_unit(Dimension{.metre = 1});
    inline constexpr Unit newton = kilogram * metre / upow(second, 2);
    inline constexpr Unit joule  = newton * metre;
    inline constexpr Unit year   = 31557600.0 * second;
    inline constexpr Unit astronomical_unit = 149597870700.0 * metre;

    /// long name, short name, unit — one entry per unit above, same order
    inline constexpr auto registry = std::to_array<NamedUnit>({
        {"metre",  "m",  metre},   {"Newton", "N",  newton},
        {"Joule",  "J",  joule},   {"years",  "yr", year},
        {"astronomical_unit", "au", astronomical_unit},
    });
}
```

- Registry strings stay **byte-identical** to today's (`"Newton"`, `"years"`,
  `"Bequerel"` misspelling included) — `UnitHelper.py` resolves `"m"`, `"mn"`,
  `"hr"`, `"dy"`, `"yr"`, `"kg"`, `"au"`, `"pc"`, so both long and short forms
  must keep resolving. C++ *identifiers* normalise to `lower_case`, which is
  both the SI convention for unit names (`newton`, `joule`; only the symbols
  N, J are capitalised) and the `.clang-tidy` constant rule, now applicable
  since these are variables rather than enum values.
- `ConversionConstants.hpp` disappears: `au_to_m` **is**
  `astronomical_unit.si_factor`.
- `static_assert`s guard the table: no duplicate/empty registry names, and
  cross-checks like `joule.dim == (newton * metre).dim`,
  `watt.dim == (joule / second).dim`.

### `prefix_table.hpp` — a prefix is a dimensionless `Unit`

This is what fixes bug 4 structurally, and removes the third `get` parameter:

```cpp
namespace shamunits::prefix {
    inline constexpr Unit mega = scale(1e6);   // Unit{Dimension{}, 1e6}
    inline constexpr auto registry = std::to_array<NamedUnit>({{"mega","M",mega}, ...});
}
```

`si.get(prefix::mega * units::year)` — the prefix multiplies the unit once, and
`power` raises the product. The `UnitPrefix` enum and its two `unordered_map`s
are deleted; Python passes prefix *strings*, which the registry resolves.

### `constant_table.hpp` — **the** constant table

Same type again, so `Constants<T>::Si` *and* the `UNITS_CONSTANTS` dimension
column both vanish — magnitude and dimension in one expression:

```cpp
namespace shamunits::constants {
    inline constexpr Unit c    = 299792458.0 * units::metre / units::second;
    inline constexpr Unit h    = 6.62607015e-34 * units::joule * units::second;  // was J.s-1
    inline constexpr Unit G    = 6.6743015e-11 * units::newton
                                 * upow(units::metre, 2) / upow(units::kilogram, 2);
    inline constexpr Unit hbar = h / (2 * pi<double>);
    inline constexpr Unit Z_0  = mu_0 * c;                 // dimension derived, not retyped
    inline constexpr Unit year = units::year;              // alias, not a second copy
}
```

Derived constants now *prove* their dimensions, which is what makes the asserts
worth having:

```cpp
static_assert(constants::Z_0.dim       == units::ohm.dim);
static_assert(constants::epsilon_0.dim == (units::farad / units::metre).dim);
static_assert(constants::sigma.dim == (units::watt / upow(units::metre,2)
                                        / upow(units::kelvin,4)).dim);
static_assert(constants::astronomical_unit == units::astronomical_unit);  // two tables today
```

Dimensionless ratios (`fine_structure`, `proton_electron_ratio`) become
dimensionless `Unit`s. `pi<T>` stays a plain variable template —
`shamphys/collapse.hpp` uses it inside a kernel. (Note for the PR, not this
change: it duplicates `shambase::constants::pi<T>` bit for bit.)

### `UnitSystem.hpp` (rewritten: ~330 lines → ~70)

Public state, constructor signature and all 14 field names stay
**byte-identical** — `shamrock/io/units_json.hpp` reads `p.s_inv`… and
reassigns `p = UnitSystem<T>(...)`, and the JSON keys are a persisted on-disk
format. The constructor gains `constexpr`.

Three entry points replace `addget` × 38, `getter_1`, `getter_2`, `runtime_get`
and `runtime_to`. None computes in `double`: `si_factor` is narrowed to `Tret`
once, and every multiply runs in `Tret`.

```cpp
/// Unit known at compile time. Every exponent below is a constant expression,
/// so pow_constexpr_fast_inv<0> expands to nothing and an unused dimension
/// costs zero instructions -- the same expansion as the old addget. u.si_factor
/// was already raised by upow at compile time, so there is no exponent to apply
/// to it and no reciprocal to form.
template<const Unit &u, class Tret = T>
constexpr Tret get() const noexcept {
    using namespace details;
    return Tret(u.si_factor)
         * pow_constexpr_fast_inv<u.dim.second  >(Tret(s),   Tret(s_inv))
         * pow_constexpr_fast_inv<u.dim.metre   >(Tret(m),   Tret(m_inv))
         * ... ;   // seven base dimensions
}

/// Unit as an ordinary argument (string-resolved units). Exponents are runtime
/// because u is, but s_inv .. cd_inv are stored, so ipow needs no division.
template<class Tret = T>
constexpr Tret get(Unit u) const noexcept {
    return Tret(u.si_factor)
         * details::ipow(Tret(s), u.dim.second, Tret(s_inv))
         * details::ipow(Tret(m), u.dim.metre,  Tret(m_inv))
         * ... ;
}

/// Convenience / Python path: exactly get(upow(u, power)). The only shape with
/// a runtime exponent on si_factor, and the only one that can divide (p < 0).
template<class Tret = T>
constexpr Tret get(Unit u, int power) const noexcept { return get<Tret>(upow(u, power)); }

/// to() negates the exponents in the body -- u.dim.* and 1.0/u.si_factor are
/// constant expressions there, so the compile-time path still never divides.
template<const Unit &u, class Tret = T> constexpr Tret to() const noexcept;
template<class Tret = T> constexpr Tret to(Unit u) const noexcept;          // = get(upow(u,-1))
template<class Tret = T> constexpr Tret to(Unit u, int power) const noexcept;

// --- Enable once the project drops clang < 18; see "Measured" below. Purely
// --- additive: get<u>() means the same thing, so no call site has to move.
//
// template<Unit u, class Tret = T>            // by value -> allows a prvalue:
// constexpr Tret get() const noexcept { ... } //   get<upow(units::metre,2)>()
```

`get<au_sq>()`, `get(upow(units::metre, 2))` and `get(units::metre, 2)` all
work. The NTTP form is the one to reach for when the unit is known statically.

Hand-checked against the current semantics:
`si.get<units::astronomical_unit>()` = `1.496e11`; with `unit_length` set to
that, `astro.get<au_sq>()` = `1`; `si.get<megayear>()` = `3.15576e13` for
`inline constexpr Unit megayear = prefix::mega * units::year`; and G in
(Myr, au, M☉) = `3.947813e13` — **measured**, and exactly the README's
`3.94781e+25` divided by 10¹², which pins bug 4 to the digit.

#### Measured on clang 15 and 18

Settled with a standalone probe (`nttp_probe.cpp`) rather than assumed:

- **`template<double D>` is rejected by clang < 18.** So is any NTTP whose type
  is a class holding a `double` — i.e. `template<Unit u>` by value. That rules
  out passing a prvalue such as `get<upow(units::metre, 2)>()`, since a prvalue
  cannot bind to a reference NTTP either.
- **`template<const Unit &u>` compiles on clang 15**, and the whole design
  (`static_assert`s included) passes. A reference NTTP has *reference* type,
  which is structural regardless of what it refers to.
- On **clang ≥ 18** the by-value form compiles too, with every `static_assert`
  passing.

Why the reference form is valid at all: `inline constexpr` namespace-scope
variables have **external linkage**, so they are legal reference-NTTP targets
with one address across every TU. (Since C++17 linkage is not even required —
static storage duration suffices — so a block-scope `static constexpr Unit`
works too; `codegen_probe.cpp`'s `test_func_local` exercises that and compiles
to bytes identical to the namespace-scope version.) Leave a comment in the
header saying why the spelling is `const Unit&`, or someone will "modernise" it
back to by-value and break the clang-15 leg.

So the reference NTTP is the form to ship: it keeps the elimination guarantee on
every supported compiler today. The only thing deferred is the nicer inline
spelling — a composed unit must be named first, which in this tree costs one
line in the `au²` example and nothing else, since every real call site names a
plain unit. Keep the by-value overload commented out next to it, and enable it
when clang < 18 is dropped (expected within about a year).

`details/utils.hpp` keeps `pow_constexpr_fast_inv` **unchanged** — the NTTP
overload reuses it verbatim — and gains one sibling, `ipow(T a, int n, T a_inv)`:
the runtime twin, same recursion shape, same **caller-supplies-the-reciprocal**
trick (so negative exponents use the exact stored `s_inv`, literally the user's
`unit_time`, never `1/(1/unit_time)`), with early returns for `n == 0` and
`n == 1`. No `get` overload divides; the only division in the library is inside
`upow(u, n)` for `n < 0`, which is constant-folded on the NTTP path. That
reciprocal handling is the one place where sloppiness would actually lose
accuracy.

### Guaranteed elimination of unused dimensions

The old `addget` emitted only the multiplies a unit actually needs, because
every exponent was a template argument. The NTTP overload preserves that
exactly:

| Call shape | What is emitted | Divisions | Guaranteed by |
| --- | --- | --- | --- |
| `constexpr auto x = si.get<u>();` | nothing — fully evaluated by the front end | 0 | the language |
| `usys.get<units::hertz>()` | exactly one multiply; the six zero-exponent dimensions expand to `T{1}` and vanish | 0 | the language (`if constexpr` in `pow_constexpr_fast_inv`) |
| `usys.get<au_sq>()` | same — `upow` folded into the named constant | 0 | the language |
| `usys.get(u)`, runtime `u` | seven `ipow` | 0 | — |
| `usys.get(u, p)` | the above plus a runtime `upow` | 1 iff `p < 0` | — |

**Measured** with `codegen_probe.cpp` at `-O2` and `-O3`, gcc 13 and clang 20
(body sizes from `nm --print-size`):

| Symbol | gcc | clang | what it emits |
| --- | --- | --- | --- |
| `probe_hertz` (NTTP) | 10 B | 6 B | **one `movsd`** — loads `s_inv`, zero arithmetic. Better than "one multiply": the `*1.0` factors fold out entirely. |
| `test_func` (runtime system, NTTP units) | 43 B | 39 B | identical 7-instruction bodies — load `si.m`, `mul` by au, **one** `divsd`, square, `mul` by au². Six of the seven constructor divisions eliminated. |
| `test_func_const` (`constexpr` input) | 13 B | 9 B | a single constant load. |
| `probe_hertz_byval` (value arg, literal) | 10 B | 52 B | **gcc folds it to the NTTP's single `movsd`; clang does not** — it spills the `Unit` and emits a real `call`, at both `-O2` and `-O3`. |
| `test_func_runtime` (genuinely runtime unit) | 525 B | 57 B | the honest worst case; no `divsd`, so the stored `_inv` members do their job. |

The NTTP rows are what `addget` emits today, so the guarantee holds. But the
assumption that the value-argument form folds "on both compilers without
fast-math" is **false for clang** — it leaves a call even with a literal
argument. That makes the NTTP form the one to use wherever the unit is known
statically, not merely the tidier option. Guidance for the header: use the
NTTP form whenever the unit is known at compile time (naming the composed unit
if needed), the value-argument form for string-resolved units, and
`get(u, power)` only where the exponent is genuinely runtime. Every current call site names a
unit literally (`ComputeEos.cpp`, `Phantom2Shamrock.cpp`, `shamphys/*`), so all
of them get the NTTP form and are codegen-identical to today.

### `Constants.hpp` (rewritten, ~15 lines)

Retained because Python binds it as a class and it is the spelling the user
approved; it is now a thin pairing of a unit system with the constant table.

```cpp
template<class T>
struct Constants {
    const UnitSystem<T> units;
    constexpr explicit Constants(const UnitSystem<T> units) : units(units) {}

    template<const Unit &c, class Tret = T>
    constexpr Tret get() const noexcept { return units.template get<c, Tret>(); }

    template<class Tret = T>
    constexpr Tret get(Unit c) const noexcept { return units.template get<Tret>(c); }

    template<class Tret = T>
    constexpr Tret get(Unit c, int power) const noexcept
        { return units.template get<Tret>(c, power); }
};
template<class T> Constants(UnitSystem<T>) -> Constants<T>;   // CTAD, used by shamphys
```

`Constants<T>::Si` is deleted (grep-verified: no users outside `src/shamunits/`);
the SI value of a constant is `constants::G.si_factor`.
`usys.get<constants::G>()` works directly and is equivalent.

### `pyUnits.cpp` — bindings become macro-free and self-extending

```cpp
auto cls = py::class_<shamunits::Constants<f64>>(m, "Constants")
               .def(py::init<shamunits::UnitSystem<f64>>());
for (const auto &e : shamunits::constants::registry)
    cls.def(std::string(e.long_name).c_str(),
            [u = e.unit](shamunits::Constants<f64> &c, i32 p) { return c.get(u, p); },
            py::arg("power") = 1);
```

Same loop for `UnitSystem.get`/`.to`, which resolve name and prefix through the
registries and multiply. A new constant appears in Python with no binding edit.

---

## Files

**New** (`src/shamunits/include/shamunits/`): `Unit.hpp` (STL-free,
device-safe), `details/registry.hpp` (`NamedUnit` + constexpr lookup replacing
the four `unordered_map`s), `unit_table.hpp`, `prefix_table.hpp`,
`constant_table.hpp`.

**Rewritten**: `UnitSystem.hpp`, `Constants.hpp`, `details/utils.hpp` (adds
runtime `ipow`), `src/shampylib/src/pyUnits.cpp`.

**Deleted**: `ConversionConstants.hpp` (`K_degC_offset` is affine, has no `Unit`
home and is dead — drop it), `Names.hpp`, the `units::UnitName` and
`UnitPrefix` enums, and ~260 lines of `\fn` doxygen walls — replaced by one
`///` per declaration, which `EXTRACT_ALL=NO` + `WARN_IF_UNDOCUMENTED=YES` in
`doc/dox.conf` require anyway.

**C++ call sites (~16, mechanical)**: `.G()`/`.c()`/`.mu_0()` →
`get<constants::G>()` — the NTTP form, since every one of these names a unit or
constant literally, so codegen stays identical to today;
`units::kg`/`s`/`m` → `units::kilogram`/`second`/`metre`.
In `shamphys/{BlackHoles,Planets,collapse,orbits}.hpp`,
`shammodels/{sph,ramses,gsph}/…/SolverConfig.hpp`,
`shammodels/sph/src/io/Phantom2Shamrock.cpp`,
`shammodels/sph/src/modules/ComputeEos.cpp`. Also drop the 7 dead `shamunits`
includes found in the survey (e.g. `NeighbourCache.cpp`).

**`doc/godbolt.cpp`**: stop hand-maintaining a sixth copy — replace with
`tools/gen_shamunits_amalgamation.py` (~40 lines: recursively inline local
`#include`s, strip `#pragma once`) plus a CI step that regenerates and `diff`s.
README prose and its example updated (its printed values are currently wrong);
`exemple.cpp` updated to the new spellings.

---

## Order of work

0. **Spikes — done, except one comparison.** `nttp_probe.cpp` settled the NTTP
   question (see "Measured on clang 15 and 18") and `codegen_probe.cpp` settled
   the emitted code (table under "Guaranteed elimination"). Both compile and run
   clean on gcc 13 and clang 20 at `-O2`/`-O3`. Two things remain: add the
   current `addget` implementation to `codegen_probe.cpp` side by side to confirm
   the NTTP form is byte-identical for the units actually used in
   `ComputeEos.cpp` and `shamphys/*`; and re-run `nttp_probe.cpp` in the
   `ghcr.io/shamrock-code/shamrock-ci:ubuntu22` (clang 15) container before
   merge, since that leg is the reason for the reference spelling. If the NTTP
   form is not byte-identical, stop and revisit before writing real code.
1. **Golden-value capture, landed first, against the current code.**
   `src/tests/shamunits/legacy_reference_Tests.cpp` printing `get<u>()`,
   `get<u,2>()`, `get<u,-2>()` (the *old* signature — this test is written
   against the current code) for all 38 units and every `Constants<f64>`
   accessor at powers 1 and 2, in SI and in the astro system (Myr/au/M☉). Paste
   the numbers back as `REQUIRE_FLOAT_EQUAL` literals. `src/tests/CMakeLists.txt`
   globs `*.cpp`, so no CMake edit. This is the whole safety net for retyping 38
   units — but **the five bugs must be corrected in the golden table as they are
   fixed**, one comment per changed line, not enshrined.
2. `Unit.hpp` + `details/registry.hpp` + `static_assert`-only tests. Nothing
   includes them yet; zero risk.
3. `unit_table.hpp` + `prefix_table.hpp`, cross-checking every entry against the
   old `addget` body one at a time — this is where transcription errors will be.
4. Rewrite `UnitSystem.hpp`; update the `pyUnits.cpp` lookup lines. **The risky
   step.** Golden test must pass except the prefix fixes (bugs 3 and 4).
5. `constant_table.hpp` + rewrite `Constants.hpp`; delete `Si`,
   `ConversionConstants.hpp`, `Names.hpp` and the doxygen walls; update the ~16
   call sites.
6. Bindings loop, README, exemple, godbolt generator + CI diff, Python test.

## Verification

- **Build**: `cd build && ./shamenv_do shammake shamrock_test && echo DONE`;
  full `./shamenv_do shammake` before running tests.
- **C++ tests** — after `test -d reference-files || ./shamenv_do pull_reffiles`,
  and asking which device to use from `./shamenv_do ./shamrock_test --smi`:
  `./shamenv_do ./shamrock_test --sycl-cfg <id>:<id> --loglevel 1 --unittest`.
  New coverage:
  - golden values for `f32` **and** `f64`, **relative** tolerance (`f64` 1e-13,
    `f32` 1e-5), never bit-exact: the composed `si_factor` is now folded once in
    `double` and the multiplications re-associate, so ≤1 ulp drift is expected
    and is strictly *more* accurate for `f32`;
  - dimension `static_assert`s, and constexpr-ness `static_assert`s on
    `si.get<au_sq>()` and `Constants{si}.get<constants::G>()` — this is the
    compile-time guarantee, and it fails the build if lost;
  - round-trip `get(upow(u,p)) * to(upow(u,p)) ≈ 1` for every unit,
    `p ∈ [-3,3]`, across SI, astro and an adversarial system (seven base units
    set to distinct primes), **including a prefixed unit** — this is what pins
    bugs 3 and 4;
  - registry-name round-trip plus
    `REQUIRE_EXCEPTION_THROW(unit_from_name("nonsense"), std::invalid_argument)`;
  - all three entry points agree: for every named unit and `p ∈ [-3,3]`,
    `get<u_p>() == get(upow(u,p)) == get(u,p)` within tolerance, for
    `UnitSystem<f32>` and `<f64>`; and `get<f32>(u)` from an `f64` system
    matches `f32(get<f64>(u))` — this pins the "no hidden `double` arithmetic"
    claim;
  - dimension algebra: `upow(u,a) * upow(u,b) == upow(u,a+b)`, and
    `get(metre / upow(second,2))` equals `get(metre) * get(upow(second,-2))` —
    the composition a scalar `power` could not express.
- **Codegen** (the answer to "are the unused ones removed?"): re-run the step-0
  comparison against the final headers and record the result in the PR — one
  multiply for `get<units::hertz>()`, nothing for a `constexpr` result, and the
  honest instruction count for the runtime form. Worth keeping as a short
  `doc/` note next to the README's zero-cost claim, which is currently
  unsubstantiated.
- **Python**: `examples/physics/run_simple_units_usage.py` must run unchanged;
  add a script under `examples/tests_ci/` (wired into
  `shamrock-acpp-phys-test.yml`) asserting `usys.get("m")`, `usys.get("year")`,
  `usys.get("s", power=-1)`, `usys.get("m", 1, "kilo") == 1e3`,
  `usys.to("yr", pref="M")` (the Myr case, bug 4), and every accessor the
  examples use — `.au() .sol_mass() .year() .second() .kb() .dalton() .G() .c()
  .mu_0()` — with and without `power=`.
- **Lint**: `pre-commit run --all-files`; tables need `// clang-format off/on`
  guards, as `Names.hpp` already uses.
- **Docs**: confirm the `doxygen_warn_main` job (`main_workflow.yml:179`) does
  not regress once the `\fn` walls become per-declaration `///`.
- **Compilers**: CI covers clang 15, 18, 20 (`.github/workflows/`). The
  reference-NTTP form was probed on clang 15 and 18 and passes on both; the
  by-value variant stays commented out because clang < 18 has no floating-point
  NTTP. Nothing else in the design is version-sensitive, but re-check the
  clang-15 leg before merge — this container only has clang 20.
- **Follow-up when clang < 18 is dropped**: uncomment the by-value `get<Unit u>`
  overload, which then allows `get<upow(units::metre, 2)>()` inline and lets the
  named `au_sq`-style constants go away. Purely additive; worth an issue so it
  is not forgotten.

## Out of scope

Strongly-typed `Quantity<Dimension>` values (SYCL kernels operate on raw
`Tscal`), affine units (°C/°F), and rewriting `UnitHelper.py`'s hand-rolled name
table — though its Myr/Gyr scaling silently becomes correct once bug 4 is fixed,
which the PR must call out as a user-visible change.
