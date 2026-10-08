# How this design was reached

Session log for the `shamunits` redesign, kept as a decision record rather than
a transcript. Every fork below was decided by the user; the rationale is kept so
the choices can be re-examined without re-deriving them.

## The request

> Redesign the unit system using features available in C++20 such that it is
> still constexpr without code duplication and no macros (unless it requires too
> much code duplication) that would be more modern and consistent (no scattering
> of the constant definition for example). In particular I want the users to be
> able to trivially add new units & constants.

## What the survey found

`src/shamunits/` is ~1600 lines across five headers. Its data is restated in
**six** hand-maintained places:

| What | Restated in |
| --- | --- |
| 38 units | `XMAC_UNITS` enum (`Names.hpp`) · 38 `addget(...)` bodies (`UnitSystem.hpp`) · 38 `case` labels in `getter_1` · dimension only in a C comment · `doc/godbolt.cpp` |
| 13 prefixes | `XMAC_UNIT_PREFIX` · 13 `case` labels in `getter_2` |
| 16 conversion factors | `ConversionConstants.hpp` + a 47-line doxygen block |
| 35 constants | `UNITS_CONSTANTS` (dimension) · `Constants<T>::Si` (numeric value, 50 lines away) · ~210 lines of `\fn` doxygen · `pyUnits.cpp` |

Adding one unit means editing four places; one constant, four or five. Nothing
machine-checks that the dimension column agrees with the value column.

There are **no C++ tests** for the module. `.codecov.yml` declares a `shamunits`
component over `src/shamunits/**`, currently uncovered.

### Five defects the duplication has already produced

1. `Constants.hpp:48,79` — `h` and `hbar` registered as **J·s⁻¹**; they are J·s.
2. `Constants.hpp:76` — `guiness_density` registered as **kg·m⁻¹**; its value is kg·m⁻³.
3. `UnitSystem.hpp:233` — `to<pref,u,power>()` forwards to `get<u,-power>()`,
   which resolves to the no-prefix overload and **silently drops the prefix**.
4. **The prefix is applied once per nesting level, not once per unit.**
   `addget(years)` is `PREF * Uget(s,1) * Cget(yr_to_s,1)`, and `Uget(s,1)`
   re-applies `PREF`, so `get<mega, units::years>()` yields `1e12·yr_to_s`
   instead of `1e6·yr_to_s`. `Joule` nests three deep, so `get<kilo, Joule>()`
   is off by 10⁶.
5. `doc/godbolt.cpp` is a stale hand-copied fork of the whole library (missing
   `sigma`, `kb`, `dalton`, `solar_radius`, `earth_radius`, `guiness_density`;
   `hbar` hardcoded; `addconstant(au)` multiplies by seconds instead of metres).

Bug 4 was confirmed independently against the library's **own README output**,
which prints `to<units::second>() = 3.15576e+19` where 1 Myr is `3.15576e13` s,
and `G = 3.94781e+25` where the correct value is `3.947813e13` — off by 10⁶ and by
10⁶ squared, G carrying s⁻². It reaches production:
`src/pylib/shamrock/utils/analysis/UnitHelper.py` uses `to("yr", pref="M")` and
`pref="G"` for Myr/Gyr axis scaling, so those plot labels are wrong by 10⁶.

### The compile-time claim, corrected

The README says "almost everything is marked `constexpr` … zero cost
abstraction". In the language sense the current library is **not** compile-time
evaluated: `UnitSystem`'s constructor is not `constexpr` (`UnitSystem.hpp:141`),
so `constexpr UnitSystem<double> si{};` does not compile today. What godbolt
shows is optimizer constant folding — everything inlines and
`pow_constexpr_fast_inv<power>` is an `if constexpr` ladder that collapses
before the optimizer runs. The guarantee is "LLVM/GCC did it", not "the standard
says so". This matters because the redesign can make it a real, testable
property, which is strictly more than the status quo offers.

## Decisions, in the order they were taken

### 1. Who must be able to add units — Shamrock devs, one line in a table

Rejected: downstream C++ headers declaring their own units; runtime registration
from Python. Neither was needed, and both constrain the design.

### 2. No macros at all

Offered: one X-macro list per category (1 line per entry, generates everything
including the C++ `.G()` accessors), versus plain `inline constexpr Unit`
declarations plus a `constexpr std::array` registry (2 adjacent lines, no
macros).

**Chosen: no macros.** Consequence accepted: the 35 named C++ accessors
(`Constants{usys}.G()`) are dropped, because hand-writing them macro-free would
be three places per constant instead of one. Python keeps them all — the
bindings loop over the constant registry, so a new constant appears in Python
with no binding edit.

### 3. Python API and JSON keys frozen; C++ call sites free

~16 C++ call sites may change. The Python surface (`shamrock.UnitSystem(...)`,
`usys.get("m")`, `shamrock.Constants(usys).au()`, …) is used by ~20 example
scripts and `src/pylib/`, and the JSON keys (`unit_time`, `unit_length`, …) are
a persisted on-disk format. Both untouchable.

### 4. Fix all five bugs, and add dimension `static_assert`s

Accepted even where it changes numbers. The Myr/Gyr plot scaling in
`UnitHelper.py` silently becomes correct; the PR must say so.

### 5. "Would that still be evaluated at compile time?"

The user asked whether the README example would still fold. This produced the
correction above (it was never a language guarantee) and the decision to make it
one: mark the constructor `constexpr`, then `constexpr double x = si.get<u>();`
and `static_assert` are available and CI-checked.

Chosen at this fork: **by-value `get` + `constexpr` proof**, over a templated
NTTP as the primary API. Also chosen: replace `.G()` with `get(constants::G)`
rather than keeping the named accessors.

### 6. "The runtime overload always pays a division the current path doesn't"

Correct, and worse than just the division: with `power` a plain `int`,
`Tret(1.0 / u.si_factor)` is evaluated eagerly as a function argument even when
`power` is positive and the reciprocal is never used. The current `Cget(cst, n)`
takes `1/cst` from a `static constexpr`, so it is a compile-time constant.

**Chosen: `power` becomes a template parameter.** The base dimensions were never
the problem — `s_inv`, `m_inv`… are stored members, so no division there either.

### 7. "We don't need `power` at all — the unit can carry it"

`upow(metre, 2)` *is* a `Unit`. Adopted, and it turned out to be the stronger
design for a reason beyond tidiness: a scalar `power` can only scale every
exponent uniformly, so `m·s⁻²` is unreachable as `u^p`, while
`metre / upow(second, 2)` is trivial. It also removes the division question
outright, since `upow` folds the `si_factor` exponentiation at compile time.

This exposed a real constraint: `get<upow(metre,2)>()` passes a **prvalue**,
which cannot bind to a `const Unit&` NTTP. That spelling requires by-value
`template<Unit u>`, and `Unit` holds a `double` — so it depends on C++20
floating-point NTTP support.

### 8. Measurement, not guesswork

CI builds on **clang 15** (`shamrock-acpp-clang-py.yml`, ubuntu22 container)
alongside 18. The repo has no existing class-type or floating-point NTTP to
learn from — every current NTTP is an enum. So `nttp_probe.cpp` was written and
the user ran it.

Result, in two rounds:

- **`template<double D>` is rejected by clang < 18.** So is any NTTP whose type
  is a class holding a `double`, i.e. by-value `template<Unit u>`.
- First read of this was that the NTTP had to be dropped entirely, and the plan
  was briefly rewritten that way. That was wrong: the probe's floating-point
  check was unconditional, so it aborted the translation unit before the
  reference variant was ever reached.
- With that check disabled, **`template<const Unit &u>` compiles on clang 15**
  and the whole design passes. A reference NTTP has *reference* type, which is
  structural regardless of what it refers to.
- On **clang ≥ 18** the by-value form compiles too.

**Chosen: ship the reference NTTP.** It keeps the guaranteed elimination of
unused dimensions on every supported compiler. The only thing deferred is the
inline spelling — a composed unit must be named first, which costs one line in
the `au²` example and nothing else, since every real call site names a plain
unit. The by-value overload stays in the header commented out, to enable when
clang < 18 is dropped (expected within about a year). Enabling it is purely
additive: `get<u>()` means the same thing either way, so no call site moves.

> Lesson for the next probe: give each feature check its own `#if`, so one
> failure cannot mask the answer to a different question.

## Worked example, old API to new

The README's own example, translated:

```cpp
using namespace shamunits;

constexpr UnitSystem<double> si{};

// Composed units need a name — the NTTP binds by reference
inline constexpr Unit megayear = prefix::mega * units::year;
inline constexpr Unit au_sq    = upow(units::astronomical_unit, 2);

constexpr UnitSystem<double> astro_units{
    si.get<megayear>(),                  // unit_time   in s
    si.get<units::astronomical_unit>(),  // unit_length in m
    si.get<constants::sol_mass>(),       // unit_mass   in kg
};

std::cout << astro_units.get<au_sq>() << std::endl;   // 1
```

| Old | New |
| --- | --- |
| `Constants<double>(si).sol_mass()` | `si.get<constants::sol_mass>()` |
| `si.get<mega, units::years>()` | `si.get<megayear>()` |
| `si.get<units::astronomical_unit>()` | unchanged |
| `si.get<units::kilogram>() * sol_mass` | folded into `si.get<constants::sol_mass>()` |
| `astro_units.get<units::astronomical_unit, 2>()` | `astro_units.get<au_sq>()` |

Three notes on that translation:

- **The mass line collapses two calls into one.** The old
  `si.get<units::kilogram>() * sol_mass` evaluates to `Si::sol_mass * kg_val²`
  and only works because `si` is SI, where the stray `kg_val` is 1. Since
  `constants::sol_mass` carries `dim = kg¹`, the new form stays correct for a
  non-SI system.
- **The time line is where bug 4 shows.** `3.15576e19` today versus `3.15576e13`
  after. Anyone re-running the example should expect that and `G` to move.
- **Everything is `constexpr` now**, so the last line can be `static_assert`ed
  instead of eyeballed — but **not** with `==`. `au²` is `2.238e22`, past 2⁵³,
  and `1/au` is inexact, so the round trip lands on 1 to within a few ulps.
  Use a relative tolerance.

A composed query such as Mega au · yr⁻¹ shows why the scalar `power` had to go —
the prefix attaches to `au` only and the two factors carry exponents +1 and −1:

```cpp
inline constexpr Unit mega_au_per_year
    = prefix::mega * units::astronomical_unit / units::year;

constexpr double v_si    = si.get<mega_au_per_year>();          // 4.74047e9 (m.s-1)
constexpr double v_astro = astro_units.get<mega_au_per_year>(); // ~1e12     (au.Myr-1)
```

`1 Mau/yr` is `1e6 au/yr`, i.e. `1e12 au/Myr`. Under the old API this had to be
assembled from two `runtime_get` calls and multiplied by hand — and the
`pref="M"` leg would hit bug 4 and come out 10⁶ too large.

## Codegen, measured after the fact

Both probes were later compiled and run on gcc 13 and clang 20 in the session
container. Results in `03-compiler-probes.md`; two things worth carrying
forward:

- **The elimination works.** `test_func` keeps exactly one division out of the
  seven the constructor writes, and a single-dimension unit through the NTTP
  (`probe_hertz`) compiles to **one `movsd`** with zero arithmetic — better than
  the "one multiply" the plan predicted, because the `*1.0` factors fold out.
- **A claim in the plan was wrong and is now corrected.** It said the
  value-argument form would fold "on both compilers without fast-math". gcc
  does; **clang does not**, at `-O2` or `-O3`, even with a literal argument — it
  spills the `Unit` and emits a real call. So the NTTP form is the one to use
  wherever the unit is statically known, not merely the tidier option.
- `G` in (Myr, au, M☉) measures `3.947813e+13`, exactly the current library's
  `3.94781e+25` over 10¹² — bug 4 confirmed to the digit.

## Still open

- One comparison: add the current `addget` implementation to
  `codegen_probe.cpp` side by side, to confirm the NTTP form is byte-identical
  for the units actually used in `ComputeEos.cpp` and `shamphys/*`.
- Re-run `nttp_probe.cpp` in the clang-15 CI container before merge. This
  container has only clang 20, so the clang-15 result above is the user's
  measurement, not a reproducible one here.
