# sph_h: Lean 4 formalization of Proposition 1 (SPH smoothing-length residual)

This is a Lean 4 + Mathlib proof of Proposition 1 (Section 2.1) of the report *Smoothing length
iteration in SPH: robust and neighbour-limited update strategies*. It contains no `sorry`,
`admit` or new axioms.

## Toolchain (pinned)

| | |
|---|---|
| Lean | `leanprover/lean4:v4.36.0-rc1` (`lean-toolchain`) |
| Mathlib | tag `v4.36.0-rc1`, commit `625d8914199bc567cca9b91f90e8e5afafa5ae51` (`lakefile.toml`, `lake-manifest.json`) |

## Build

```bash
# install elan once: https://github.com/leanprover/elan
lake exe cache get    # download prebuilt Mathlib
lake build            # builds the SphH library and checks every proof
lake env lean Axioms.lean   # prints the axioms of each main theorem
```

Every theorem listed in `Axioms.lean` depends only on `propext`, `Classical.choice` and
`Quot.sound`.

## Setup

One particle of mass `m > 0` with `n` neighbours at distances `r : Fin n → ℝ`, `r i > 0`;
`N_part = n + 1`, `h_fact = 1.2`, `cnorm = π · 1.2³`.

* `w q = (1/4)·max(2−q,0)³ − max(1−q,0)³`, `dw q = −(3/4)·max(2−q,0)² + 3·max(1−q,0)²`
  (`SphH/Kernel.lean`; `w_eq_piecewise` checks it against the usual piecewise M4 form).
* `g r h = (1 + Σ w(r_i/h)) / cnorm`, `dg r h = Σ (−dw(r_i/h))·r_i/h² / cnorm`.
* `rhoH m h = m (1.2/h)³`, `rhoSum m r h = m (1 + Σ w(r_i/h)) / (π h³)`,
  `f = rhoSum − rhoH`, `df = −3f/h + rhoH·dg` (`SphH/Defs.lean`).

## What is proven

| Statement | Lean name | File |
|---|---|---|
| `w` has derivative `dw`; `w(0)=1`; `w'<0` on `(0,2)`; `w=w'=0` on `[2,∞)` | `hasDerivAt_w`, `w_zero`, `dw_neg`, `w_eq_zero`, `dw_eq_zero` | Kernel |
| `5 < 1.728π < 6` | `five_lt_cnorm_lt_six` | Deriv |
| `f = ρ_h (g − 1)` | `f_eq` | Deriv |
| `g' = dg` and `f' = df` are true derivatives (for `h ≠ 0`) | `hasDerivAt_g`, `hasDerivAt_f` | Deriv |
| Eq. (2): `f' = Σ_b m ∂_h W(r_ab,h) + 3ρ_h/h` | `df_eq_eq2` (with `hasDerivAt_W`) | Deriv |
| (i-a) `g` monotone on `(0,∞)` | `monotoneOn_g` | Prop1 |
| (i-a) `g` strictly monotone on `{h \| ∃ i, r_i < 2h}`, which is `(min r_i/2, ∞)` if `n ≥ 1` | `strictMonoOn_g`, `setOf_exists_lt_eq_Ioi` | Prop1 |
| (i-b) `g h = 1/cnorm` for `0 < h ≤ min r_i/2`, and near `0⁺` | `g_eq_of_forall_le`, `g_eq_of_le_min`, `eventually_g_eq_nhdsGT` | Prop1 |
| (i-b) `g → (n+1)/cnorm` as `h → ∞` | `tendsto_g_atTop` | Prop1 |
| (i-c) `(∃ h > 0, g h = 1) ↔ 6 ≤ n + 1` | `exists_root_iff` | Prop1 |
| (i-c) the root is unique, `f < 0` below it and `f > 0` above it | `root_unique`, `f_neg_of_lt_root`, `f_pos_of_root_lt`, `existsUnique_root` | Prop1 |
| (ii) `f' ≥ −3f/h`; `f < 0 → f' > 0` | `df_ge`, `df_pos_of_f_neg` | Prop1 |
| (ii-b) `f' ≤ 0 → f > 0` | `f_pos_of_df_nonpos` | Prop1 |
| (iii) `f > 0 ∧ f' < 0 → h − f/f' ≥ 4h/3` | `nr_ge` (and `nr_ge_of_df_neg`) | Prop1 |
| (iii-sharp) if `n + 1 ≥ 6`: eventually `f > 0 ∧ f' < 0`, and `(h − f/f')/h → 4/3` | `nr_tendsto` | Prop1 |
| Generic kernel, `d` dimensions, unequal masses: (ii), (ii-b), and `h − f/f' ≥ (1 + 1/d) h` | `Generic.Setup.hasDerivAt_f`, `df_ge`, `df_pos_of_f_neg`, `f_pos_of_df_nonpos`, `nr_ge` | Generic |
| The M4, `d = 3` case as an instance of the generic one | `Generic.m4_f`, `m4_df`, `m4_nr_ge`, `m4_f_pos_of_df_nonpos` | Generic |

## Differences from the report's wording

* **(ii-b)**: strict monotonicity of `g` alone does not give `g'(h⋆) > 0` at the root. The proof
  uses the explicit `g'`: at the root `g = 1 > 1/cnorm`, so some `r_i < 2h⋆`, and that term of
  `g'` is `> 0` (`exists_lt_of_g_eq_one`, `dg_pos`).
* **(iii)**: the hypothesis `f > 0` is not needed. `f' < 0` alone gives `h − f/f' ≥ 4h/3`
  (`nr_ge_of_df_neg`), and (ii-b) already turns `f' < 0` into `f > 0`.
* **Eq. (2)**: the identity holds with the sum over *all* particles `b`, self term `b = a`
  (`r_aa = 0`, `∂_h W(0,h) = −3/(π h⁴)`) included.
* **Generic version**: (ii) and (iii) need `K` differentiable with `K' ≤ 0` on `[0,∞)`
  (the report's remark leaves differentiability out), `C, h_fact, m_a > 0`, `m_b ≥ 0`, `d ≥ 1`.
  Compact support is not needed for them. Generic (ii-b) also needs:
  support in `[0,R)` with `K' < 0` on `(0,R)`, `m_b > 0`, **and** `C·K(0) ≠ h_fact^d`, meaning
  the self term alone is not a root. Without this last condition the statement fails: if no
  neighbour lies within `R h` and `C·K(0) = h_fact^d`, then `f = f' = 0`. For M4 this
  holds because `π·1.2³ ≠ 1`.
* Lean's convention `x / 0 = 0` is never used for the results. All analytic statements assume
  `h > 0` (derivatives: `h ≠ 0`).
