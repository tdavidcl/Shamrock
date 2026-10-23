import SphH.Prop1

/-!
# Generic version: any differentiable kernel, `d` dimensions, unequal masses

Kernel `W(r, h) = C K(r/h) / h^d`. Particle `a` has mass `ma`; neighbour `i` has mass `mb i`
at distance `r i`. With `ρ_h = ma (h_fact/h)^d` and `ρ_sum = Σ_b m_b W(r_ab, h)` (self term
included), `g = ρ_sum / ρ_h = C (K 0 + Σ (mb i / ma) K(r_i/h)) / h_fact^d`.
-/

noncomputable section

open Real Set Filter Topology

namespace SphH.Generic

/-- Data of the generic problem. -/
structure Setup (n : ℕ) where
  /-- Dimension. -/
  d : ℕ
  /-- Unnormalized kernel `K` and its derivative `K'`. -/
  K : ℝ → ℝ
  K' : ℝ → ℝ
  /-- Kernel normalisation: `W(r, h) = C K(r/h) / h^d`. -/
  C : ℝ
  hfact : ℝ
  ma : ℝ
  mb : Fin n → ℝ
  r : Fin n → ℝ

/-- Hypotheses used for (ii) and (iii). -/
structure Setup.Hyp {n : ℕ} (S : Setup n) : Prop where
  d_pos : 0 < S.d
  hasDerivAt_K : ∀ q, HasDerivAt S.K (S.K' q) q
  K'_nonpos : ∀ q, 0 ≤ q → S.K' q ≤ 0
  C_pos : 0 < S.C
  hfact_pos : 0 < S.hfact
  ma_pos : 0 < S.ma
  mb_nonneg : ∀ i, 0 ≤ S.mb i
  r_pos : ∀ i, 0 < S.r i

variable {n : ℕ} (S : Setup n)

namespace Setup

def g (h : ℝ) : ℝ := S.C * (S.K 0 + ∑ i, S.mb i / S.ma * S.K (S.r i / h)) / S.hfact ^ S.d

def dg (h : ℝ) : ℝ :=
  S.C * (∑ i, S.mb i / S.ma * (-S.K' (S.r i / h) * S.r i / h ^ 2)) / S.hfact ^ S.d

def rhoH (h : ℝ) : ℝ := S.ma * (S.hfact / h) ^ S.d

def rhoSum (h : ℝ) : ℝ := S.C / h ^ S.d * (S.ma * S.K 0 + ∑ i, S.mb i * S.K (S.r i / h))

def f (h : ℝ) : ℝ := S.rhoSum h - S.rhoH h

def df (h : ℝ) : ℝ := -S.d * S.f h / h + S.rhoH h * S.dg h

variable {S}

theorem rhoH_pos (hS : S.Hyp) {h : ℝ} (hh : 0 < h) : 0 < S.rhoH h := by
  have := hS.ma_pos; have := hS.hfact_pos
  unfold rhoH; positivity

theorem rhoSum_eq (hS : S.Hyp) {h : ℝ} (hh : h ≠ 0) : S.rhoSum h = S.rhoH h * S.g h := by
  have := hS.ma_pos.ne'; have := hS.hfact_pos.ne'
  have hsum : ∑ i, S.mb i * S.K (S.r i / h) = S.ma * ∑ i, S.mb i / S.ma * S.K (S.r i / h) := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun i _ => by field_simp
  simp only [rhoSum, rhoH, g, div_pow]
  rw [hsum]
  set T := ∑ i, S.mb i / S.ma * S.K (S.r i / h)
  clear_value T
  field_simp

theorem f_eq (hS : S.Hyp) {h : ℝ} (hh : h ≠ 0) : S.f h = S.rhoH h * (S.g h - 1) := by
  rw [f, rhoSum_eq hS hh]; ring

theorem hasDerivAt_g (hS : S.Hyp) {h : ℝ} (hh : h ≠ 0) : HasDerivAt S.g (S.dg h) h := by
  have hterm : ∀ i ∈ (Finset.univ : Finset (Fin n)),
      HasDerivAt (fun y => S.mb i / S.ma * S.K (S.r i / y))
        (S.mb i / S.ma * (-S.K' (S.r i / h) * S.r i / h ^ 2)) h := by
    intro i _
    have hd := ((hS.hasDerivAt_K (S.r i / h)).comp h
      ((hasDerivAt_inv hh).const_mul (S.r i))).const_mul (S.mb i / S.ma)
    convert hd using 1
    · ext y; simp [Function.comp, div_eq_mul_inv]
    · field_simp
  exact (((HasDerivAt.fun_sum hterm).const_add (S.K 0)).const_mul S.C).div_const _

theorem hasDerivAt_rhoH {h : ℝ} (hh : h ≠ 0) :
    HasDerivAt S.rhoH (-S.d * S.rhoH h / h) h := by
  have hd := (((hasDerivAt_inv hh).const_mul S.hfact).pow S.d).const_mul S.ma
  convert hd using 1
  · ext y; simp [rhoH, div_eq_mul_inv]
  · simp only [rhoH]
    rcases Nat.eq_zero_or_pos S.d with h0 | hpos
    · simp [h0]
    · obtain ⟨k, hk⟩ : ∃ k, S.d = k + 1 := ⟨S.d - 1, by omega⟩
      rw [hk, Nat.add_sub_cancel, pow_succ, div_eq_mul_inv S.hfact h]
      push_cast
      field_simp

/-- Generic (ii): `f'` is the derivative of `f`. -/
theorem hasDerivAt_f (hS : S.Hyp) {h : ℝ} (hh : h ≠ 0) : HasDerivAt S.f (S.df h) h := by
  have hd := (hasDerivAt_rhoH (S := S) hh).mul ((hasDerivAt_g hS hh).sub_const 1)
  have hev : (fun y => S.rhoH y * (S.g y - 1)) =ᶠ[𝓝 h] S.f := by
    filter_upwards [isOpen_ne.mem_nhds hh] with y hy
    exact (f_eq hS hy).symm
  refine (hd.congr_of_eventuallyEq hev.symm).congr_deriv ?_
  rw [df, f_eq hS hh]; ring

theorem dg_nonneg (hS : S.Hyp) {h : ℝ} (hh : 0 < h) : 0 ≤ S.dg h := by
  have := hS.C_pos; have := hS.hfact_pos; have := hS.ma_pos
  unfold dg
  apply div_nonneg _ (by positivity)
  apply mul_nonneg hS.C_pos.le
  apply Finset.sum_nonneg fun i _ => ?_
  have h1 := hS.K'_nonpos _ (div_nonneg (hS.r_pos i).le hh.le)
  have := hS.r_pos i
  have := hS.mb_nonneg i
  apply mul_nonneg (by positivity)
  apply div_nonneg _ (by positivity)
  nlinarith

/-- Generic (ii): `f' ≥ −d f / h`. -/
theorem df_ge (hS : S.Hyp) {h : ℝ} (hh : 0 < h) : -S.d * S.f h / h ≤ S.df h := by
  unfold df
  have := mul_nonneg (rhoH_pos hS hh).le (dg_nonneg hS hh)
  linarith

/-- Generic (ii): `f < 0 → f' > 0`. -/
theorem df_pos_of_f_neg (hS : S.Hyp) {h : ℝ} (hh : 0 < h) (hf : S.f h < 0) : 0 < S.df h := by
  have hd : (0 : ℝ) < S.d := by exact_mod_cast hS.d_pos
  have : 0 < -S.d * S.f h / h := div_pos (by nlinarith) hh
  linarith [df_ge hS hh]

/-- Generic (iii): if `f' < 0` (in particular if `f > 0` and `f' < 0`), the Newton–Raphson
iterate is at least `(1 + 1/d) h`. -/
theorem nr_ge (hS : S.Hyp) {h : ℝ} (hh : 0 < h) (hdf : S.df h < 0) :
    (1 + 1 / S.d) * h ≤ h - S.f h / S.df h := by
  have hd : (0 : ℝ) < S.d := by exact_mod_cast hS.d_pos
  have key := df_ge hS hh
  rw [div_le_iff₀ hh] at key
  have : S.f h / S.df h ≤ -h / S.d := by
    rw [div_le_iff_of_neg hdf, div_mul_eq_mul_div, div_le_iff₀ hd]; nlinarith
  have : (1 + 1 / S.d) * h = h + h / S.d := by ring
  rw [this, neg_div] at *
  linarith

/-- Generic (ii-b): `f' ≤ 0 → f > 0`. Besides (ii)'s hypotheses this needs a compact support
`[0, R)` with `K' < 0` on `(0, R)`, positive neighbour masses, and that the self term alone is
not a root (`C K(0) ≠ h_fact^d`); without the latter, `f = f' = 0` whenever no neighbour lies
within `R h`. -/
theorem f_pos_of_df_nonpos (hS : S.Hyp) {R : ℝ} (hK0 : ∀ q, R ≤ q → S.K q = 0)
    (hK' : ∀ q, 0 < q → q < R → S.K' q < 0) (hmb : ∀ i, 0 < S.mb i)
    (hself : S.C * S.K 0 ≠ S.hfact ^ S.d) {h : ℝ} (hh : 0 < h) (hdf : S.df h ≤ 0) :
    0 < S.f h := by
  rcases lt_trichotomy (S.f h) 0 with hf | hf | hf
  · linarith [df_pos_of_f_neg hS hh hf]
  · exfalso
    have hρ := rhoH_pos hS hh
    have hg : S.g h = 1 := by
      have := hf; rw [f_eq hS hh.ne'] at this
      have := (mul_eq_zero.1 this).resolve_left hρ.ne'
      linarith
    have hex : ∃ i, S.r i < R * h := by
      by_contra hcon
      push Not at hcon
      apply hself
      have hpow : 0 < S.hfact ^ S.d := pow_pos hS.hfact_pos _
      rw [g, Finset.sum_eq_zero fun i _ => by
        rw [hK0 _ ((le_div_iff₀ hh).2 (hcon i)), mul_zero], add_zero,
        div_eq_one_iff_eq hpow.ne'] at hg
      exact hg
    obtain ⟨i, hi⟩ := hex
    have hdg : 0 < S.dg h := by
      have := hS.C_pos; have := hS.hfact_pos; have := hS.ma_pos
      unfold dg
      apply div_pos _ (by positivity)
      apply mul_pos hS.C_pos
      refine Finset.sum_pos' (fun j _ => ?_) ⟨i, Finset.mem_univ _, ?_⟩
      · have h1 := hS.K'_nonpos _ (div_nonneg (hS.r_pos j).le hh.le)
        have := hS.r_pos j
        have := hS.mb_nonneg j
        apply mul_nonneg (by positivity)
        apply div_nonneg _ (by positivity)
        nlinarith
      · have h1 := hK' _ (div_pos (hS.r_pos i) hh) (by rw [div_lt_iff₀ hh]; exact hi)
        have := hS.r_pos i
        have := hmb i
        apply mul_pos (by positivity)
        apply div_pos _ (by positivity)
        nlinarith
    have := mul_pos hρ hdg
    simp only [df, hf, mul_zero, zero_div, zero_add] at hdf
    linarith
  · exact hf

end Setup

/-! ### The M4, `d = 3`, equal-mass case as an instance -/

/-- The concrete problem of `SphH.Prop1` as a generic setup. -/
def m4 (m : ℝ) (r : Fin n → ℝ) : Setup n where
  d := 3
  K := w
  K' := dw
  C := 1 / π
  hfact := SphH.hfact
  ma := m
  mb := fun _ => m
  r := r

theorem m4_hyp {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) : (m4 m r).Hyp where
  d_pos := by norm_num [m4]
  hasDerivAt_K := hasDerivAt_w
  K'_nonpos := fun _ => dw_nonpos
  C_pos := by simp [m4, pi_pos]
  hfact_pos := hfact_pos
  ma_pos := hm
  mb_nonneg := fun _ => hm.le
  r_pos := hr

theorem m4_f (m : ℝ) (r : Fin n → ℝ) : (m4 m r).f = SphH.f m r := by
  ext h
  simp only [Setup.f, Setup.rhoSum, Setup.rhoH, m4, SphH.f, SphH.rhoSum, SphH.rhoH, w_zero,
    ← Finset.mul_sum]
  ring

theorem m4_rhoH (m : ℝ) (r : Fin n → ℝ) : (m4 m r).rhoH = SphH.rhoH m := rfl

theorem m4_dg {m : ℝ} (hm : m ≠ 0) (r : Fin n → ℝ) : (m4 m r).dg = SphH.dg r := by
  ext h
  simp only [Setup.dg, SphH.dg, cnorm, m4, div_self hm, one_mul]
  ring

theorem m4_df {m : ℝ} (hm : m ≠ 0) (r : Fin n → ℝ) : (m4 m r).df = SphH.df m r := by
  ext h
  rw [Setup.df, SphH.df, m4_f, m4_rhoH, m4_dg hm]
  norm_num [m4]

/-- (iii) for the M4 kernel recovered from the generic bound `(1 + 1/d) h`, `d = 3`. -/
theorem m4_nr_ge {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h : ℝ}
    (hh : 0 < h) (hdf : SphH.df m r h < 0) : 4 * h / 3 ≤ h - SphH.f m r h / SphH.df m r h := by
  have := Setup.nr_ge (m4_hyp hm hr) hh (by rwa [m4_df hm.ne'])
  rw [m4_df hm.ne', m4_f] at this
  have h3 : (m4 m r).d = 3 := rfl
  rw [h3] at this
  norm_num at this
  linarith

/-- (ii-b) for the M4 kernel recovered from the generic statement. -/
theorem m4_f_pos_of_df_nonpos {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i)
    {h : ℝ} (hh : 0 < h) (hdf : SphH.df m r h ≤ 0) : 0 < SphH.f m r h := by
  have := Setup.f_pos_of_df_nonpos (m4_hyp hm hr) (R := 2) (fun _ => w_eq_zero)
    (fun _ => dw_neg) (fun _ => hm) ?_ hh (by rwa [m4_df hm.ne'])
  · rwa [m4_f] at this
  · simp only [m4, w_zero, mul_one]
    have := five_lt_cnorm_lt_six.1
    rw [cnorm] at this
    intro heq
    rw [one_div, inv_eq_iff_eq_inv] at heq
    have hp := pi_pos
    have : π * SphH.hfact ^ 3 = 1 := by rw [heq, inv_mul_cancel₀ (pow_ne_zero _ hfact_pos.ne')]
    linarith

end SphH.Generic
