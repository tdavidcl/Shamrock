import SphH.Defs

/-!
# Basic facts and derivatives of `g`, `ρ_h` and `f`
-/

noncomputable section

open Real Set Filter Topology

namespace SphH

variable {n : ℕ}

theorem hfact_pos : 0 < hfact := by norm_num [hfact]

theorem cnorm_eq : cnorm = 1.728 * π := by norm_num [cnorm, hfact]; ring

theorem cnorm_pos : 0 < cnorm := by rw [cnorm_eq]; positivity

/-- The key numeric fact `5 < 1.728 π < 6`. -/
theorem five_lt_cnorm_lt_six : 5 < cnorm ∧ cnorm < 6 := by
  rw [cnorm_eq]
  constructor
  · nlinarith [pi_gt_three]
  · nlinarith [pi_lt_d2]

theorem one_lt_cnorm : 1 < cnorm := by linarith [five_lt_cnorm_lt_six.1]

theorem rhoH_pos {m h : ℝ} (hm : 0 < m) (hh : 0 < h) : 0 < rhoH m h := by
  unfold rhoH; have := hfact_pos; positivity

/-- `ρ_sum = ρ_h · g` (also at `h = 0`, by Lean's `x / 0 = 0` convention). -/
theorem rhoSum_eq (m : ℝ) (r : Fin n → ℝ) (h : ℝ) : rhoSum m r h = rhoH m h * g r h := by
  rcases eq_or_ne h 0 with rfl | hh
  · simp [rhoSum, rhoH]
  · have := cnorm_pos.ne'
    have := pi_pos.ne'
    have := hfact_pos.ne'
    unfold rhoSum rhoH g cnorm at *
    field_simp

/-- The identity `f = ρ_h (g − 1)`. -/
theorem f_eq (m : ℝ) (r : Fin n → ℝ) (h : ℝ) : f m r h = rhoH m h * (g r h - 1) := by
  rw [f, rhoSum_eq]; ring

theorem hasDerivAt_g (r : Fin n → ℝ) {h : ℝ} (hh : h ≠ 0) : HasDerivAt (g r) (dg r h) h := by
  have hterm : ∀ i ∈ (Finset.univ : Finset (Fin n)),
      HasDerivAt (fun y => w (r i / y)) (-dw (r i / h) * r i / h ^ 2) h := by
    intro i _
    have hd := (hasDerivAt_w (r i / h)).comp h ((hasDerivAt_inv hh).const_mul (r i))
    convert hd using 1
    · ext y; simp [Function.comp, div_eq_mul_inv]
    · field_simp
  exact ((HasDerivAt.fun_sum hterm).const_add 1).div_const cnorm

theorem hasDerivAt_rhoH (m : ℝ) {h : ℝ} (hh : h ≠ 0) :
    HasDerivAt (rhoH m) (-3 * rhoH m h / h) h := by
  have hd := (((hasDerivAt_inv hh).const_mul hfact).pow 3).const_mul m
  convert hd using 1
  · ext y; simp [rhoH, div_eq_mul_inv]
  · have hinv : h⁻¹ ^ 2 * h ^ 2 = 1 := by rw [← mul_pow, inv_mul_cancel₀ hh, one_pow]
    simp only [rhoH]; field_simp; linear_combination (3 * m * hfact ^ 3) * hinv

/-- (ii), first part: `f'` is the true derivative of `f`. -/
theorem hasDerivAt_f (m : ℝ) (r : Fin n → ℝ) {h : ℝ} (hh : h ≠ 0) :
    HasDerivAt (f m r) (df m r h) h := by
  have hd := (hasDerivAt_rhoH m hh).mul ((hasDerivAt_g r hh).sub_const 1)
  have hf : f m r = fun y => rhoH m y * (g r y - 1) := funext (f_eq m r)
  rw [hf]
  convert hd using 1
  rw [df, f_eq]; ring

theorem continuousOn_g (r : Fin n → ℝ) : ContinuousOn (g r) (Ioi 0) := fun _ hh =>
  (hasDerivAt_g r (ne_of_gt hh)).continuousAt.continuousWithinAt

/-- `∂W/∂h` is the true derivative of `W`. -/
theorem hasDerivAt_W (r : ℝ) {h : ℝ} (hh : h ≠ 0) :
    HasDerivAt (fun y => W r y) (dWdh r h) h := by
  have hw := (hasDerivAt_w (r / h)).comp h ((hasDerivAt_inv hh).const_mul r)
  have hp := ((hasDerivAt_pow 3 h).const_mul π)
  have hd := hw.div hp (by have := pi_pos; positivity)
  convert hd using 1
  · ext y; simp [W, Function.comp, div_eq_mul_inv]
  · have := pi_pos.ne'
    simp only [dWdh, Function.comp]
    field_simp
    ring

/-- The report's Eq. (2): `f' = Σ_b m ∂_h W(r_ab, h) + 3 ρ_h / h`, the sum running over all
particles `b` including the self term `b = a` (`r_aa = 0`). -/
theorem df_eq_eq2 (m : ℝ) (r : Fin n → ℝ) {h : ℝ} (hh : h ≠ 0) :
    df m r h = (m * dWdh 0 h + ∑ i, m * dWdh (r i) h) + 3 * rhoH m h / h := by
  have := pi_pos.ne'
  have := hfact_pos.ne'
  have hterm : ∀ i, m * dWdh (r i) h =
      (-3 * m / (π * h ^ 4)) * w (r i / h) + (m / (π * h ^ 3)) * (-dw (r i / h) * r i / h ^ 2) := by
    intro i; simp only [dWdh]; field_simp; ring
  simp only [hterm, Finset.sum_add_distrib, ← Finset.mul_sum]
  simp only [df, f, rhoSum, rhoH, dg, dWdh, cnorm, w_zero, zero_div, dw_zero]
  set S := ∑ i, w (r i / h)
  set S2 := ∑ i, -dw (r i / h) * r i / h ^ 2
  clear_value S S2
  field_simp
  ring

end SphH
