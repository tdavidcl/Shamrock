import Mathlib

/-!
# The M4 cubic spline kernel

`w q = (1/4)·max(2−q,0)³ − max(1−q,0)³`, with derivative
`dw q = −(3/4)·max(2−q,0)² + 3·max(1−q,0)²`.
-/

noncomputable section

open Real Set Filter Topology

namespace SphH

/-- Unnormalized 3D M4 kernel, closed form. -/
def w (q : ℝ) : ℝ := (1 / 4) * max (2 - q) 0 ^ 3 - max (1 - q) 0 ^ 3

/-- Derivative of `w`. -/
def dw (q : ℝ) : ℝ := -(3 / 4) * max (2 - q) 0 ^ 2 + 3 * max (1 - q) 0 ^ 2

/-- The closed form agrees with the usual piecewise definition. -/
theorem w_eq_piecewise (q : ℝ) :
    w q = if q < 1 then 1 - (3 / 2) * q ^ 2 + (3 / 4) * q ^ 3
      else if q < 2 then (2 - q) ^ 3 / 4 else 0 := by
  unfold w
  split_ifs with h1 h2
  · rw [max_eq_left (by linarith), max_eq_left (by linarith)]; ring
  · rw [max_eq_left (by linarith), max_eq_right (by linarith)]; ring
  · rw [max_eq_right (by linarith), max_eq_right (by linarith)]; ring

theorem w_zero : w 0 = 1 := by norm_num [w]

theorem w_eq_zero {q : ℝ} (hq : 2 ≤ q) : w q = 0 := by
  unfold w; rw [max_eq_right (by linarith), max_eq_right (by linarith)]; ring

theorem dw_eq_zero {q : ℝ} (hq : 2 ≤ q) : dw q = 0 := by
  unfold dw; rw [max_eq_right (by linarith), max_eq_right (by linarith)]; ring

theorem dw_zero : dw 0 = 0 := by norm_num [dw]

theorem dw_neg {q : ℝ} (h0 : 0 < q) (h2 : q < 2) : dw q < 0 := by
  unfold dw
  rcases lt_or_ge q 1 with h1 | h1
  · rw [max_eq_left (by linarith), max_eq_left (by linarith)]; nlinarith
  · rw [max_eq_left (by linarith), max_eq_right (by linarith)]; nlinarith

theorem dw_nonpos {q : ℝ} (h0 : 0 ≤ q) : dw q ≤ 0 := by
  rcases eq_or_lt_of_le h0 with rfl | h0
  · rw [dw_zero]
  rcases lt_or_ge q 2 with h2 | h2
  · exact (dw_neg h0 h2).le
  · rw [dw_eq_zero h2]

/-- `x ↦ max x 0 ^ 3` has derivative `3 · max x 0 ^ 2` everywhere. -/
theorem hasDerivAt_posPart_cube (x : ℝ) :
    HasDerivAt (fun y : ℝ => max y 0 ^ 3) (3 * max x 0 ^ 2) x := by
  have hcont : Continuous (fun y : ℝ => max y 0 ^ 3) := by fun_prop
  have hcont' : Continuous (fun y : ℝ => 3 * max y 0 ^ 2) := by fun_prop
  refine hasDerivAt_of_hasDerivAt_of_ne' (x := 0) (fun y hy => ?_) hcont.continuousAt
    hcont'.continuousAt x
  rcases lt_or_gt_of_ne hy with hy | hy
  · have hev : (fun _ : ℝ => (0 : ℝ)) =ᶠ[𝓝 y] fun z => max z 0 ^ 3 := by
      filter_upwards [Iio_mem_nhds hy] with z hz
      rw [max_eq_right (le_of_lt hz)]; ring
    rw [max_eq_right hy.le]
    simpa using (hasDerivAt_const y (0 : ℝ)).congr_of_eventuallyEq hev.symm
  · have hev : (fun z : ℝ => z ^ 3) =ᶠ[𝓝 y] fun z => max z 0 ^ 3 := by
      filter_upwards [Ioi_mem_nhds hy] with z hz
      rw [max_eq_left (le_of_lt hz)]
    rw [max_eq_left hy.le]
    simpa using (hasDerivAt_pow 3 y).congr_of_eventuallyEq hev.symm

theorem hasDerivAt_w (q : ℝ) : HasDerivAt w (dw q) q := by
  have h2 := (hasDerivAt_posPart_cube (2 - q)).comp q ((hasDerivAt_id q).const_sub 2)
  have h1 := (hasDerivAt_posPart_cube (1 - q)).comp q ((hasDerivAt_id q).const_sub 1)
  have := (h2.const_mul (1 / 4)).sub h1
  convert this using 1
  · ext y; simp [w, Function.comp]
  · simp [dw]; ring

theorem differentiable_w : Differentiable ℝ w := fun q => (hasDerivAt_w q).differentiableAt

theorem continuous_w : Continuous w := differentiable_w.continuous

theorem continuous_dw : Continuous dw := by unfold dw; fun_prop

theorem deriv_w : deriv w = dw := funext fun q => (hasDerivAt_w q).deriv

theorem antitoneOn_w : AntitoneOn w (Ici 0) := by
  refine antitoneOn_of_deriv_nonpos (convex_Ici 0) continuous_w.continuousOn
    differentiable_w.differentiableOn fun x hx => ?_
  rw [interior_Ici] at hx
  rw [deriv_w]; exact dw_nonpos (le_of_lt hx)

theorem strictAntiOn_w : StrictAntiOn w (Icc 0 2) := by
  refine strictAntiOn_of_deriv_neg (convex_Icc 0 2) continuous_w.continuousOn fun x hx => ?_
  rw [interior_Icc] at hx
  rw [deriv_w]; exact dw_neg hx.1 hx.2

theorem w_le_one {q : ℝ} (hq : 0 ≤ q) : w q ≤ 1 :=
  w_zero ▸ antitoneOn_w (mem_Ici.2 le_rfl) (mem_Ici.2 hq) hq

theorem w_nonneg {q : ℝ} (hq : 0 ≤ q) : 0 ≤ w q := by
  rcases le_total q 2 with h | h
  · rw [← w_eq_zero (le_refl (2 : ℝ))]
    exact antitoneOn_w (mem_Ici.2 hq) (mem_Ici.2 (by norm_num)) h
  · rw [w_eq_zero h]

end SphH
