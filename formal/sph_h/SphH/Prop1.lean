import SphH.Deriv

/-!
# Proposition 1: properties of the smoothing-length residual

All statements are for one particle of mass `m > 0` with `n` neighbours at distances
`r i > 0`.
-/

noncomputable section

open Real Set Filter Topology

namespace SphH

variable {n : ℕ}

/-! ### Kernel along `h ↦ w (r / h)` -/

theorem w_div_le {r a b : ℝ} (hr : 0 < r) (ha : 0 < a) (hab : a ≤ b) :
    w (r / a) ≤ w (r / b) :=
  have hb : 0 < b := lt_of_lt_of_le ha hab
  antitoneOn_w (mem_Ici.2 (by positivity)) (mem_Ici.2 (by positivity))
    (div_le_div_of_nonneg_left hr.le ha hab)

theorem w_div_lt {r a b : ℝ} (hr : 0 < r) (ha : 0 < a) (hra : r < 2 * a) (hab : a < b) :
    w (r / a) < w (r / b) := by
  have hb : 0 < b := ha.trans hab
  have h1 : r / a < 2 := by rw [div_lt_iff₀ ha]; linarith
  have h2 : r / b < r / a := div_lt_div_of_pos_left hr ha hab
  exact strictAntiOn_w ⟨by positivity, by linarith⟩ ⟨by positivity, h1.le⟩ h2

/-! ### (i-a) monotonicity of `g` -/

/-- (i-a) `g` is monotone on `(0, ∞)`. -/
theorem monotoneOn_g {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) : MonotoneOn (g r) (Ioi 0) := by
  intro a ha b _ hab
  unfold g
  apply div_le_div_of_nonneg_right _ cnorm_pos.le
  have := Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) => w_div_le (hr i) ha hab
  linarith

/-- (i-a) `g` is strictly monotone on `{h | ∃ i, r_i < 2h}`. -/
theorem strictMonoOn_g {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) :
    StrictMonoOn (g r) {h | ∃ i, r i < 2 * h} := by
  rintro a ⟨i, hi⟩ b _ hab
  have ha : 0 < a := by linarith [hr i]
  unfold g
  apply div_lt_div_of_pos_right _ cnorm_pos
  have := Finset.sum_lt_sum (fun j _ => w_div_le (hr j) ha hab.le)
    ⟨i, Finset.mem_univ _, w_div_lt (hr i) ha hi hab⟩
  linarith

/-- When `n ≥ 1`, `{h | ∃ i, r_i < 2h} = (min r_i / 2, ∞)`. -/
theorem setOf_exists_lt_eq_Ioi (r : Fin n → ℝ) (hne : (Finset.univ : Finset (Fin n)).Nonempty) :
    {h | ∃ i, r i < 2 * h} = Ioi (Finset.univ.inf' hne r / 2) := by
  ext h
  simp only [mem_ofPred_eq, mem_Ioi, div_lt_iff₀ (two_pos : (0 : ℝ) < 2), Finset.inf'_lt_iff,
    Finset.mem_univ, true_and]
  simp only [mul_comm]

/-! ### (i-b) limits of `g` -/

/-- (i-b) `g h = 1/(π h_fact³)` as soon as no neighbour is in the support (`h ≤ min r_i / 2`). -/
theorem g_eq_of_forall_le {r : Fin n → ℝ} {h : ℝ} (hh : 0 < h) (hle : ∀ i, 2 * h ≤ r i) :
    g r h = 1 / cnorm := by
  unfold g
  rw [Finset.sum_eq_zero fun i _ => w_eq_zero ((le_div_iff₀ hh).2 (by linarith [hle i])),
    add_zero]

/-- (i-b), stated with `min r_i` for `n ≥ 1`. -/
theorem g_eq_of_le_min {r : Fin n → ℝ} (hne : (Finset.univ : Finset (Fin n)).Nonempty) {h : ℝ}
    (hh : 0 < h) (hle : h ≤ Finset.univ.inf' hne r / 2) : g r h = 1 / cnorm :=
  g_eq_of_forall_le hh fun i => by
    have : Finset.univ.inf' hne r ≤ r i := Finset.inf'_le r (Finset.mem_univ i)
    linarith

/-- (i-b) `g` is eventually constant equal to `1/(π h_fact³)` as `h → 0⁺`. -/
theorem eventually_g_eq_nhdsGT {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) :
    ∀ᶠ h in 𝓝[>] 0, g r h = 1 / cnorm := by
  have hall : ∀ᶠ h in 𝓝[>] (0 : ℝ), ∀ i, 2 * h ≤ r i := by
    rw [eventually_all]
    intro i
    have : ∀ᶠ h in 𝓝 (0 : ℝ), h < r i / 2 := eventually_lt_nhds (by linarith [hr i])
    filter_upwards [nhdsWithin_le_nhds this] with h hh
    linarith
  filter_upwards [hall, self_mem_nhdsWithin] with h hle hh
  exact g_eq_of_forall_le hh hle

/-- Continuity at `0` of `q ↦ φ q` makes `h ↦ Σ φ(r_i/h)` tend to `n φ 0` at infinity. -/
theorem tendsto_sum_comp_div (r : Fin n → ℝ) {φ : ℝ → ℝ} (hφ : ContinuousAt φ 0) :
    Tendsto (fun h => ∑ i, φ (r i / h)) atTop (𝓝 (n * φ 0)) := by
  have : Tendsto (fun h => ∑ i, φ (r i / h)) atTop (𝓝 (∑ _i : Fin n, φ 0)) :=
    tendsto_finsetSum _ fun i _ =>
      hφ.tendsto.comp (tendsto_const_nhds.div_atTop tendsto_id)
  simpa using this

/-- (i-b) `g → (n+1)/(π h_fact³)` as `h → ∞`. -/
theorem tendsto_g_atTop (r : Fin n → ℝ) : Tendsto (g r) atTop (𝓝 ((n + 1) / cnorm)) := by
  have := ((tendsto_sum_comp_div r continuous_w.continuousAt).const_add 1).div_const cnorm
  rw [w_zero, mul_one, add_comm] at this
  exact this

/-! ### (i-c) existence and uniqueness of the root -/

theorem g_le {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h : ℝ} (hh : 0 < h) :
    g r h ≤ (n + 1) / cnorm := by
  unfold g
  apply div_le_div_of_nonneg_right _ cnorm_pos.le
  have : ∑ i, w (r i / h) ≤ ∑ _i : Fin n, (1 : ℝ) :=
    Finset.sum_le_sum fun i _ => w_le_one (by have := hr i; positivity)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at this
  linarith

/-- At a root, at least one neighbour is inside the kernel support. -/
theorem exists_lt_of_g_eq_one {r : Fin n → ℝ} {h : ℝ} (hh : 0 < h) (hg : g r h = 1) :
    ∃ i, r i < 2 * h := by
  by_contra hcon
  push Not at hcon
  rw [g_eq_of_forall_le hh hcon, div_eq_one_iff_eq cnorm_pos.ne'] at hg
  linarith [one_lt_cnorm]

/-- (i-c) A root of `g = 1` exists iff `N_part = n + 1 ≥ 6`. -/
theorem exists_root_iff {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) :
    (∃ h, 0 < h ∧ g r h = 1) ↔ 6 ≤ n + 1 := by
  constructor
  · rintro ⟨h, hh, hg⟩
    have h1 := g_le hr hh
    rw [hg, le_div_iff₀ cnorm_pos, one_mul] at h1
    have h5 : (5 : ℝ) < n + 1 := five_lt_cnorm_lt_six.1.trans_le h1
    have : (4 : ℝ) < n := by linarith
    have : 4 < n := by exact_mod_cast this
    omega
  · intro hn
    have hn' : (6 : ℝ) ≤ n + 1 := by exact_mod_cast hn
    have hlim : 1 < (n + 1) / cnorm := by
      rw [one_lt_div cnorm_pos]; linarith [five_lt_cnorm_lt_six.2]
    obtain ⟨h₀, hg₀, hh₀⟩ := ((eventually_g_eq_nhdsGT hr).and self_mem_nhdsWithin).exists
    obtain ⟨h₁, hg₁, hh₁⟩ :=
      (((tendsto_g_atTop r).eventually (lt_mem_nhds hlim)).and (eventually_gt_atTop h₀)).exists
    have hcont : ContinuousOn (g r) (Icc h₀ h₁) :=
      (continuousOn_g r).mono fun x hx => lt_of_lt_of_le hh₀ hx.1
    obtain ⟨h, hmem, hgh⟩ := intermediate_value_Icc hh₁.le hcont
      ⟨by rw [hg₀, div_le_one cnorm_pos]; exact one_lt_cnorm.le, hg₁.le⟩
    exact ⟨h, lt_of_lt_of_le hh₀ hmem.1, hgh⟩

/-- (i-c) The root is unique. -/
theorem root_unique {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h₁ h₂ : ℝ} (hh₁ : 0 < h₁)
    (hh₂ : 0 < h₂) (hg₁ : g r h₁ = 1) (hg₂ : g r h₂ = 1) : h₁ = h₂ :=
  (strictMonoOn_g hr).injOn (exists_lt_of_g_eq_one hh₁ hg₁) (exists_lt_of_g_eq_one hh₂ hg₂)
    (hg₁.trans hg₂.symm)

/-- (i-c) `f < 0` below the root. -/
theorem f_neg_of_lt_root {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i)
    {hs h : ℝ} (hhs : 0 < hs) (hgs : g r hs = 1) (hh : 0 < h) (hlt : h < hs) : f m r h < 0 := by
  rw [f_eq]
  apply mul_neg_of_pos_of_neg (rhoH_pos hm hh)
  rw [sub_neg]
  by_cases hS : ∃ i, r i < 2 * h
  · exact hgs ▸ strictMonoOn_g hr hS (exists_lt_of_g_eq_one hhs hgs) hlt
  · push Not at hS
    rw [g_eq_of_forall_le hh hS, div_lt_one cnorm_pos]; exact one_lt_cnorm

/-- (i-c) `f > 0` above the root. -/
theorem f_pos_of_root_lt {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i)
    {hs h : ℝ} (hhs : 0 < hs) (hgs : g r hs = 1) (hlt : hs < h) : 0 < f m r h := by
  rw [f_eq]
  apply mul_pos (rhoH_pos hm (hhs.trans hlt))
  rw [sub_pos]
  obtain ⟨i, hi⟩ := exists_lt_of_g_eq_one hhs hgs
  exact hgs ▸ strictMonoOn_g hr ⟨i, hi⟩ ⟨i, by linarith⟩ hlt

/-- (i-c), packaged: if `n + 1 ≥ 6` there is a unique root `h⋆ > 0` of `g = 1`, with `f < 0`
on `(0, h⋆)` and `f > 0` on `(h⋆, ∞)`. -/
theorem existsUnique_root {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i)
    (hn : 6 ≤ n + 1) :
    ∃! hs, 0 < hs ∧ g r hs = 1 ∧ (∀ h, 0 < h → h < hs → f m r h < 0) ∧
      (∀ h, hs < h → 0 < f m r h) := by
  obtain ⟨hs, hhs, hgs⟩ := (exists_root_iff hr).2 hn
  refine ⟨hs, ⟨hhs, hgs, fun h hh hlt => f_neg_of_lt_root hm hr hhs hgs hh hlt,
    fun h hlt => f_pos_of_root_lt hm hr hhs hgs hlt⟩, ?_⟩
  rintro h' ⟨hh', hg', -, -⟩
  exact root_unique hr hh' hhs hg' hgs

/-! ### (ii) derivative of `f` -/

theorem dg_nonneg {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h : ℝ} (hh : 0 < h) : 0 ≤ dg r h := by
  unfold dg
  apply div_nonneg _ cnorm_pos.le
  apply Finset.sum_nonneg fun i _ => ?_
  have := dw_nonpos (div_nonneg (hr i).le hh.le)
  have := hr i
  apply div_nonneg _ (by positivity)
  nlinarith

theorem dg_pos {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h : ℝ} (hh : 0 < h)
    (hS : ∃ i, r i < 2 * h) : 0 < dg r h := by
  obtain ⟨i, hi⟩ := hS
  unfold dg
  apply div_pos _ cnorm_pos
  refine Finset.sum_pos' (fun j _ => ?_) ⟨i, Finset.mem_univ _, ?_⟩
  · have := dw_nonpos (div_nonneg (hr j).le hh.le)
    have := hr j
    apply div_nonneg _ (by positivity)
    nlinarith
  · have := dw_neg (div_pos (hr i) hh) (by rw [div_lt_iff₀ hh]; linarith)
    have := hr i
    apply div_pos _ (by positivity)
    nlinarith

/-- (ii) `f' ≥ −3 f / h`. -/
theorem df_ge {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h : ℝ} (hh : 0 < h) :
    -3 * f m r h / h ≤ df m r h := by
  unfold df
  have := mul_nonneg (rhoH_pos hm hh).le (dg_nonneg hr hh)
  linarith

/-- (ii) `f < 0 → f' > 0`. -/
theorem df_pos_of_f_neg {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h : ℝ}
    (hh : 0 < h) (hf : f m r h < 0) : 0 < df m r h := by
  have : 0 < -3 * f m r h / h := div_pos (by linarith) hh
  linarith [df_ge hm hr hh]

/-- (ii-b) `f' ≤ 0 → f > 0`. -/
theorem f_pos_of_df_nonpos {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h : ℝ}
    (hh : 0 < h) (hdf : df m r h ≤ 0) : 0 < f m r h := by
  rcases lt_trichotomy (f m r h) 0 with hf | hf | hf
  · linarith [df_pos_of_f_neg hm hr hh hf]
  · exfalso
    have hg : g r h = 1 := by
      have := hf; rw [f_eq] at this
      have := (mul_eq_zero.1 this).resolve_left (rhoH_pos hm hh).ne'
      linarith
    have := mul_pos (rhoH_pos hm hh) (dg_pos hr hh (exists_lt_of_g_eq_one hh hg))
    simp only [df, hf, mul_zero, zero_div, zero_add] at hdf
    linarith
  · exact hf

/-! ### (iii) Newton–Raphson step bound -/

/-- (iii), sharpened: `f' < 0` alone gives the Newton–Raphson bound (`f > 0` then follows
from (ii-b)). -/
theorem nr_ge_of_df_neg {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h : ℝ}
    (hh : 0 < h) (hdf : df m r h < 0) : 4 * h / 3 ≤ h - f m r h / df m r h := by
  have key := df_ge hm hr hh
  rw [div_le_iff₀ hh] at key
  have : f m r h / df m r h ≤ -h / 3 := by
    rw [div_le_iff_of_neg hdf]; nlinarith
  linarith

/-- (iii) If `f > 0` and `f' < 0`, the Newton–Raphson iterate is at least `4h/3`. -/
theorem nr_ge {m : ℝ} (hm : 0 < m) {r : Fin n → ℝ} (hr : ∀ i, 0 < r i) {h : ℝ} (hh : 0 < h)
    (_hf : 0 < f m r h) (hdf : df m r h < 0) : 4 * h / 3 ≤ h - f m r h / df m r h :=
  nr_ge_of_df_neg hm hr hh hdf

/-! ### (iii-sharp) the bound `4/3` is attained asymptotically -/

/-- `h · g'(h) = Σ φ(r_i/h) / (π h_fact³)` with `φ q = −w'(q) q`. -/
theorem mul_dg_eq (r : Fin n → ℝ) {h : ℝ} (hh : h ≠ 0) :
    h * dg r h = (∑ i, -dw (r i / h) * (r i / h)) / cnorm := by
  unfold dg
  rw [mul_div_assoc', Finset.mul_sum]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  field_simp

theorem tendsto_mul_dg (r : Fin n → ℝ) : Tendsto (fun h => h * dg r h) atTop (𝓝 0) := by
  have hφ : ContinuousAt (fun q => -dw q * q) 0 := by
    have := continuous_dw; fun_prop
  have := (tendsto_sum_comp_div r hφ).div_const cnorm
  simp only [mul_zero, zero_div] at this
  refine this.congr' ?_
  filter_upwards [eventually_gt_atTop 0] with h hh
  exact (mul_dg_eq r hh.ne').symm

/-- (iii-sharp) If `n + 1 ≥ 6`, then eventually `f > 0` and `f' < 0`, and the ratio between the
Newton–Raphson iterate and `h` tends to `4/3` as `h → ∞`. -/
theorem nr_tendsto {m : ℝ} (hm : 0 < m) (r : Fin n → ℝ) (hn : 6 ≤ n + 1) :
    (∀ᶠ h in atTop, 0 < f m r h ∧ df m r h < 0) ∧
      Tendsto (fun h => (h - f m r h / df m r h) / h) atTop (𝓝 (4 / 3)) := by
  set L := (n + 1) / cnorm - 1 with hL
  have hn' : (6 : ℝ) ≤ n + 1 := by exact_mod_cast hn
  have hLpos : 0 < L := by
    rw [hL, sub_pos, one_lt_div cnorm_pos]; linarith [five_lt_cnorm_lt_six.2]
  have hg1 : Tendsto (fun h => g r h - 1) atTop (𝓝 L) := (tendsto_g_atTop r).sub_const 1
  -- `D h = h g'(h) − 3 (g h − 1)`, so that `f' = (ρ_h / h) D`.
  have hD : Tendsto (fun h => h * dg r h - 3 * (g r h - 1)) atTop (𝓝 (-3 * L)) := by
    have := (tendsto_mul_dg r).sub (hg1.const_mul 3)
    simpa using this
  have hDneg : ∀ᶠ h in atTop, h * dg r h - 3 * (g r h - 1) < 0 :=
    hD.eventually (gt_mem_nhds (by linarith))
  have hgpos : ∀ᶠ h in atTop, 0 < g r h - 1 := hg1.eventually (lt_mem_nhds hLpos)
  have hdf_eq : ∀ h, h ≠ 0 →
      df m r h = rhoH m h / h * (h * dg r h - 3 * (g r h - 1)) := by
    intro h hh; rw [df, f_eq]; field_simp; ring
  refine ⟨?_, ?_⟩
  · filter_upwards [hDneg, hgpos, eventually_gt_atTop 0] with h hDh hgh hh
    refine ⟨by rw [f_eq]; exact mul_pos (rhoH_pos hm hh) hgh, ?_⟩
    rw [hdf_eq h hh.ne']
    exact mul_neg_of_pos_of_neg (div_pos (rhoH_pos hm hh) hh) hDh
  · have hlim : Tendsto (fun h => 1 - (g r h - 1) / (h * dg r h - 3 * (g r h - 1))) atTop
        (𝓝 (1 - L / (-3 * L))) :=
      tendsto_const_nhds.sub (hg1.div hD (by linarith))
    have hval : 1 - L / (-3 * L) = 4 / 3 := by field_simp; ring
    rw [hval] at hlim
    refine hlim.congr' ?_
    filter_upwards [hDneg, eventually_gt_atTop 0] with h hDh hh
    have hρ := rhoH_pos hm hh
    rw [hdf_eq h hh.ne', f_eq]
    field_simp

end SphH
