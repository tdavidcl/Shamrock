import SphH.Kernel

/-!
# Smoothing-length residual: definitions

One particle `a` of mass `m` with `n` neighbours at distances `r i` (`i : Fin n`).
-/

noncomputable section

open Real Set Filter Topology

namespace SphH

/-- `h_fact = 1.2`. -/
def hfact : ℝ := 1.2

/-- Normalisation `π · h_fact³` (the 3D M4 normalisation is `1/π`). -/
def cnorm : ℝ := π * hfact ^ 3

variable {n : ℕ}

/-- `g h = (1 + Σ w(r_i/h)) / (π h_fact³)`; the `1` is the self term `w 0`. -/
def g (r : Fin n → ℝ) (h : ℝ) : ℝ := (1 + ∑ i, w (r i / h)) / cnorm

/-- Explicit derivative of `g`. -/
def dg (r : Fin n → ℝ) (h : ℝ) : ℝ := (∑ i, -dw (r i / h) * r i / h ^ 2) / cnorm

/-- `ρ_h(h) = m (h_fact/h)³`. -/
def rhoH (m h : ℝ) : ℝ := m * (hfact / h) ^ 3

/-- `ρ_sum(h) = m (1 + Σ w(r_i/h)) / (π h³)`. -/
def rhoSum (m : ℝ) (r : Fin n → ℝ) (h : ℝ) : ℝ := m * (1 + ∑ i, w (r i / h)) / (π * h ^ 3)

/-- The residual `f = ρ_sum − ρ_h`. -/
def f (m : ℝ) (r : Fin n → ℝ) (h : ℝ) : ℝ := rhoSum m r h - rhoH m h

/-- Explicit formula for `f'`. -/
def df (m : ℝ) (r : Fin n → ℝ) (h : ℝ) : ℝ := -3 * f m r h / h + rhoH m h * dg r h

/-- The 3D M4 kernel `W(r, h) = w(r/h) / (π h³)`. -/
def W (r h : ℝ) : ℝ := w (r / h) / (π * h ^ 3)

/-- `∂W/∂h`, explicit. -/
def dWdh (r h : ℝ) : ℝ := -3 * w (r / h) / (π * h ^ 4) - dw (r / h) * r / (π * h ^ 5)

end SphH
