import WordEnergy.Boltzmann

namespace WordEnergy.RealModel
noncomputable section

-- expm1(x) in real arithmetic is exp(x) - 1.
def delta (u : SpinState → ℝ) (interaction : SpinState → SpinState → ℝ) (i : SpinState) : ℝ :=
  ∑ j, probability u j * (Real.exp (interaction i j) - 1)

theorem message_eq_one_add_delta (u : SpinState → ℝ) (interaction : SpinState → SpinState → ℝ)
    (i : SpinState) : message u interaction i = 1 + delta u interaction i := by
  unfold message delta
  simp only [mul_sub, mul_one, Finset.sum_sub_distrib]
  have h := probability_normalized u
  linarith

-- Exact real-arithmetic analogue of the joint code's log1p argument.
theorem joint_log1p_argument (q d e : SpinState → ℝ) (normalized : (∑ i, q i) = 1) :
    1 + (∑ i, q i * (d i + e i + d i * e i)) =
      expectation q (fun i => (1 + d i) * (1 + e i)) := by
  unfold expectation
  calc
    1 + (∑ i, q i * (d i + e i + d i * e i)) =
        (∑ i, q i) + (∑ i, q i * (d i + e i + d i * e i)) := by rw [normalized]
    _ = ∑ i, (q i + q i * (d i + e i + d i * e i)) := by rw [Finset.sum_add_distrib]
    _ = ∑ i, q i * ((1 + d i) * (1 + e i)) := by
      apply Finset.sum_congr rfl
      intro i hi
      ring

theorem pair_log1p_argument (q d : SpinState → ℝ) (normalized : (∑ i, q i) = 1) :
    1 + (∑ i, q i * d i) = expectation q (fun i => 1 + d i) := by
  unfold expectation
  calc
    1 + (∑ i, q i * d i) = (∑ i, q i) + (∑ i, q i * d i) := by rw [normalized]
    _ = ∑ i, (q i + q i * d i) := by rw [Finset.sum_add_distrib]
    _ = ∑ i, q i * (1 + d i) := by
      apply Finset.sum_congr rfl
      intro i hi
      ring

end
end WordEnergy.RealModel
