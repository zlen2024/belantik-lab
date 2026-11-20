import WordEnergy.CodeMapping

namespace WordEnergy.RealModel
noncomputable section

-- Scores match score_windows in ideal real arithmetic, with expm1(x)=exp(x)-1
-- and log1p(x)=log(1+x); the same offset appears in both compared scores.
theorem four_spin_code_gap_sign (center first second : SpinState → ℝ)
    (interaction1 interaction2 : SpinState → SpinState → ℝ) (offset : ℝ) :
    let q := probability center
    let d := delta first interaction1
    let e := delta second interaction2
    0 < ((Real.log (1 + ∑ i, q i * (d i + e i + d i * e i)) + offset) -
      (Real.log (1 + ∑ i, q i * d i) + Real.log (1 + ∑ i, q i * e i) + offset)) ↔
      0 < covariance q (message first interaction1) (message second interaction2) := by
  dsimp only
  rw [joint_log1p_argument _ _ _ (probability_normalized center)]
  rw [pair_log1p_argument _ _ (probability_normalized center)]
  rw [pair_log1p_argument _ _ (probability_normalized center)]
  simp_rw [← message_eq_one_add_delta]
  exact boltzmann_log_gap_sign center first second interaction1 interaction2 offset


theorem four_spin_code_gap_negative (center first second : SpinState → ℝ)
    (interaction1 interaction2 : SpinState → SpinState → ℝ) (offset : ℝ) :
    let q := probability center
    let d := delta first interaction1
    let e := delta second interaction2
    ((Real.log (1 + ∑ i, q i * (d i + e i + d i * e i)) + offset) -
      (Real.log (1 + ∑ i, q i * d i) + Real.log (1 + ∑ i, q i * e i) + offset)) < 0 ↔
      covariance q (message first interaction1) (message second interaction2) < 0 := by
  dsimp only
  rw [joint_log1p_argument _ _ _ (probability_normalized center)]
  rw [pair_log1p_argument _ _ (probability_normalized center)]
  rw [pair_log1p_argument _ _ (probability_normalized center)]
  simp_rw [← message_eq_one_add_delta]
  exact boltzmann_log_gap_negative center first second interaction1 interaction2 offset

end
end WordEnergy.RealModel
