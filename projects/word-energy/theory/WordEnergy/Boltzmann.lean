import WordEnergy.RealCovariance

namespace WordEnergy.RealModel

noncomputable section

abbrev SpinState := Fin 16

def normalizer (u : SpinState → ℝ) : ℝ := ∑ i, Real.exp (u i)
def probability (u : SpinState → ℝ) (i : SpinState) : ℝ := Real.exp (u i) / normalizer u

theorem normalizer_pos (u : SpinState → ℝ) : 0 < normalizer u := by
  unfold normalizer
  exact Finset.sum_pos (by intro i hi; exact Real.exp_pos _) ⟨0, Finset.mem_univ _⟩

theorem probability_pos (u : SpinState → ℝ) (i : SpinState) : 0 < probability u i := by
  exact div_pos (Real.exp_pos _) (normalizer_pos u)

theorem probability_normalized (u : SpinState → ℝ) : (∑ i, probability u i) = 1 := by
  unfold probability
  simp only [div_eq_mul_inv, ← Finset.sum_mul]
  exact mul_inv_cancel₀ (ne_of_gt (normalizer_pos u))

-- The interaction exponent is unrestricted: it can be the sum of four g*s*t terms.
def message (u : SpinState → ℝ) (interaction : SpinState → SpinState → ℝ) (i : SpinState) : ℝ :=
  ∑ j, probability u j * Real.exp (interaction i j)

theorem message_pos (u : SpinState → ℝ) (interaction : SpinState → SpinState → ℝ) (i : SpinState) :
    0 < message u interaction i := by
  unfold message
  exact Finset.sum_pos (by intro j hj; exact mul_pos (probability_pos u j) (Real.exp_pos _)) ⟨0, Finset.mem_univ _⟩

theorem expectation_positive (u : SpinState → ℝ) (f : SpinState → ℝ) (hf : ∀ i, 0 < f i) :
    0 < expectation (probability u) f := by
  unfold expectation
  exact Finset.sum_pos (by intro i hi; exact mul_pos (probability_pos u i) (hf i)) ⟨0, Finset.mem_univ _⟩

-- A, B are exact Boltzmann messages from context logits and interactions.
theorem boltzmann_log_gap_sign (center first second : SpinState → ℝ)
    (interaction1 interaction2 : SpinState → SpinState → ℝ) (offset : ℝ) :
    let q := probability center
    let a := message first interaction1
    let b := message second interaction2
    0 < ((Real.log (expectation q (fun i => a i * b i)) + offset) -
      (Real.log (expectation q a) + Real.log (expectation q b) + offset)) ↔
      0 < covariance q a b := by
  dsimp only
  apply log_gap_sign _ _ _ offset (probability_normalized center)
  · exact expectation_positive center _ (by intro i; exact mul_pos (message_pos first interaction1 i) (message_pos second interaction2 i))
  · exact expectation_positive center _ (message_pos first interaction1)
  · exact expectation_positive center _ (message_pos second interaction2)


theorem boltzmann_log_gap_negative (center first second : SpinState → ℝ)
    (interaction1 interaction2 : SpinState → SpinState → ℝ) (offset : ℝ) :
    let q := probability center
    let a := message first interaction1
    let b := message second interaction2
    ((Real.log (expectation q (fun i => a i * b i)) + offset) -
      (Real.log (expectation q a) + Real.log (expectation q b) + offset)) < 0 ↔
      covariance q a b < 0 := by
  dsimp only
  apply log_gap_negative _ _ _ offset (probability_normalized center)
  · exact expectation_positive center _ (by intro i; exact mul_pos (message_pos first interaction1 i) (message_pos second interaction2 i))
  · exact expectation_positive center _ (message_pos first interaction1)
  · exact expectation_positive center _ (message_pos second interaction2)

end
end WordEnergy.RealModel
