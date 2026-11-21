import WordEnergy.Boltzmann

namespace WordEnergy.SharedStarCritic
noncomputable section

-- At zero coupling the pair law is the product of the unary laws. This
-- algebraic identity applies coordinatewise to the raw spin-product readout.
theorem independent_product_expectation (q r f h : Fin 16 → ℝ) :
    (∑ i, ∑ j, q i * r j * (f i * h j)) =
      (∑ i, q i * f i) * (∑ j, r j * h j) := by
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  ring

-- One independent uniform spin supplies a center-state-independent message.
-- The four-spin diagonal uniform-leaf message is the product of these factors.
theorem uniform_spin_message (g s : ℝ) (spin : s = 1 ∨ s = -1) :
    (Real.exp (g * s) + Real.exp (-(g * s))) / 2 =
      (Real.exp g + Real.exp (-g)) / 2 := by
  rcases spin with h | h
  · subst s
    simp
  · subst s
    simp only [mul_neg, mul_one, neg_neg]
    ring

#print independent_product_expectation
#print axioms independent_product_expectation
#print uniform_spin_message
#print axioms uniform_spin_message

end
end WordEnergy.SharedStarCritic
