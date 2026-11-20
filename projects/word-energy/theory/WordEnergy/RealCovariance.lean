import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace WordEnergy.RealModel

-- q: center distribution; a, b: messages from the two context occurrences.
def expectation {n : Nat} (q f : Fin n → ℝ) : ℝ :=
  ∑ i, q i * f i

def covariance {n : Nat} (q a b : Fin n → ℝ) : ℝ :=
  expectation q (fun i => (a i - expectation q a) * (b i - expectation q b))

theorem covariance_identity {n : Nat} (q a b : Fin n → ℝ)
    (normalized : (∑ i, q i) = 1) :
    covariance q a b = expectation q (fun i => a i * b i) - expectation q a * expectation q b := by
  unfold covariance expectation
  calc
    (∑ i, q i * ((a i - ∑ j, q j * a j) * (b i - ∑ j, q j * b j))) =
      ∑ i, (q i * (a i * b i) - (∑ j, q j * b j) * (q i * a i) -
        (∑ j, q j * a j) * (q i * b i) +
        ((∑ j, q j * a j) * (∑ j, q j * b j)) * q i) := by
          apply Finset.sum_congr rfl
          intro i hi
          ring
    _ = (∑ i, q i * (a i * b i)) - (∑ i, q i * a i) * (∑ i, q i * b i) := by
          simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
          rw [normalized]
          ring

-- Full window prior cancels, as long as both scores use the same offset.
theorem log_gap_sign {n : Nat} (q a b : Fin n → ℝ) (offset : ℝ)
    (normalized : (∑ i, q i) = 1)
    (joint_pos : 0 < expectation q (fun i => a i * b i))
    (first_pos : 0 < expectation q a) (second_pos : 0 < expectation q b) :
    0 < ((Real.log (expectation q (fun i => a i * b i)) + offset) -
       (Real.log (expectation q a) + Real.log (expectation q b) + offset)) ↔
      0 < covariance q a b := by
  rw [covariance_identity q a b normalized]
  rw [← Real.log_mul (ne_of_gt first_pos) (ne_of_gt second_pos)]
  have prod_pos : 0 < expectation q a * expectation q b := mul_pos first_pos second_pos
  rw [show Real.log (expectation q (fun i => a i * b i)) + offset -
      (Real.log (expectation q a * expectation q b) + offset) =
      Real.log (expectation q (fun i => a i * b i)) -
      Real.log (expectation q a * expectation q b) by ring]
  rw [sub_pos, Real.log_lt_log_iff prod_pos joint_pos, sub_pos]


theorem log_gap_negative {n : Nat} (q a b : Fin n → ℝ) (offset : ℝ)
    (normalized : (∑ i, q i) = 1)
    (joint_pos : 0 < expectation q (fun i => a i * b i))
    (first_pos : 0 < expectation q a) (second_pos : 0 < expectation q b) :
    ((Real.log (expectation q (fun i => a i * b i)) + offset) -
       (Real.log (expectation q a) + Real.log (expectation q b) + offset)) < 0 ↔
      covariance q a b < 0 := by
  rw [covariance_identity q a b normalized]
  rw [← Real.log_mul (ne_of_gt first_pos) (ne_of_gt second_pos)]
  have prod_pos : 0 < expectation q a * expectation q b := mul_pos first_pos second_pos
  rw [show Real.log (expectation q (fun i => a i * b i)) + offset -
      (Real.log (expectation q a * expectation q b) + offset) =
      Real.log (expectation q (fun i => a i * b i)) -
      Real.log (expectation q a * expectation q b) by ring]
  rw [sub_neg, Real.log_lt_log_iff joint_pos prod_pos, sub_neg]

end WordEnergy.RealModel
