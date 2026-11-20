import WordEnergy.Covariance

namespace WordEnergy

theorem weighted_constant (xs : List α) (q : α → Rat) (c : Rat) :
    weightedSum xs q (fun _ => c) = c * mass xs q := by
  induction xs with
  | nil => simp [weightedSum, mass]
  | cons x xs ih =>
    simp only [weightedSum, mass, List.map_cons, List.sum_cons] at *
    grind

theorem weighted_scale (xs : List α) (q b : α → Rat) (c : Rat) :
    weightedSum xs q (fun x => c * b x) = c * weightedSum xs q b := by
  induction xs with
  | nil => simp [weightedSum]
  | cons x xs ih =>
    simp only [weightedSum, List.map_cons, List.sum_cons] at *
    grind

-- A state-independent context message cannot introduce shared-center covariance.
theorem constant_message_zero_covariance (xs : List α) (q b : α → Rat) (c : Rat)
    (normalized : mass xs q = 1) :
    weightedSum xs q (fun x => ((fun _ => c) x - weightedSum xs q (fun _ => c)) *
      (b x - weightedSum xs q b)) = 0 := by
  have hc := weighted_constant xs q c
  rw [normalized] at hc
  have h := finite_covariance_identity xs q (fun _ => c) b normalized
  have hs := weighted_scale xs q b c
  grind

end WordEnergy
