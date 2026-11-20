import Lean

namespace WordEnergy

-- A finite list denotes occurrences of states; no distinctness assumption is needed.
def weightedSum (xs : List α) (q f : α → Rat) : Rat :=
  (xs.map fun x => q x * f x).sum

def mass (xs : List α) (q : α → Rat) : Rat :=
  (xs.map q).sum

-- The algebraic identity is valid even before the weights sum to one.
theorem centered_sum_expansion (xs : List α) (q a b : α → Rat) (u v : Rat) :
    weightedSum xs q (fun x => (a x - u) * (b x - v)) =
      weightedSum xs q (fun x => a x * b x) -
      v * weightedSum xs q a - u * weightedSum xs q b + u * v * mass xs q := by
  induction xs with
  | nil => simp [weightedSum, mass]; grind
  | cons x xs ih =>
    simp only [weightedSum, mass, List.map_cons, List.sum_cons] at *
    grind

theorem finite_covariance_identity (xs : List α) (q a b : α → Rat)
    (normalized : mass xs q = 1) :
    weightedSum xs q (fun x => (a x - weightedSum xs q a) *
                                  (b x - weightedSum xs q b)) =
      weightedSum xs q (fun x => a x * b x) -
      weightedSum xs q a * weightedSum xs q b := by
  have h := centered_sum_expansion xs q a b (weightedSum xs q a) (weightedSum xs q b)
  rw [normalized] at h
  grind

end WordEnergy

namespace WordEnergy

def toyStates : List Bool := [false, true]
def toyQ (_ : Bool) : Rat := 1 / 2
def toyA (x : Bool) : Rat := if x then 3 else 1
def toySame (x : Bool) : Rat := if x then 3 else 1
def toyOpposite (x : Bool) : Rat := if x then 1 else 3

theorem toy_valid_probability : mass toyStates toyQ = 1 := by simp [mass, toyStates, toyQ]; grind

theorem toy_positive_gap :
    weightedSum toyStates toyQ (fun x => toyA x * toySame x) -
      weightedSum toyStates toyQ toyA * weightedSum toyStates toyQ toySame = 1 := by simp [weightedSum, toyStates, toyQ, toyA, toySame]; grind

theorem toy_negative_gap :
    weightedSum toyStates toyQ (fun x => toyA x * toyOpposite x) -
      weightedSum toyStates toyQ toyA * weightedSum toyStates toyQ toyOpposite = -1 := by simp [weightedSum, toyStates, toyQ, toyA, toyOpposite]; grind

end WordEnergy
