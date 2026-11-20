import Lean

namespace WordEnergy

-- Each token occurrence has four vertices; the contexts are offset by 4 and 8.
def edges : List (Nat × Nat) :=
  [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),
   (4,5),(4,6),(4,7),(5,6),(5,7),(6,7),
   (8,9),(8,10),(8,11),(9,10),(9,11),(10,11),
   (0,4),(1,5),(2,6),(3,7),(0,8),(1,9),(2,10),(3,11)]

-- Exact blocks from audit_thrml_four_spin_joint.py, transposed into color labels.
def color : Nat → Nat
  | 0 => 0 | 1 => 1 | 2 => 2 | 3 => 3
  | 4 => 3 | 5 => 0 | 6 => 1 | 7 => 2
  | 8 => 3 | 9 => 0 | 10 => 1 | 11 => 2
  | _ => 4

theorem schedule_valid :
    edges.length = 26 ∧
    (∀ e ∈ edges, e.1 < 12 ∧ e.2 < 12 ∧ color e.1 < 4 ∧
       color e.2 < 4 ∧ color e.1 ≠ color e.2) ∧
    (∀ v : Fin 12, color v.val < 4) := by
  decide

-- The center token induces K4: three colors cannot distinguish all four spins.
theorem center_needs_four_colors :
    ∀ a b c d : Fin 3,
      ¬ (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) := by
  decide

end WordEnergy
