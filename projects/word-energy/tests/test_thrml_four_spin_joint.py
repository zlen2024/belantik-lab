"""Four-color schedule respects all 26 K4 star edges."""
import unittest
from scripts.audit_thrml_four_spin_joint import graph_check,EDGES,BLOCKS,STATES
class ColorTest(unittest.TestCase):
 def test_structure(self):
  graph_check();self.assertEqual(len(EDGES),26);self.assertEqual(STATES.shape,(4096,12));self.assertEqual(len(BLOCKS),4)
  self.assertEqual(max(sum(i in edge for edge in EDGES) for i in range(12)),5)
if __name__=='__main__':unittest.main()
