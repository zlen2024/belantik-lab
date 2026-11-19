"""Canonical candidate keys preserve same-ID and swapped-context ties."""
import unittest
import numpy as np
from word_energy.wikitext_joint_evaluation import canonical_scores
class JointEvalTest(unittest.TestCase):
 def test_duplicate_candidates_and_context_permutations(self):
  windows=np.array([[1,2,3],[1,3,2],[2,2,2]],np.int32);noise=np.array([[1,0],[1,0],[2,1]],np.int32)
  def score(_,x):return x[:,0]+x[:,1]*.1+x[:,2]*.01
  p,n=canonical_scores(None,score,windows,noise,chunk_size=2)
  self.assertEqual(p[0],n[0,0]);self.assertEqual(p[1],n[1,0]);self.assertEqual(p[0],p[1]);self.assertEqual(n[0,1],n[1,1]);self.assertEqual(p[2],n[2,0])
if __name__=='__main__':unittest.main()
