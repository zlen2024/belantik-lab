"""Checks joint window assembly preserves row and OOV boundaries."""
import unittest
import numpy as np
from scripts.build_wikitext2_joint_context_data import windows
class WindowTest(unittest.TestCase):
 def test_distinct_contexts_and_boundaries(self):
  ids=np.array([0,1,2,-1,-1,3,4,5,-1,6,7],np.int32);docs=np.array([0,0,0,-1,-1,1,1,1,1,1,1],np.int32)
  pos=np.array([[1,0],[6,5],[9,10]],np.int32);pairs=ids[pos];pd=docs[pos[:,0]]
  words,positions,d,source_rows,n=windows(ids,docs,pairs,pos,pd,200,np.random.default_rng(4).integers(2**31))
  self.assertEqual(n,2)
  for tokens,p,doc in zip(words,positions,d):
   self.assertNotEqual(p[1],p[2]);np.testing.assert_array_equal(tokens,ids[p]);self.assertTrue(all(docs[x]==doc for x in p))
   for context in p[1:]:self.assertTrue(np.all(ids[min(p[0],context):max(p[0],context)+1]>=0))
if __name__=='__main__':unittest.main()
