"""Tests for article parser, row/OOV boundaries and macro averaging."""
import unittest
import numpy as np
from scripts.build_wikitext2_document_data import sample,doc_macro,SECTION

class WikiDataTest(unittest.TestCase):
    def test_no_oov_or_row_crossing(self):
        ids=np.array([0,-1,1,-1,-1,2,3,4,-1,-1,5,6],np.int32)
        docs=np.array([0,0,0,-1,-1,0,0,0,-1,-1,1,1],np.int32)
        pairs,pos,chosen=sample(ids,docs,2000,np.random.default_rng(1))
        for (a,b),d in zip(pos,chosen):
            self.assertGreaterEqual(ids[a],0);self.assertGreaterEqual(ids[b],0)
            self.assertEqual(d,docs[b]);self.assertTrue(np.all(ids[min(a,b):max(a,b)+1]>=0))
        self.assertFalse(np.any((pairs[:,0]==0)|(pairs[:,0]==1)))
    def test_section_headings_are_excluded(self):
        self.assertIsNotNone(SECTION.match(' = = Gameplay = = \n'))
        self.assertIsNone(SECTION.match(' An ordinary paragraph. '))

    def test_document_macro_not_pair_weighted(self):
        pos=np.array([2.,2.,0.]);neg=np.array([[1.],[1.],[1.]]);docs=np.array([0,0,1])
        self.assertEqual(doc_macro(pos,neg,docs),.5)

if __name__=='__main__':unittest.main()
