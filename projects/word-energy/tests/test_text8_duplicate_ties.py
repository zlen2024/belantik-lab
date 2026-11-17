import unittest
import numpy as np
from scripts.audit_text8_duplicate_ties import repair_duplicate_scores
from word_energy.text8_lexical_models import pair_metrics


class DuplicateTieTest(unittest.TestCase):
    def test_only_same_ids_are_repaired(self):
        positive=np.array([1.0]); negative=np.array([[1.0+1e-15,1.0+1e-15]])
        fixed,count,error=repair_duplicate_scores(positive,negative,np.array([7]),np.array([[7,8]]))
        self.assertEqual(count,1)
        self.assertEqual(fixed[0,0],1.0)
        self.assertEqual(fixed[0,1],negative[0,1])
        self.assertAlmostEqual(pair_metrics(positive,fixed,np.array([1]))['noise_pairwise_auc'],.25)
        self.assertLess(error,1e-12)

    def test_large_disagreement_fails_loudly(self):
        with self.assertRaises(ValueError):
            repair_duplicate_scores(np.array([1.]),np.array([[1.01]]),np.array([7]),np.array([[7]]))


if __name__=='__main__': unittest.main()
