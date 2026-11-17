import unittest
import numpy as np
from scripts.audit_text8_frequency_strata import groups,breakdown


class FrequencyStrataTest(unittest.TestCase):
    def test_partition_thresholds_and_same_id_exclusion(self):
        arrays={'counts':np.array([1,2,4,8,8,16]),
                'validation_pairs':np.array([[0,0],[0,1],[0,3],[0,2],[0,5]]),
                'validation_noise':np.array([[3,0],[2,1],[4,3],[1,2],[1,5]])}
        masks,distinct=groups(arrays)
        self.assertEqual(int(distinct.sum()),5)
        self.assertEqual([int(m.sum()) for m in masks.values()],[1,1,1,1,1])
        score=.25*np.log(arrays['counts'].astype(float))
        result=breakdown(score[arrays['validation_pairs'][:,1]],score[arrays['validation_noise']],masks)
        self.assertEqual([result[k]['auc'] for k in masks],[0.,0.,.5,1.,1.])


if __name__=='__main__': unittest.main()
