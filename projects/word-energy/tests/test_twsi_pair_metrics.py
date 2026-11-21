import numpy as np
from scripts.train_twsi_pair_pilot import metrics

def test_pair_ties_and_bidirectional_strict_group():
    rows=np.array([[1,2,3,4,1],[1,5,3,4,0],[7,2,8,9,1]],dtype=np.int32)
    scores=np.array([[2.,1.],[1.,2.],[0.,0.]])
    result=metrics(scores,rows)
    assert result['pair_accuracy']==2.5/3
    assert result['bidirectional_groups']==1
    assert result['bidirectional_all_strict_accuracy']==1.
    scores[1]=[1.,1.]
    assert metrics(scores,rows)['bidirectional_all_strict_accuracy']==0.
