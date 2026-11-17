import unittest
import tempfile
from pathlib import Path
import zipfile
import numpy as np
from word_energy.text8_development import ROOT, make_pairs, noise_rows, region_tokens, sha_array
from word_energy.text8_lexical_models import pair_metrics


class Text8DevelopmentTests(unittest.TestCase):
    def test_region_trims_only_partial_words(self):
        temp_root=ROOT/'.tmp'; temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path=Path(directory)/'fixture.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('text8',b'alpha beta gamma delta')
            self.assertEqual(region_tokens(path,2,13),['beta'])
            self.assertEqual(region_tokens(path,6,10),['beta'])

    def test_positions_preserve_oov_gaps_and_split_boundary(self):
        ids=np.asarray([0,-1,-1,1,2,-1,3,4],dtype=np.int32)
        pairs,positions,stats=make_pairs(ids,1000,np.random.default_rng(13))
        np.testing.assert_array_equal(pairs[:,0],ids[positions[:,0]])
        np.testing.assert_array_equal(pairs[:,1],ids[positions[:,1]])
        self.assertTrue(np.all(np.isin(positions[:,1]-positions[:,0],[-2,-1,1,2])))
        self.assertFalse(np.any(pairs<0))
        self.assertEqual(stats['oov_tokens'],3)
        self.assertFalse(np.any(positions[:,0]==0))

    def test_noise_is_unconditional_and_reproducible(self):
        positive=np.array([0,1,1],dtype=np.int32)
        q=np.array([1.,0.])
        a=noise_rows(q,positive,5,np.random.default_rng(11))
        b=noise_rows(q,positive,5,np.random.default_rng(11))
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.any(a==positive[:,None]))
        self.assertEqual(sha_array(a),sha_array(b))

    def test_pair_metrics_handle_ties(self):
        m=pair_metrics(np.array([1.,0.]),np.array([[1.,0.],[1.,0.]]),np.array([3,8]))
        self.assertEqual(m['noise_pairwise_auc'],.5)
        self.assertAlmostEqual(m['noise_mrr_at_21'],(0.75+(1/2+1/3)/2)/2)


if __name__=='__main__': unittest.main()
