"""Protocol and isolation checks for fresh synthetic Ising corpora."""
import unittest
from unittest.mock import patch
import numpy as np

from word_energy.fresh_synthetic import REPLICATES, make_fresh_corpus
from word_energy.ising.exact_training import ExactTrainingConfig, evaluate_checkpoint, prepare_data, train_smoke
from word_energy.ising.spectral_initializer import spectral_biases
from word_energy.synthetic import vocabulary


class FreshSyntheticTest(unittest.TestCase):
    def test_split_isolation_reproducibility_and_legacy_hash(self):
        legacy=prepare_data()
        self.assertEqual(legacy['protocol_sha256'],'ced585cb88b56356f37c885a970a6202ee7c1739585c9cdbe8bdaf3830c49a7c')
        namespaces=[]; train_hashes=[]
        for replicate in REPLICATES:
            corpus=make_fresh_corpus(replicate); again=make_fresh_corpus(replicate)
            self.assertEqual(corpus['pair_index_splits'],again['pair_index_splits'])
            self.assertEqual([len(corpus['splits'][n]) for n in ('train','validation','test')],[48,18,18])
            self.assertTrue(set(corpus['vocabulary'][2:]).isdisjoint(vocabulary()))
            for family,splits in corpus['pair_index_splits'].items():
                self.assertEqual(len(set(splits['train']) & set(splits['validation'])),0)
                self.assertEqual(len(set(splits['train']) & set(splits['test'])),0)
                self.assertEqual(len(set(splits['validation']) & set(splits['test'])),0)
                self.assertEqual(len(set(splits['train']+splits['validation']+splits['test'])),28)
            for before in namespaces: self.assertTrue(set(corpus['vocabulary'][2:]).isdisjoint(before))
            namespaces.append(set(corpus['vocabulary'][2:]))
            d=prepare_data(corpus=corpus); train_hashes.append(d['protocol_sha256'])
            self.assertNotEqual(d['protocol_sha256'],legacy['protocol_sha256'])
            spectral=spectral_biases(d['train_windows'],len(d['ids']))
            self.assertEqual(int(spectral['counts'].sum()),48*6)
            for family in ('F','R','M'):
                for token in corpus['pools'][family]:
                    self.assertGreaterEqual(int((d['train_windows']==d['ids'][token]).sum()),2)
        self.assertEqual(len(set(train_hashes)),len(REPLICATES))

    def test_family_partitions_do_not_all_share_same_pair_indices(self):
        corpus=make_fresh_corpus(REPLICATES[0]); parts=corpus['pair_index_splits']
        self.assertNotEqual(parts['F']['train'],parts['R']['train'])

    def test_injected_protocol_is_used_by_training_diagnosis_and_evaluation(self):
        from scripts.diagnose_ising_validation import diagnose_raw
        data=prepare_data(corpus=make_fresh_corpus(REPLICATES[0]))
        with patch('word_energy.ising.exact_training.prepare_data',side_effect=AssertionError('legacy data accessed')), patch('scripts.diagnose_ising_validation.prepare_data',side_effect=AssertionError('legacy data accessed')):
            raw, metrics=train_smoke(ExactTrainingConfig(seed=251,lexical_updates=1,joint_updates=1,validation_every=1,normalize_spokes=True),data_override=data)
            diag=diagnose_raw(raw,251,'injection-test','fixture',normalize_spokes=True,data_override=data)
            evaluated=evaluate_checkpoint(raw,'validation',normalize_spokes=True,data_override=data)
        self.assertEqual(metrics['protocol.sha256'],data['protocol_sha256'])
        self.assertEqual(diag['protocol_sha256'],data['protocol_sha256'])
        self.assertAlmostEqual(diag['validation_macro_sense_accuracy'],evaluated['validation.macro_sense_accuracy'])

if __name__=='__main__': unittest.main()
