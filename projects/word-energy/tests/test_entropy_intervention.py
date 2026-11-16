import unittest
from unittest.mock import patch
import numpy as np

from word_energy.ising.entropy_intervention import entropy_kl_to_uniform, schedule_weight
from word_energy.ising.exact_training import _imports, ExactTrainingConfig, prepare_data, train_smoke
from word_energy.fresh_synthetic import make_fresh_corpus
from scripts.run_ising_spectral_ablation import random_initial


class EntropyInterventionTest(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual([schedule_weight(x) for x in (-1,0,49,50,99,100)], [0,1,1,1,.02,0])
        self.assertAlmostEqual(sum(schedule_weight(u) for u in range(400)), sum(schedule_weight(u-100) for u in range(400)))

    def test_gradient_target_finite_difference_and_symmetry(self):
        jax,jnp,_ = _imports()
        raw = {'biases':jnp.array([[.2,-.4],[.3,.2]]), 'internal':jnp.array([.1,-.3]), 'interaction':jnp.array([.4,.8])}
        fun = lambda x: entropy_kl_to_uniform(x,0)
        grad=jax.grad(fun)(raw)
        self.assertTrue(np.all(np.asarray(grad['biases'])[1] == 0))
        self.assertEqual(float(grad['internal'][1]),0)
        self.assertTrue(np.all(np.asarray(grad['interaction']) == 0))
        eps=1e-5
        plus={k:v for k,v in raw.items()}; minus={k:v for k,v in raw.items()}
        plus['biases']=raw['biases'].at[0,0].add(eps); minus['biases']=raw['biases'].at[0,0].add(-eps)
        self.assertAlmostEqual(float((fun(plus)-fun(minus))/(2*eps)),float(grad['biases'][0,0]),places=8)
        uniform={'biases':jnp.zeros((2,2)),'internal':jnp.zeros(2),'interaction':jnp.ones(2)}
        self.assertAlmostEqual(float(fun(uniform)),0,places=12)
        self.assertLess(np.linalg.norm(np.asarray(jax.grad(fun)(uniform)['biases'])),1e-12)
        for axes in ((0,1),(1,0)):
            for signs in ((-1,-1),(-1,1),(1,-1),(1,1)):
                flip={'biases':raw['biases'][:,jnp.array(axes)]*jnp.array(signs), 'internal':raw['internal']*np.prod(signs), 'interaction':raw['interaction'][jnp.array(axes)]}
                self.assertAlmostEqual(float(fun(flip)),float(fun(raw)),places=12)

    def test_zero_intervention_parity(self):
        data=prepare_data(corpus=make_fresh_corpus(227)); initial=random_initial(251,len(data['ids']))
        cfg=ExactTrainingConfig(seed=251,lexical_updates=20,joint_updates=0,normalize_spokes=True)
        snapshots=[]
        for intervention in (None,{'target_id':data['ids']['bank'],'offset':0,'strength':0.0}):
            record={}
            train_smoke(cfg,initial_override=initial,data_override=data,lexical_intervention=intervention,
                        snapshot_callback=lambda stage,step,raw: record.update({f'{stage}_{step}':raw}))
            snapshots.append(record)
        for key in ('biases','internal','interaction'):
            np.testing.assert_array_equal(snapshots[0]['lexical_20'][key],snapshots[1]['lexical_20'][key])

    def test_bonus_not_evaluated_during_joint(self):
        data=prepare_data(corpus=make_fresh_corpus(227)); initial=random_initial(251,len(data['ids']))
        cfg=ExactTrainingConfig(seed=251,lexical_updates=0,joint_updates=1,normalize_spokes=True,validation_every=1)
        with patch('word_energy.ising.entropy_intervention.entropy_kl_to_uniform',side_effect=AssertionError('bonus leaked')):
            train_smoke(cfg,initial_override=initial,data_override=data,lexical_intervention={'target_id':data['ids']['bank'],'offset':0,'strength':0.1})

    def test_active_then_off_uses_pure_lexical_step(self):
        from word_energy.ising import exact_training
        data=prepare_data(corpus=make_fresh_corpus(227)); initial=random_initial(251,len(data['ids']))
        cfg=ExactTrainingConfig(seed=251,lexical_updates=2,joint_updates=0,normalize_spokes=True,validation_every=2)
        names=[]; original=exact_training._optimizer_step
        def spy(loss,*args):
            names.append(loss.__name__)
            return original(loss,*args)
        with patch('word_energy.ising.entropy_intervention.schedule_weight',side_effect=lambda u: 1.0 if u==0 else 0.0), patch('word_energy.ising.exact_training._optimizer_step',side_effect=spy):
            train_smoke(cfg,initial_override=initial,data_override=data,lexical_intervention={'target_id':data['ids']['bank'],'offset':0,'strength':0.1})
        self.assertIn('augmented_loss',names)
        self.assertIn('lexical_loss',names)


if __name__ == '__main__': unittest.main()
