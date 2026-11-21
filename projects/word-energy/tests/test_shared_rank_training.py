import numpy as np
from word_energy.contextual_substitution import ising_shared_correlations
from word_energy.ising.exact_training import _imports
from word_energy.shared_rank_training import shared_pair_scores, paired_rank_loss, validate_training_batch


def fixture():
    rng = np.random.default_rng(127)
    return {'biases': rng.normal(0,.2,(6,4)), 'internal': rng.normal(0,.2,(6,6)),
            'interaction': rng.normal(0,.2,4)}


def test_shared_pair_matches_exact_readout_and_permutation():
    params = fixture()
    rows = np.array([[0,1,2,3],[0,0,0,0],[3,2,3,5]], dtype=np.int32)
    scores = np.asarray(shared_pair_scores(params, rows))
    expected = np.array([ising_shared_correlations(params,int(t),[int(c)],np.array([a,b])) for t,c,a,b in rows])
    np.testing.assert_allclose(scores,expected,atol=1e-12,rtol=0)
    np.testing.assert_allclose(np.asarray(shared_pair_scores(params,rows[:,[0,1,3,2]])),scores[:,::-1],atol=1e-12,rtol=0)


def test_loss_gradient_against_finite_difference():
    jax,jnp,_ = _imports()
    p = {k:jnp.asarray(v) for k,v in fixture().items()}
    rows = np.array([[0,1,2,3],[3,2,3,5]], dtype=np.int32)
    f = lambda v: paired_rank_loss(v, rows, [True,False], .2)
    grad = jax.grad(f)(p)
    for name,index in [('biases',(0,0)),('internal',(0,0)),('interaction',(0,))]:
        plus,minus=dict(p),dict(p)
        plus[name]=p[name].at[index].add(1e-5)
        minus[name]=p[name].at[index].add(-1e-5)
        numeric=(float(f(plus))-float(f(minus)))/2e-5
        np.testing.assert_allclose(float(grad[name][index]),numeric,atol=1e-7,rtol=1e-4)


def test_host_batch_validation_rejects_ambiguous_labels_and_ids():
    import pytest
    rows = np.array([[0,1,2,3]], dtype=np.int32)
    validate_training_batch(rows, np.array([True]), .2, 6)
    for bad_rows, bad_labels, bad_temp in [
        (rows, np.array([[True]]), .2), (rows, np.array([1]), .2),
        (np.array([[0,1,2,2]]), np.array([True]), .2),
        (np.array([[0,1,-1,3]]), np.array([True]), .2),
        (rows, np.array([True]), 0.), (rows, np.array([True]), np.inf),
    ]:
        with pytest.raises(ValueError):
            validate_training_batch(bad_rows,bad_labels,bad_temp,6)
