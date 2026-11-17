"""Evaluate lexical pairs once per distinct pair; preserve mathematical ties.

Historical raw evaluators remain unchanged so earlier runs remain reproducible.
This module is the default evaluation interface for new text8 analyses.
"""
from __future__ import annotations
import numpy as np
from word_energy.text8_lexical_models import pair_metrics
from word_energy.ising.exact_training import _imports


def stable_tied_ising_scores(raw,pairs,coupling_scale=1.0):
    """Normalized Ising F2 log overlap via log1p/expm1, exact zero at W=0."""
    jax,jnp,_=_imports()
    pairs=jnp.asarray(pairs,dtype=jnp.int32)
    states=jnp.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]],dtype=jnp.float64)
    b=2.5*jnp.tanh(raw['biases'])
    k=2.0*jnp.tanh(raw['internal'])
    g=1.5*jax.nn.sigmoid(raw['interaction'])*coupling_scale
    unary=states@b.T+(states[:,0]*states[:,1])[:,None]*k[None,:]
    probabilities=jax.nn.softmax(unary,axis=0).T
    products=states[:,None,:]*states[None,:,:]
    perturbation=jnp.expm1(jnp.sum(products*g,axis=-1))
    left=probabilities[pairs[:,0]]; right=probabilities[pairs[:,1]]
    return jnp.log1p(jnp.einsum('bi,ij,bj->b',left,perturbation,right))


def stable_ising_score_adapter(params,pairs,noise):
    """Signature compatible with canonical_candidate_scores."""
    jax,jnp,_=_imports()
    pairs=jnp.asarray(pairs,dtype=jnp.int32); noise=jnp.asarray(noise,dtype=jnp.int32)
    left=jnp.broadcast_to(pairs[:,0,None],noise.shape)
    negatives=jnp.stack((left,noise),axis=-1).reshape((-1,2))
    return stable_tied_ising_scores(params,pairs),stable_tied_ising_scores(params,negatives).reshape(noise.shape)


def canonical_candidate_scores(params,score_fn,pairs,noise,*,symmetric=False,zero_coupling=False,chunk_size=512):
    pairs=np.asarray(pairs,dtype=np.int32); noise=np.asarray(noise,dtype=np.int32)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or noise.ndim!=2 or len(noise)!=len(pairs):
        raise ValueError('invalid candidate arrays')
    if len(pairs)==0 or np.any(pairs<0) or np.any(noise<0) or chunk_size<=0:
        raise ValueError('empty/invalid candidates')
    candidates=np.concatenate((pairs[:,1,None],noise),axis=1)
    centers=np.broadcast_to(pairs[:,0,None],candidates.shape)
    raw_pairs=np.stack((centers,candidates),axis=-1).reshape(-1,2)
    if symmetric:
        raw_pairs=np.sort(raw_pairs,axis=1)
    unique,inverse=np.unique(raw_pairs,axis=0,return_inverse=True)
    if zero_coupling:
        if not symmetric: raise ValueError('zero-coupling analytical shortcut applies only to Ising')
        unique_scores=np.zeros(len(unique),dtype=np.float64)
    else:
        blocks=[]
        for start in range(0,len(unique),chunk_size):
            block=unique[start:start+chunk_size]; size=len(block)
            if size<chunk_size:
                block=np.concatenate((block,np.repeat(block[-1:],chunk_size-size,axis=0)),axis=0)
            positive,_=score_fn(params,block,block[:,1,None])
            blocks.append(np.asarray(positive,dtype=np.float64)[:size])
        unique_scores=np.concatenate(blocks)
    if not np.isfinite(unique_scores).all(): raise FloatingPointError('nonfinite score')
    scores=unique_scores[inverse].reshape(candidates.shape)
    return scores[:,0],scores[:,1:]


def evaluate(params,score_fn,pairs,noise,center_counts,*,symmetric=False,zero_coupling=False):
    positive,negative=canonical_candidate_scores(params,score_fn,pairs,noise,symmetric=symmetric,zero_coupling=zero_coupling)
    return pair_metrics(positive,negative,center_counts)
