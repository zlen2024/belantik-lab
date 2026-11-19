"""Interpretability diagnostics for the 16-state unary and joint posterior."""
from itertools import combinations
import numpy as np
from scipy.special import expit,softmax
STATES=np.array([[2*((i>>j)&1)-1 for j in range(4)] for i in range(16)],dtype=np.float64)
PAIRS=tuple(combinations(range(4),2));PRODUCTS=np.stack([STATES[:,i]*STATES[:,j] for i,j in PAIRS],axis=1)

def strict_local_maxima(logits):
    logits=np.asarray(logits,dtype=float)
    if logits.ndim!=2 or logits.shape[1]!=16:raise ValueError('expected [words,16] unary logits')
    neighbors=np.arange(16)[:,None]^(1<<np.arange(4))[None,:]
    return np.all(logits[:,:,None]>logits[:,neighbors],axis=-1)

def diagnostics(params,a,vocabulary):
    raw={k:np.asarray(v) for k,v in params.items()};w=a['validation_windows']
    b=2.5*np.tanh(raw['biases']);k=2*np.tanh(raw['internal']);g=1.5*expit(raw['interaction'])
    logits=b@STATES.T+k@PRODUCTS.T;q=softmax(logits,axis=1);center=q[w[:,0]]
    cross=np.exp(np.sum(STATES[:,None,:]*STATES[None,:,:]*g/2,axis=-1));d1=q[w[:,1]]@cross.T;d2=q[w[:,2]]@cross.T
    post=center*d1*d2;post/=post.sum(axis=1,keepdims=True)
    joint=np.log(np.sum(center*d1*d2,axis=1));add=np.log(np.sum(center*d1,axis=1))+np.log(np.sum(center*d2,axis=1))
    minima=strict_local_maxima(logits);counts=minima.sum(axis=1);hist=np.bincount(counts,minlength=17)
    bank=vocabulary.index('bank') if 'bank' in vocabulary else None
    unary_ent=-np.sum(center*np.log(center),axis=1);post_ent=-np.sum(post*np.log(post),axis=1)
    return {'unary_entropy_mean':float(unary_ent.mean()),'posterior_entropy_mean':float(post_ent.mean()),'unary_effective_states_mean':float(np.exp(unary_ent).mean()),
      'posterior_effective_states_mean':float(np.exp(post_ent).mean()),'posterior_unary_tv_mean':float((.5*np.abs(post-center).sum(axis=1)).mean()),
      'joint_minus_add_score_mean':float(np.mean(joint-add)),'joint_minus_add_score_abs_mean':float(np.mean(np.abs(joint-add))),
      'replace_second_with_first_abs_score_delta_mean':float(np.mean(np.abs(joint-np.log(np.sum(center*d1*d1,axis=1))))),
      'couplings':g.tolist(),'cross_near_saturation_count':int(np.sum(g>1.49)),
      'internal_near_saturation_fraction':float(np.mean(np.abs(k)>1.99)),
      'strict_unary_minima_histogram':{str(i):int(v) for i,v in enumerate(hist) if v},
      'strict_unary_multi_minima_fraction':float(np.mean(counts>=2)),
      'bank_strict_unary_minimum_states':np.flatnonzero(minima[bank]).tolist() if bank is not None else None,
      'local_minimum_definition':'strict unary-energy local minimum relative to four one-spin flips; no context and no sense interpretation',
      'readout':'joint star; counterfactual for additive-trained parameters'}
