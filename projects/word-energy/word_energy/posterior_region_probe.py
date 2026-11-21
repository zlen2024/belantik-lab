"""Target marginal and candidate-derived regions for an exact Ising star."""
import numpy as np
from scipy.special import expit,logsumexp
from word_energy.wikitext_four_spin_wic import STATES,unary

def posterior_and_regions(params, target_ids, contexts, candidate_a, candidate_b):
    """Four cue slots; ID len(q) denotes uniform dummy. Candidate slots are neutral."""
    q=unary(params);n=len(q)
    t=np.asarray(target_ids);c=np.asarray(contexts);a=np.asarray(candidate_a);b=np.asarray(candidate_b)
    if t.ndim!=1 or c.shape!=(len(t),4) or a.shape!=t.shape or b.shape!=t.shape:
        raise ValueError('unexpected batch shape')
    if any(x.dtype.kind not in 'iu' for x in (t,c,a,b)) or np.any(t<0) or np.any(t>=n) or np.any(c<0) or np.any(c>n) or np.any(a<0) or np.any(a>=n) or np.any(b<0) or np.any(b>=n) or np.any(a==b):
        raise ValueError('invalid IDs')
    expanded=np.vstack((q,np.full((1,16),1/16)))
    g=1.5*expit(params['interaction'])/6
    edge=(STATES*g)@STATES.T
    log_messages=logsumexp(edge[None,:,:]+np.log(expanded)[:,None,:],axis=2)
    logits=np.log(q[t])+np.sum(log_messages[c],axis=1)
    posterior=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
    region=(q[a]>q[b]).astype(float)+.5*(q[a]==q[b])
    return posterior,q[t],region,q[a],q[b]
