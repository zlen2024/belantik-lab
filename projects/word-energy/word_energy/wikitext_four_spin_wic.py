"""Exact 16-state center posterior for pair-trained/four-spin Ising."""
from itertools import combinations
import numpy as np
from scipy.special import expit,softmax
STATES=np.array([[2*((i>>j)&1)-1 for j in range(4)] for i in range(16)],dtype=np.float64)
PRODUCTS=np.stack([STATES[:,i]*STATES[:,j] for i,j in combinations(range(4),2)],axis=1)

def unary(params):
    logits=(2.5*np.tanh(params['biases']))@STATES.T+(2*np.tanh(params['internal']))@PRODUCTS.T
    return softmax(logits,axis=1)

def posterior(q,g,target,context):
    if not context:raise ValueError('empty context')
    g=np.asarray(g,dtype=float)/len(context);cross=np.exp(np.sum(STATES[:,None,:]*STATES[None,:,:]*g,axis=-1))
    p=q[target].copy()
    for c in sorted(context):p*=cross@q[c]
    total=p.sum()
    if not np.isfinite(total) or total<=0:raise ValueError('invalid posterior')
    return p/total

def star(params,target,context):return posterior(unary(params),1.5*expit(params['interaction']),target,context)
