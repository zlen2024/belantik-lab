"""Canonical center-candidate scoring for symmetric two-context windows."""
import numpy as np

def canonical_scores(params,score,windows,noise,chunk_size=512):
    w=np.asarray(windows,dtype=np.int32);noise=np.asarray(noise,dtype=np.int32)
    if w.ndim!=2 or w.shape[1]!=3 or noise.ndim!=2 or len(noise)!=len(w) or not len(w) or chunk_size<1:raise ValueError('invalid windows/noise')
    if np.any(w<0) or np.any(noise<0):raise ValueError('negative token ID')
    contexts=np.sort(w[:,1:],axis=1);candidates=np.concatenate((w[:,0,None],noise),axis=1)
    packed=np.empty((len(w),candidates.shape[1],3),dtype=np.int32);packed[:,:,0]=candidates;packed[:,:,1:]=contexts[:,None,:]
    unique,inverse=np.unique(packed.reshape((-1,3)),axis=0,return_inverse=True);blocks=[]
    for i in range(0,len(unique),chunk_size):
        b=unique[i:i+chunk_size];size=len(b)
        if size<chunk_size:b=np.concatenate((b,np.repeat(b[-1:],chunk_size-size,axis=0)))
        blocks.append(np.asarray(score(params,b),dtype=np.float64)[:size])
    vals=np.concatenate(blocks)
    if not np.isfinite(vals).all():raise ValueError('nonfinite scores')
    out=vals[inverse].reshape(candidates.shape);return out[:,0],out[:,1:]
