"""FFT equivalent of existing rank-split Rhat/ESS/MCSE diagnostic."""
import numpy as np
from scipy.stats import rankdata,norm

def split_rank_rhat_ess_fft(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or values.shape[0]<2 or values.shape[1]<4:raise ValueError('need at least two chains and four draws')
    half=values.shape[1]//2;split=np.concatenate((values[:,:half],values[:,-half:]),axis=0)
    ranked=rankdata(split.reshape(-1),method='average').reshape(split.shape);normal=norm.ppf((ranked-.5)/ranked.size)
    within=np.mean(np.var(normal,axis=1,ddof=1))
    if within<=1e-15:return float('inf'),0.,float('inf')
    draws=normal.shape[1];chains=normal.shape[0];between=draws*np.var(np.mean(normal,axis=1),ddof=1)
    variance=(draws-1)*within/draws+between/draws;rh=float(np.sqrt(variance/within))
    centered=normal-np.mean(normal,axis=1,keepdims=True);f=np.fft.rfft(centered,n=2*draws,axis=1)
    ac=np.fft.irfft(f*np.conj(f),n=2*draws,axis=1)[:,:draws]/(draws-np.arange(draws))[None,:]
    autocov=np.mean(ac,axis=0);corr=1-(within-autocov)/variance;corr[0]=1.;tau=1.
    for lag in range(1,draws-1,2):
        pair=corr[lag]+corr[lag+1]
        if pair<=0:break
        tau+=2*pair
    ess=float(min(chains*draws,chains*draws/max(tau,1.)))
    mcse=float(np.std(values.reshape(-1),ddof=1)/np.sqrt(ess)) if ess>0 else float('inf')
    return rh,ess,mcse
