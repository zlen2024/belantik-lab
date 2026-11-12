"""Independent post-run checks of calibration records and OU matrix integration."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent


def main():
    d=json.loads((ROOT/'scaling_results/results.json').read_text())
    assert d['metadata']['source_sha256']==hashlib.sha256((ROOT/'run_scaling.py').read_bytes()).hexdigest()
    assert d['metadata']['protocol_sha256']==hashlib.sha256((ROOT/'SCALING_PROTOCOL.md').read_bytes()).hexdigest()
    assert len(d['per_device'])==2160
    keyed={(x['n'],x['seed'],x['scope'],x['delta'],x['architecture']):x for x in d['per_device']}
    for x in d['per_device']:
        assert x['random_max']<=x['worst_bias']+1e-10
        assert x['worst_bias']<=x['delta']/(1-x['delta'])+1e-10
        if x['scope']=='grounding_only':
            twin=keyed[(x['n'],x['seed'],x['scope'],x['delta'],'star' if x['architecture']=='clique' else 'clique')]
            assert abs(x['worst_bias']-twin['worst_bias'])<1e-10
    for s in d['analytic_sampling']:
        for z in s['curves']:
            assert np.allclose(np.array(z['total_expected_rmse'])**2,
                               np.array(z['bias_squared'])+np.array(z['sampling_variance']),atol=1e-12)
        if s['case']=='clique_ideal':
            n=s['n']
            for z in s['curves']:
                t=z['horizon']
                trace=2*(t+np.expm1(-t))/(4*t*t)+(n-1)*(2*t+np.expm1(-2*t))/(8*t*t)
                assert np.allclose(z['sampling_variance'],[trace,2*trace],rtol=1e-10)
    # Different formula: integrate covariance in original coordinates using
    # F = T B^-1 - B^-2 (I-exp(-BT)), B=MH, instead of eigenmode factors.
    n=16; a=2*np.eye(n)-np.ones((n,n))/n
    h=np.zeros((n+1,n+1)); h[:n,:n]=2*np.eye(n)
    h[:n,n]=h[n,:n]=-1; h[n,n]=n
    integration_checks=[]
    for rate in [1,16]:
        mobility=np.diag(np.r_[np.ones(n),rate]); drift=mobility@h
        c=np.linalg.inv(h); invb=np.linalg.inv(drift)
        for t in [10.,5000.]:
            f=t*invb-invb@invb@(np.eye(n+1)-expm(-drift*t))
            avg=(f@c+c@f.T)/(4*t*t)
            trace=float(np.trace(a@avg[:n,:n]))
            saved=next(s for s in d['analytic_sampling'] if s['n']==n and s['case']==f'star_ideal_r{rate}')
            saved=next(z for z in saved['curves'] if z['horizon']==t)
            error=float(np.max(abs(np.array(saved['sampling_variance'])-np.array([trace,2*trace]))))
            assert error<1e-10
            integration_checks.append(dict(n=n,rate=rate,horizon=t,absolute_discrepancy=error))
    audit=dict(record_count=2160,hashes_verified=True,grounding_only_architectures_identical=True,
               perturbation_bounds_verified=True,mse_decomposition_verified=True,
               independent_matrix_integral_checks=integration_checks)
    (ROOT/'scaling_results/audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
