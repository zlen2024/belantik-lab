"""Post-run numerical audit; does not change the frozen experiment protocol."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import expm
from scipy.stats import chi2

root = Path(__file__).resolve().parent
data = json.loads((root/'results/results.json').read_text())
assert data['metadata']['source_sha256'] == hashlib.sha256((root/'run_experiment.py').read_bytes()).hexdigest()
assert data['metadata']['protocol_sha256'] == hashlib.sha256((root/'PROTOCOL.md').read_bytes()).hexdigest()
n = data['config']['steps']; dt = data['config']['dt']; chains = data['config']['chains']
output = {}
for name, dim, edges, weight in [
    ('clique', 4, [(i,j) for i in range(4) for j in range(i+1,4)], 1.),
    ('star', 5, [(i,4) for i in range(4)], 4.),
    ('cycle', 4, [(0,1),(1,2),(2,3),(3,0)], 4/3)]:
    h = np.diag([1.]*4+[0.]*(dim-4))
    for i,j in edges:
        e = np.eye(dim)[i]-np.eye(dim)[j]
        h += weight*np.outer(e,e)
    c = np.linalg.solve(h, np.eye(dim))
    lam,v = np.linalg.eigh(h)
    k = np.arange(1,n)
    factors = np.array([(n+2*np.dot(n-k, np.exp(-l*dt*k)))/n**2/chains/l for l in lam])
    cm = ((v*factors)@v.T)[:4,:4]
    records = [r for r in data['per_seed'] if r['system']==name]
    means = np.array([r['centered_mean'] for r in records])
    stat = float(np.einsum('bi,ij,bj->', means, np.linalg.inv(cm), means))
    df = len(means)*4
    p_two_sided = float(min(1., 2*min(chi2.cdf(stat,df), chi2.sf(stat,df))))
    y = np.r_[np.full(4,.5),np.zeros(dim-4)]
    ac_at_one = float(y@expm(-h)@c@y/(y@c@y))
    assert np.isclose(ac_at_one, data['curves'][name]['acf_exact'][round(1/dt)], atol=1e-12)
    tau = float(y@c@c@y/(y@c@y))
    assert np.isclose(tau,data['summary'][name]['tau_y_exact_continuous'],atol=1e-12)
    ci = data['summary'][name]['tau_y_measured']
    output[name] = dict(gaussian_mean_chi_square=stat, df=df,
                       mean_two_sided_p=p_two_sided,
                       exact_window_tau_inside_95percent_ci=bool(ci['low']<=data['summary'][name]['tau_y_exact_window']<=ci['high']),
                       direct_matrix_tau=tau)
assert len(data['per_seed']) == 30
assert all(np.isfinite(r['mean_two_sided_p']) for r in output.values())
(root/'results/audit.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
print('Hashes, independent spring assembly, matrix-exponential ACF and direct inverse tau verified.')
