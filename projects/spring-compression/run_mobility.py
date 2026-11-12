"""Exact OU auxiliary-mobility experiment; see MOBILITY_PROTOCOL.md."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import numpy as np
import scipy
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.signal import lfilter
from scipy.stats import chi2, t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'mobility_results'
DT, N, CHAINS, SEEDS, MAXLAG = .05, 100000, 4, 10, 800
A = 5*np.eye(4)-np.ones((4,4))
H = np.zeros((5,5)); H[:4,:4] = 5*np.eye(4)
H[:4,4] = H[4,:4] = -4; H[4,4] = 16
TARGET = np.linalg.inv(A)
w0,v0 = np.linalg.eigh(A)
ROOT_A = (v0*np.sqrt(w0))@v0.T


def ci(x):
    x = np.asarray(x)
    m = x.mean(axis=0)
    d = student_t.ppf(.975,len(x)-1)*x.std(axis=0,ddof=1)/np.sqrt(len(x))
    return dict(mean=m.tolist(),low=(m-d).tolist(),high=(m+d).tolist())


def acf(y):
    y = y-y.mean(axis=1,keepdims=True)
    f = np.fft.rfft(y,n=2*N,axis=1)
    cov = np.fft.irfft(f*f.conj(),n=2*N,axis=1)[:,:MAXLAG+1]
    cov /= N-np.arange(MAXLAG+1)
    cov = cov.mean(axis=0)
    return cov/cov[0]


def spectral(h,m):
    root = np.diag(np.sqrt(m))
    lam,v = np.linalg.eigh(root@h@root)
    transform = root@v
    return lam,transform


def simulate(lam,transform,rng):
    u0 = rng.normal(size=(CHAINS,len(lam)))/np.sqrt(lam)
    u = np.empty((CHAINS,N,len(lam)))
    for j,l in enumerate(lam):
        rho = np.exp(-l*DT)
        noise = rng.normal(size=(CHAINS,N))*np.sqrt(-np.expm1(-2*l*DT)/l)
        u[:,:,j],_ = lfilter([1.],[1.,-rho],noise,axis=1,zi=(rho*u0[:,j])[:,None])
    return u@transform.T


def run():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    cases = [('clique','baseline',None,1.,1.)]
    for regime,ratios in [('fixed_visible',[.25,1,2,4,16]),('fixed_trace',[.25,1,4,16])]:
        for r in ratios:
            a = 1. if regime=='fixed_visible' else 4/(4+r)
            cases.append((f'{regime}_r{r:g}',regime,r,a,a*r))
    results,records = [],[]
    for index,(name,regime,r,a,b) in enumerate(cases):
        h = A if regime=='baseline' else H
        mobility = np.full(len(h),a); mobility[-1] = a if regime=='baseline' else b
        m = np.diag(mobility); cov = np.linalg.inv(h)
        lam,trans = spectral(h,mobility)
        assert np.all(lam>0)
        assert np.allclose(trans@np.diag(1/lam)@trans.T,cov,atol=1e-11)
        drift = m@h
        assert np.allclose(drift@cov+cov@drift.T,2*m,atol=1e-11)
        p = expm(-drift*DT)
        q = trans@np.diag(-np.expm1(-2*lam*DT)/lam)@trans.T
        assert np.allclose(p@cov@p.T+q,cov,atol=1e-11)
        c = np.r_[np.full(4,.5),np.zeros(len(h)-4)]
        weights = (trans.T@c)**2/lam
        exact_ac = np.exp(-np.outer(np.arange(MAXLAG+1)*DT,lam))@weights/weights.sum()
        tau = float(np.sum(weights/lam)/weights.sum())
        direct_tau = float(c@cov@np.linalg.solve(m,cov@c)/(c@cov@c))
        formula = 1. if regime=='baseline' else 1/a+1/(4*b)
        assert np.isclose(tau,formula,rtol=1e-11)
        assert np.isclose(tau,direct_tau,rtol=1e-11)
        rho = np.exp(-lam*DT)
        factor = (N*(1+rho)/(1-rho)-2*rho*(-np.expm1(-lam*DT*N))/(1-rho)**2)/N**2
        cm = (trans@np.diag(factor/lam/CHAINS)@trans.T)[:4,:4]
        # Independently sum all finite observation lags, avoiding closed-form formula.
        lag = np.arange(1,N)
        factors_sum = np.array([(N+2*np.dot(N-lag,np.exp(-l*DT*lag)))/N**2/l/CHAINS for l in lam])
        cm_check = (trans@np.diag(factors_sum)@trans.T)[:4,:4]
        assert np.allclose(cm,cm_check,rtol=1e-10,atol=1e-13)
        acs,taus,errs,means = [],[],[],[]
        for seed in range(SEEDS):
            rng = np.random.default_rng(np.random.SeedSequence([20261111,index,seed]))
            state = simulate(lam,trans,rng)[:,:,:4]
            mean = state.mean(axis=(0,1))
            flat = state.reshape(-1,4)-mean
            empirical_cov = flat.T@flat/len(flat)
            err = float(np.linalg.norm(ROOT_A@empirical_cov@ROOT_A-np.eye(4),ord=2))
            ac = acf(state.sum(axis=2)/2)
            measured_tau = float(np.trapezoid(ac,dx=DT))
            acs.append(ac); taus.append(measured_tau); errs.append(err); means.append(mean)
            records.append(dict(case=name,seed=seed,centered_mean=mean.tolist(),tau_window=measured_tau,covariance_error=err))
        means = np.array(means)
        statistic = float(np.einsum('bi,ij,bj->',means,np.linalg.inv(cm),means))
        pvalue = float(min(1.,2*min(chi2.cdf(statistic,40),chi2.sf(statistic,40))))
        measured = ci(taus)
        window_tau = float(np.trapezoid(exact_ac,dx=DT))
        result = dict(case=name,regime=regime,ratio=r,visible_mobility=a,auxiliary_mobility=b,
                      mobility_trace=float(mobility.sum()),auxiliary_noise_scale=float(np.sqrt(b)),
                      lambda_min=float(lam.min()),lambda_max=float(lam.max()),euler_stability_ceiling=float(2/lam.max()),
                      tau_exact=tau,tau_exact_window=window_tau,tau_measured=measured,
                      exact_in_measured_marginal_95_ci=bool(measured['low']<=window_tau<=measured['high']),
                      ess_rate_relative_to_clique=1/tau,covariance_error=ci(errs),mean_gaussian_two_sided_p=pvalue,
                      acf_exact=exact_ac.tolist(),acf_measured=ci(acs))
        results.append(result)
        print(name,'tau exact',round(tau,6),'measured',measured,'cov err',round(np.mean(errs),5),flush=True)
    bad_m = np.diag([1.,1.,1.,1.,4.])
    bad_cov = solve_continuous_lyapunov(bad_m@H,2*np.eye(5))
    bad_error = float(np.linalg.norm(ROOT_A@bad_cov[:4,:4]@ROOT_A-np.eye(4),ord=2))
    assert bad_error>1e-3
    data = dict(metadata=dict(date='2026-11-10',python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                 matplotlib=matplotlib.__version__,platform=platform.platform(),
                 command='python -I "'+str(Path(__file__).resolve())+'"',
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 protocol_sha256=hashlib.sha256((ROOT/'MOBILITY_PROTOCOL.md').read_bytes()).hexdigest(),
                 wall_seconds_before_plot=time.perf_counter()-start),
                 config=dict(dt=DT,samples=N,chains=CHAINS,seeds=list(range(SEEDS)),acf_window=DT*MAXLAG),
                 summary=results,per_seed=records,
                 negative_control=dict(aux_mobility=4,incorrect_noise='sqrt(2)I',visible_covariance_error=bad_error))
    (OUT/'results.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    plot(results)
    report(data)
    print('Saved',OUT,flush=True)


def plot(results):
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    r = np.geomspace(.1,64,300)
    colors = dict(fixed_visible='#2563eb',fixed_trace='#dc2626')
    for regime,label,pred in [('fixed_visible','Visible mobility fixed',1+1/(4*r)),
                             ('fixed_trace','Total mobility fixed at 4',(4+r)/4*(1+1/(4*r)))]:
        col = colors[regime]
        subset = [x for x in results if x['regime']==regime]
        xs = np.array([x['ratio'] for x in subset])
        ys = np.array([x['tau_measured']['mean'] for x in subset])
        lo = np.array([x['tau_measured']['low'] for x in subset]); hi = np.array([x['tau_measured']['high'] for x in subset])
        axes[0,0].plot(r,pred,color=col,label=label)
        axes[0,0].errorbar(xs,ys,yerr=np.array([ys-lo,hi-ys]),fmt='o',color=col,capsize=3)
        axes[0,1].plot(r,1/pred,color=col,label=label)
        axes[1,1].loglog(xs,[x['lambda_max'] for x in subset],'o-',color=col,label=label)
    axes[0,0].axhline(1,color='black',ls=':',label='Clique baseline')
    axes[0,0].set(xscale='log',yscale='log',xlabel='Auxiliary / visible mobility r',ylabel='Integrated ACF time',title='Exact predictions and measured 95% CI')
    axes[0,1].axhline(1,color='black',ls=':')
    axes[0,1].set(xscale='log',xlabel='Auxiliary / visible mobility r',ylabel='ESS per physical time / clique',title='Predicted efficiency for mean mode only')
    for name,col in [('clique','#111827'),('fixed_visible_r1','#dc2626'),('fixed_visible_r16','#2563eb')]:
        item = next(x for x in results if x['case']==name)
        tt = np.arange(MAXLAG+1)*DT
        axes[1,0].plot(tt,item['acf_exact'],color=col,label=name)
        axes[1,0].plot(tt,item['acf_measured']['mean'],'--',color=col)
    axes[1,0].set(xlim=(0,6),xlabel='Lag (physical time)',ylabel='ACF',title='Faster auxiliary approaches clique dynamics')
    axes[1,1].axhline(5,color='black',ls=':',label='Clique fastest rate')
    axes[1,1].set(xlabel='Auxiliary / visible mobility r',ylabel='Largest decay rate',title='Fast modes grow; exact stepping hides stiffness cost')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Auxiliary mobility: reduced sampling penalty, conditional resource tradeoff\nGaussian OU, kBT=1 | trace(mobility) is a proxy, not power',fontsize=12)
    fig.savefig(OUT/'mobility.png',dpi=170)
    plt.close(fig)


def report(data):
    lines=['# Keputusan mobility auxiliary','',
           '2026-11-10. Eksperimen susulan; keputusan asal tidak diubah.','',
           'Model: dq=-MH(q-mu)dt+sqrt(2M)dW; kBT=1. M mengubah drift dan noise bersama. Semua stationary covariance ialah H^-1.','',
           '10 seed x 4 chain x 100000 samples per case, dt=.05. Exact OU transitions. ACF window 40. CI Student-t marginal 95% merentas seed, bukan simultaneous.','',
           '## Hasil','',
           '| Regime | r | a visible | b auxiliary | tau exact | tau measured [95% CI] | ESS rate / clique (theory) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for x in data['summary']:
        c=x['tau_measured']
        lines.append(f"| {x['regime']} | {x['ratio']} | {x['visible_mobility']:.4f} | {x['auxiliary_mobility']:.4f} | {x['tau_exact']:.5f} | {c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}] | {x['ess_rate_relative_to_clique']:.4f} |")
    lines += ['', '## Derivasi dan tafsiran','',
              'Untuk c=(1/2,1/2,1/2,1/2,0), tau=c^T H^-1 M^-1 H^-1 c/(c^T H^-1 c)=1/a+1/(4b).',
              'Dengan a=1, b=r: tau=1+1/(4r). Pada r=16, tau=1.015625: hanya 1.5625% melebihi clique, dan predicted ESS rate meningkat 23.08% berbanding star r=1. Ini khusus mean-mode stationary dalam physical-time unit yang sama.',
              'Tiada finite r dengan a=1 mengatasi clique dalam metrik ini. CI mungkin tidak dapat membezakan r besar daripada clique; perbezaan kecil ditentukan formula, bukan dakwaan significance Monte Carlo.',
              'Dengan 4a+b=4: tau=17/16+r/4+1/(4r), minimum 25/16 pada r=1. Maka mempercepat auxiliary sambil mengurangkan mobility visible tidak menghasilkan kemenangan dalam proxy ini.',
              'trace(M) bukan watt, chip area, atau harga komponen. Bajet ini hanya ujian sensitivity terhadap cara sumber diperuntukkan.','',
              '## Kos yang masih belum diukur','',
              '- Untuk r=16 dan a=1, trace(M)=20 berbanding clique 4; amplitude noise auxiliary 4 kali baseline. Ini tuntutan parameter dinamik, bukan tenaga elektrik 5 kali.',
              '- Fast relaxation rate meningkat. Ia boleh menuntut resolusi temporal lebih tinggi dalam model numerik biasa atau sistem fizikal; tiada spesifikasi circuit diukur.',
              '- Exact OU integration menghapuskan Euler bias tetapi tidak membuktikan implementasi hardware dapat mencapai kadar tersebut.',
              '- Friction/mobility, damping, noise bandwidth, auxiliary capacitance/mass, calibration dan energy per answer perlu model litar atau hardware sebelum menilai kos sebenar.',
              '- Tiada pengukuran burn-in atau solution RMSE baharu; eksperimen ini hanya stationary ACF dan covariance.','',
              '## Verifikasi','',
              'SPD, fluctuation-dissipation Lyapunov identity, exact-step stationarity, tau via independent inverse dan spectral formula, serta finite-sample covariance via explicit lag sum semuanya lulus.',
              'Mean Gaussian chi-square diagnostics serta covariance errors direkodkan dalam JSON; non-rejection bukan bukti semua aspek sampler betul.',
              f"Negative control r=4, noise tidak diselaraskan: covariance visible whitening error {data['negative_control']['visible_covariance_error']:.4%}. Ini menunjukkan drift-only rescaling menukar equilibrium.",
              '', '## Artifak','', '- MOBILITY_PROTOCOL.md: setup sebelum run.', '- run_mobility.py: kod.',
              '- mobility_results/results.json: metadata, hashes, full per-seed records, exact curves, CI.',
              '- mobility_results/mobility.png: figure.',
              f"- Wall-clock sebelum plot: {data['metadata']['wall_seconds_before_plot']:.2f} saat; bukan physical horizon.",
              '', 'Formula dan angka di atas ialah derivasi serta pengukuran tempatan untuk model yang ditetapkan, bukan benchmark chip.']
    (ROOT/'MOBILITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    run()
