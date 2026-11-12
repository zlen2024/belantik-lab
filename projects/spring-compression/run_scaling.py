"""Frozen synthetic calibration study; exact solves, not hardware measurements."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import numpy as np
import scipy
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'scaling_results'
SIZES=[16,64,256]
DELTAS=[0.,.005,.01,.03,.05,.1]
SCOPES=['coupling_only','grounding_only','both']
SEEDS=20
TIMES=[10.,100.,1000.,5000.,10000.,100000.]


def root_matrix(n,power):
    # (2I-11^T/n)^power, using its two eigenspaces.
    return 2**power*np.eye(n)+(1-2**power)*np.ones((n,n))/n


def mean_ci(values):
    x=np.asarray(values); mean=float(x.mean())
    half=float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return dict(mean=mean,low=mean-half,high=mean+half,
                median=float(np.median(x)),q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))


def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,mid-half)),float(min(1,mid+half))]


def forces(n):
    rng=np.random.default_rng(310+n)
    f=rng.normal(size=(n,32)); f/=np.linalg.norm(f,axis=0)
    contrast=np.zeros(n); contrast[0]=1/np.sqrt(2); contrast[1]=-1/np.sqrt(2)
    return np.column_stack([np.ones(n)/np.sqrt(n),contrast,f])


def make_full_star(k,g,eta=0.):
    h=np.zeros((len(k)+1,len(k)+1))
    h[:-1,:-1]=np.diag(k+g)
    h[:-1,-1]=h[-1,:-1]=-k
    h[-1,-1]=k.sum()+eta
    return h


def metrics(aeff,invroot,whitened_force,k=None):
    vals,vec=np.linalg.eigh(invroot@aeff@invroot)
    assert vals.min()>0
    inv_r=(vec/vals)@vec.T
    errors=np.linalg.norm((inv_r-np.eye(len(vals)))@whitened_force,axis=0)/np.linalg.norm(whitened_force,axis=0)
    worst=float(np.max(np.abs(1/vals-1)))
    assert np.max(errors)<=worst+1e-10
    c=np.ones(len(vals))/np.sqrt(len(vals))
    response=invroot@(inv_r@c)
    var=c@response
    norm=response@response
    aux=0. if k is None else float(k@response/k.sum())
    return dict(worst_bias=worst,uniform_bias=float(errors[0]),contrast_bias=float(errors[1]),
                random_median=float(np.median(errors[2:])),random_max=float(np.max(errors[2:])),
                tau_r1=float((norm+aux*aux)/var),tau_r16=float((norm+aux*aux/16)/var)), (vals,vec)


def sampling_curve(n,h,r,force,tag):
    mobility=np.ones(len(h))
    if len(h)>n: mobility[-1]=r
    rootm=np.sqrt(mobility)
    vals,vec=np.linalg.eigh(rootm[:,None]*h*rootm[None,:])
    trans=rootm[:,None]*vec
    cov=(trans/vals)@trans.T
    a0=2*np.eye(n)-np.ones((n,n))/n
    xref=.5*force+.5*force.mean(axis=0,keepdims=True)
    mu=cov[:n,:n]@force
    bias=mu-xref
    denom=np.sum(xref*(a0@xref),axis=0)
    bias2=np.sum(bias*(a0@bias),axis=0)/denom
    curves=[]
    for horizon in TIMES:
        factors=2*(vals*horizon+np.expm1(-vals*horizon))/(4*horizon**2*vals**3)
        cb=(trans*factors)@trans.T
        variance=float(np.trace(a0@cb[:n,:n]))/denom
        curves.append(dict(horizon=horizon,bias_squared=bias2.tolist(),sampling_variance=variance.tolist(),
                           total_expected_rmse=np.sqrt(bias2+variance).tolist()))
    return dict(n=n,case=tag,aux_mobility=r,force_labels=['uniform','contrast'],curves=curves)


def main():
    OUT.mkdir(exist_ok=True)
    started=time.perf_counter()
    rows=[]; summaries=[]; analytic=[]; controls=[]; audit_count=0
    for n in SIZES:
        a0=2*np.eye(n)-np.ones((n,n))/n
        invroot=root_matrix(n,-.5)
        force=forces(n); whiteforce=invroot@force
        upper=np.triu_indices(n,1)
        ideal_star=make_full_star(np.ones(n),np.ones(n))
        ideal_eff=ideal_star[:-1,:-1]-np.outer(ideal_star[:-1,-1],ideal_star[-1,:-1])/ideal_star[-1,-1]
        assert np.allclose(ideal_eff,a0,atol=1e-12)
        ideal_metrics,_=metrics(ideal_eff,invroot,whiteforce,np.ones(n))
        assert abs(ideal_metrics['tau_r1']-(1+1/n))<1e-10
        assert abs(ideal_metrics['tau_r16']-(1+1/(16*n)))<1e-10
        for tag,h,r in [('clique_ideal',a0,1),('star_ideal_r1',ideal_star,1),('star_ideal_r16',ideal_star,16)]:
            analytic.append(sampling_curve(n,h,r,force[:,:2],tag))
        for seed in range(SEEDS):
            rng=np.random.default_rng(np.random.SeedSequence([20261112,n,seed]))
            gnoise=rng.uniform(-1,1,n)
            knoise=rng.uniform(-1,1,n)
            edge=np.zeros((n,n)); edge[upper]=rng.uniform(-1,1,len(upper[0]))/n
            edge+=edge.T
            lap_noise=np.diag(edge.sum(axis=1))-edge
            for scope in SCOPES:
                for delta in DELTAS:
                    g=np.ones(n)+(delta*gnoise if scope!='coupling_only' else 0)
                    k=np.ones(n)+(delta*knoise if scope!='grounding_only' else 0)
                    dense=a0+np.diag(g-1)+(delta*lap_noise if scope!='grounding_only' else 0)
                    star=np.diag(g+k)-np.outer(k,k)/k.sum()
                    for name,aeff,weights in [('clique',dense,None),('star',star,k)]:
                        met,spectrum=metrics(aeff,invroot,whiteforce,weights)
                        assert met['worst_bias']<=delta/(1-delta)+1e-10
                        if delta==0: assert met['worst_bias']<1e-10
                        if scope=='coupling_only': assert met['uniform_bias']<1e-10
                        if seed==0 and delta==.1:
                            vals,v=spectrum
                            worst_index=int(np.argmax(abs(1/vals-1)))
                            b=root_matrix(n,.5)@v[:,worst_index]
                            x0=np.linalg.solve(a0,b); xt=np.linalg.solve(aeff,b)
                            direct=np.sqrt((xt-x0)@a0@(xt-x0)/(x0@a0@x0))
                            assert abs(direct-met['worst_bias'])<1e-10
                            if name=='star':
                                full=make_full_star(k,g)
                                sol=np.linalg.solve(full,np.r_[b,0.])[:n]
                                assert np.allclose(sol,xt,atol=1e-10)
                            audit_count+=1
                        rows.append(dict(n=n,seed=seed,scope=scope,delta=delta,architecture=name,**met))
                    if seed==0 and scope=='both' and delta==.1:
                        analytic.append(sampling_curve(n,dense,1,force[:,:2],'clique_noisy_seed0_delta0.1'))
                        for rate in [1,16]:
                            analytic.append(sampling_curve(n,make_full_star(k,g),rate,force[:,:2],f'star_noisy_seed0_delta0.1_r{rate}'))
        for scope in SCOPES:
            for delta in DELTAS:
                for name in ['clique','star']:
                    subset=[x for x in rows if x['n']==n and x['scope']==scope and x['delta']==delta and x['architecture']==name]
                    passed=sum(x['worst_bias']<=.05 for x in subset)
                    summaries.append(dict(n=n,scope=scope,delta=delta,architecture=name,passed=passed,total=SEEDS,
                                          pass_wilson95=wilson(passed,SEEDS),
                                          **{key:mean_ci([x[key] for x in subset]) for key in
                                             ['worst_bias','uniform_bias','contrast_bias','random_median','random_max','tau_r1','tau_r16']}))
        for delta in [.05,.1]:
            for sign in [-1,1]:
                for scope in SCOPES:
                    k=1+(sign*delta if scope!='grounding_only' else 0)
                    g=1+(sign*delta if scope!='coupling_only' else 0)
                    ratios=np.array([g,(g+k)/2])
                    controls.append(dict(n=n,type='common_mode',delta=delta,sign=sign,scope=scope,
                                         worst_bias=float(np.max(abs(1/ratios-1))),
                                         uniform_bias=float(abs(1/g-1))))
        for eta in [0.,.001,.01,.1,1.]:
            full=make_full_star(np.ones(n),np.ones(n),eta)
            eff=full[:n,:n]-np.ones((n,n))/(n+eta)
            met,_=metrics(eff,invroot,whiteforce,np.ones(n))
            expected=eta/(n+2*eta)
            assert abs(met['worst_bias']-expected)<1e-10
            controls.append(dict(n=n,type='auxiliary_ground',eta=eta,worst_bias=met['worst_bias'],formula=expected))
        print('n',n,'done; selected both-error rows:',flush=True)
        for s in summaries:
            if s['n']==n and s['scope']=='both' and s['delta'] in [.05,.1]:
                print(s['architecture'],'delta',s['delta'],'worst median',round(s['worst_bias']['median'],5),
                      'random median',round(s['random_median']['median'],5),'pass',s['passed'],'/',SEEDS,flush=True)
    assert len(rows)==len(SIZES)*len(SCOPES)*len(DELTAS)*2*SEEDS
    data=dict(metadata=dict(date='2026-11-10',python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
              matplotlib=matplotlib.__version__,platform=platform.platform(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256((ROOT/'SCALING_PROTOCOL.md').read_bytes()).hexdigest(),
              command='python -I "'+str(Path(__file__).resolve())+'"',wall_seconds_before_plot=time.perf_counter()-started),
              config=dict(sizes=SIZES,deltas=DELTAS,scopes=SCOPES,seeds=list(range(SEEDS)),random_force_count=32,
                          thermal_trajectory_simulated=False,stationary_time_average_chains=4,sampling_horizons=TIMES),
              verification=dict(direct_worst_direction_checks=audit_count,perturbation_bound_all_pass=True,
                                uniform_coupling_only_all_pass=True),
              per_device=rows,summary=summaries,controls=controls,analytic_sampling=analytic)
    (OUT/'results.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    plot(data)
    report(data)
    print('Saved',OUT,flush=True)


def select(data,n,scope,delta,name):
    return next(x for x in data['summary'] if (x['n'],x['scope'],x['delta'],x['architecture'])==(n,scope,delta,name))


def plot(data):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    cols={'clique':'#2563eb','star':'#dc2626'}
    for name in cols:
        col=cols[name]
        ss=[select(data,256,'both',d,name) for d in DELTAS]
        axes[0,0].plot(100*np.array(DELTAS),[100*x['worst_bias']['median'] for x in ss],'o-',color=col,label=name)
        axes[0,0].fill_between(100*np.array(DELTAS),[100*x['worst_bias']['q05'] for x in ss],
                               [100*x['worst_bias']['q95'] for x in ss],color=col,alpha=.15)
        for delta,style in [(.05,'--'),(.1,'-')]:
            sr=[select(data,n,'both',delta,name) for n in SIZES]
            axes[0,1].plot(SIZES,[100*x['worst_bias']['median'] for x in sr],style+'o',color=col,label=f'{name}, bound {delta:.0%}')
    for ax in axes[0,:]: ax.axhline(5,color='black',ls=':',label='5% bias target')
    axes[0,0].set(xlabel='Component relative-error bound (%)',ylabel='Worst-direction solution bias (%)',title='n=256, coupling + grounding errors\nBand: empirical 5th-95th device percentiles')
    axes[0,1].set(xscale='log',xlabel='Visible nodes n (condition number fixed at 2)',ylabel='Median worst-direction bias (%)',title='Scaling does not remove all-direction bias')
    colors_n={16:'#059669',64:'#7c3aed',256:'#ea580c'}
    for n in SIZES:
        for name,style in [('clique_ideal','-'),('star_ideal_r1','--')]:
            s=next(x for x in data['analytic_sampling'] if x['n']==n and x['case']==name)
            axes[1,0].loglog(TIMES,[100*z['total_expected_rmse'][1] for z in s['curves']],style,color=colors_n[n],label=f'n={n}, {name.split("_")[0]}')
    axes[1,0].axhline(5,color='black',ls=':')
    axes[1,0].set(xlabel='Stationary averaging horizon per chain (4 chains)',ylabel='Expected relative RMSE (%)',title='Analytic thermal error: contrast force, unit norm\nNo large thermal trajectories simulated')
    n=256
    for name,col in cols.items():
        scope_vals=[select(data,n,scope,.1,name)['worst_bias']['median']*100 for scope in SCOPES]
        offset=-.18 if name=='clique' else .18
        axes[1,1].bar(np.arange(3)+offset,scope_vals,width=.36,color=col,label=name)
    axes[1,1].set_xticks(np.arange(3),['Coupling only','Grounding only','Both'])
    axes[1,1].axhline(5,color='black',ls=':')
    axes[1,1].set(ylabel='Median worst-direction bias (%)',title='n=256, 10% bounded independent errors')
    for ax in axes.flat: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Gaussian compression: scaling and synthetic calibration robustness\nExact device solves | 20 devices per condition | weights normalized to hold conditioning fixed',fontsize=12)
    fig.savefig(OUT/'scaling.png',dpi=170)
    plt.close(fig)


def report(data):
    lines=['# Scaling dan calibration error','',
           '2026-11-10. Synthetic-device study dengan exact linear algebra; bukan hardware measurement.','',
           '## Setup','',
           'n=16,64,256; clique weight 1/n; star weight 1; g=1. A=2I-11^T/n, condition number 2. Normalisasi ini berbeza daripada eksperimen asal n=4 weight1.',
           'Bound delta merujuk kepada Uniform[-delta,+delta] relative component errors, bukan standard deviation. 20 seed per condition; grounding paired. Force directions random tetap untuk semua devices.',
           'Worst-direction relative A-norm solution bias dikira untuk semua b melalui eigenvalues. Ia sama dengan whitened spectral covariance error.','',
           '## Hasil utama: coupling dan grounding kedua-duanya tidak tepat','',
           '| n | Bound | Sistem | Median worst bias | Median random-force bias | Devices lulus <=5% |',
           '|---:|---:|---|---:|---:|---:|']
    for n in SIZES:
        for delta in [.05,.1]:
            for name in ['clique','star']:
                s=select(data,n,'both',delta,name)
                lines.append(f"| {n} | {delta:.0%} | {name} | {s['worst_bias']['median']:.2%} | {s['random_median']['median']:.2%} | {s['passed']}/{s['total']} |")
    lines+=['','Pass rate ialah antara synthetic devices sahaja; Wilson 95% intervals ada dalam JSON. Zero failures dalam 20 devices tidak menjamin tiada hardware failures.',
            'Mean 95% CI dan empirical 5/95 percentiles disimpan berasingan dalam JSON. Semua delta/scopes yang berkongsi calibration draws berkorelasi.','',
            '## Sensitivity mengikut komponen, n=256 dan bound 10%','',
            '| Scope | Clique median worst bias | Star median worst bias |','|---|---:|---:|']
    for scope in SCOPES:
        vals=[select(data,256,scope,.1,name)['worst_bias']['median'] for name in ['clique','star']]
        lines.append(f'| {scope} | {vals[0]:.2%} | {vals[1]:.2%} |')
    lines+=['','Coupling-only mengekalkan uniform solution tepat kerana semua grounding kekal 1. Ujian uniform sahaja tidak mengesan coupling error ini.',
            '## Common-mode dan auxiliary parasitic grounding','',
            'Jika semua coupling dan grounding tersasar -10% bersama, A_actual=.9 A: relative solution bias 11.111% untuk kedua-dua architecture, semua n. Ini tidak hilang melalui iid averaging atau thermal averaging.',
            'Auxiliary grounding asal sifar: ralat multiplicative sahaja tidak mengujinya. Ujian tambahan additive eta menghasilkan uniform bias eta/(n+2*eta). Formula awal diperbetulkan selepas assertion Schur-complement gagal; perubahan dicatat dalam protokol.','',
            '| n | Auxiliary parasitic ground eta | Worst bias |','|---:|---:|---:|']
    for s in data['controls']:
        if s['type']=='auxiliary_ground' and s['eta'] in [.01,.1]:
            lines.append(f"| {s['n']} | {s['eta']} | {s['worst_bias']:.3%} |")
    lines+=['','## Thermal sampling: ramalan analitik, bukan trajectory measurements','',
            'Primary calibration study mengasingkan bias menggunakan exact solves. Thermal noise diperkirakan secara berasingan menggunakan exact covariance continuous-time stationary average, 4 chains.',
            'Unit-norm force ditetapkan; semakin banyak coordinates, semakin besar jumlah noise vector relatif kepada signal ini. Ini bergantung pada force normalization dan bukan universal scaling bagi setiap workload.','',
            '| n | Sistem | Expected uniform RMSE T=5000 | Expected contrast RMSE T=5000 |','|---:|---|---:|---:|']
    for s in data['analytic_sampling']:
        if s['case'] in ['clique_ideal','star_ideal_r1']:
            z=next(x for x in s['curves'] if x['horizon']==5000)
            lines.append(f"| {s['n']} | {s['case']} | {z['total_expected_rmse'][0]:.2%} | {z['total_expected_rmse'][1]:.2%} |")
    lines+=['','Untuk noisy seed0 bound10%, JSON memisahkan bias_squared, sampling_variance, dan total_expected_rmse pada setiap horizon. Seed0 ialah ilustrasi ditetapkan, bukan typical-device estimate.',
            'Bias calibration tidak turun dengan T. Formula error decomposition ialah exact expectation: total MSE = squared bias + variance, dalam A-norm.',
            'Ideal star mean-mode tau=1+1/(n*r): penalti mengecil dengan n untuk family ini, tetapi auxiliary degree=n dan fastest mode meningkat. Ini tidak membuktikan scalability hardware.','',
            '## Semakan dan batas','',
            f"{data['verification']['direct_worst_direction_checks']} independent direct-solve worst-direction checks lulus; star full-vs-Schur solves, ideal invariants, perturbation bound dan coupling-only uniform invariants lulus.",
            'Semua bounded calibration cases memenuhi worst bias <=delta/(1-delta). Formula ini membekalkan guarantee konservatif; pass fraction hasil synthetic model memberi bukti lebih terhad.',
            'Dense eigensolver digunakan untuk audit, bukan comparison runtime. Digital clique baseline mempunyai structured O(n) solve. Tiada training, optimization, large thermal trajectories, chip energy, maximum-degree routing atau circuit validation.',
            f"Wall-clock sebelum plot: {data['metadata']['wall_seconds_before_plot']:.2f} saat.",
            '', '## Fail','', '- SCALING_PROTOCOL.md: protokol sebelum run.', '- run_scaling.py: kod.',
            '- scaling_results/results.json: metadata/hash, 2160 device-condition records, summary, controls dan exact sampling predictions.',
            '- scaling_results/scaling.png: figure.',
            '', 'Formula dan hasil ialah derivasi serta numerical experiments tempatan. Calibration distribution sebenar belum diukur; tiada dakwaan toleransi hardware.']
    (ROOT/'SCALING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
