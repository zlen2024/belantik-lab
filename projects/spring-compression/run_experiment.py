"""Bounded exact Gaussian experiments. See PROTOCOL.md for frozen choices."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import numpy as np
import scipy
from scipy.signal import lfilter
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
DT, STEPS, CHAINS, SEEDS, LAG = 0.05, 100000, 4, 10, 200
TIMES = np.array([.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
B = np.array([[1, 1, 1, 1], [1, 0, -1, 0], [1, -1, 1, -1]], dtype=float)
B /= np.linalg.norm(B, axis=1)[:, None]
LABELS = ['uniform', 'low', 'high']
A = 5*np.eye(4) - np.ones((4, 4))
C0 = np.linalg.inv(A)
ev, vec = np.linalg.eigh(A)
ASQ = (vec*np.sqrt(ev))@vec.T


def matrices():
    star = np.zeros((5, 5))
    star[:4, :4] = 5*np.eye(4)
    star[:4, 4] = star[4, :4] = -4
    star[4, 4] = 16
    cycle = np.eye(4)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        edge = np.eye(4)[i] - np.eye(4)[j]
        cycle += 4/3*np.outer(edge, edge)
    return {'clique': A, 'star': star, 'cycle': cycle}


def ci(values):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    half = student_t.ppf(.975, len(values)-1)*values.std(axis=0, ddof=1)/np.sqrt(len(values))
    return {'mean': mean.tolist(), 'low': (mean-half).tolist(), 'high': (mean+half).tolist()}


def sample_ou(h, rng, steps, initial=None):
    lam, v = np.linalg.eigh(h)
    rho = np.exp(-lam*DT)
    u0 = (rng.normal(size=(CHAINS, len(lam)))/np.sqrt(lam)
          if initial is None else np.tile(initial@v, (CHAINS, 1)))
    u = np.empty((CHAINS, steps, len(lam)))
    for j in range(len(lam)):
        noise = rng.normal(size=(CHAINS, steps))*np.sqrt(-np.expm1(-2*lam[j]*DT)/lam[j])
        u[:, :, j], _ = lfilter([1.], [1., -rho[j]], noise, axis=1,
                                zi=(rho[j]*u0[:, j])[:, None])
    return u@v.T


def acf(values):
    z = values-values.mean(axis=1, keepdims=True)
    n = z.shape[1]
    spectrum = np.fft.rfft(z, n=2*n, axis=1)
    cov = np.fft.irfft(spectrum*spectrum.conj(), n=2*n, axis=1)[:, :LAG+1]
    cov /= n-np.arange(LAG+1)
    pooled = cov.mean(axis=0)
    return pooled/pooled[0]


def cov_error(empirical, reference):
    lam, v = np.linalg.eigh(reference)
    invsqrt = (v/np.sqrt(lam))@v.T
    return float(np.linalg.norm(invsqrt@empirical@invsqrt-np.eye(4), ord=2))


def exact_mean_cov(h, n):
    lam, v = np.linalg.eigh(h)
    r = np.exp(-lam*DT)
    # Variance of a stationary average of n observations, then CHAINS averages.
    factor = (n*(1+r)/(1-r)-2*r*(-np.expm1(-lam*DT*n))/(1-r)**2)/n**2
    return (v*(factor/lam/CHAINS))@v.T


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    systems = matrices()
    checks, records, curves, results = {}, [], {}, {}
    original_mu = B@C0
    for index, (name, h) in enumerate(systems.items()):
        lam, v = np.linalg.eigh(h)
        inv = np.linalg.inv(h)
        cov = inv[:4, :4]
        mu = B@cov
        effective = h[:4, :4] if len(h) == 4 else h[:4, :4]-np.outer(h[:4, 4], h[4, :4])/h[4, 4]
        inv_asq = np.linalg.inv(ASQ)
        ratios = np.linalg.eigvalsh(inv_asq@effective@inv_asq)
        assert lam.min() > 0
        if name in ('clique', 'star'):
            assert np.linalg.norm(effective-A)/np.linalg.norm(A) < 1e-10
            assert np.linalg.norm(cov-C0)/np.linalg.norm(C0) < 1e-10
        expected = np.ones(3) if name != 'cycle' else np.array([1., 15/11, 15/19])
        amps = np.array([m@b/(o@b) for m, o, b in zip(mu, original_mu, B)])
        assert np.allclose(amps, expected, atol=1e-10, rtol=1e-10)
        # Verify the exact OU transition preserves its intended covariance.
        m = (v*np.exp(-lam*DT))@v.T
        q = (v*(-np.expm1(-2*lam*DT)/lam))@v.T
        assert np.allclose(m@inv@m.T+q, inv, atol=1e-12)
        checks[name] = dict(min_eigenvalue=float(lam.min()), spectral_ratios=ratios.tolist(),
                            schur_relative_error=float(np.linalg.norm(effective-A)/np.linalg.norm(A)),
                            mean_amplitude_ratios=amps.tolist())
        c = np.zeros(len(h)); c[:4] = .5
        weights = (v.T@c)**2/lam
        rho_exact = np.exp(-np.outer(np.arange(LAG+1)*DT, lam))@weights/weights.sum()
        tau_cont = float((weights/lam).sum()/weights.sum())
        tau_window = float(np.trapezoid(rho_exact, dx=DT))
        tau_discrete = float(DT/2*(weights/np.tanh(lam*DT/2)).sum()/weights.sum())
        assert abs(tau_cont-({'star': 1.25}.get(name, 1.))) < 1e-10
        y_acfs, square_acfs, rmses, relax = [], [], [], []
        for seed in range(SEEDS):
            rng = np.random.default_rng(np.random.SeedSequence([20261110, index, seed, 0]))
            states = sample_ou(h, rng, STEPS)
            visible = states[:, :, :4]
            means = visible.mean(axis=(0, 1))
            flat = visible.reshape(-1, 4)
            empirical_cov = (flat-means).T@(flat-means)/len(flat)
            y = visible.sum(axis=2)/2
            ac = acf(y)
            ac2 = acf(y*y)
            tau = float(np.trapezoid(ac, dx=DT))
            tau2 = float(np.trapezoid(ac2, dx=DT))
            y_acfs.append(ac); square_acfs.append(ac2)
            empirical_errors = []
            for direction in range(3):
                error = means+mu[direction]-original_mu[direction]
                denom = original_mu[direction]@A@original_mu[direction]
                empirical_errors.append(float(np.sqrt(error@A@error/denom)))
            records.append(dict(system=name, seed=seed, tau_y=tau, tau_y_squared=tau2,
                                centered_mean=means.tolist(),
                                ess_y=CHAINS*STEPS*DT/(2*tau),
                                mean_relative_errors=empirical_errors,
                                mean_own_absolute_A_norm=float(np.linalg.norm(ASQ@means)),
                                covariance_own_error=cov_error(empirical_cov, cov),
                                covariance_original_error=cov_error(empirical_cov, C0)))
            cumsum = visible.sum(axis=0).cumsum(axis=0)
            error_by_time = []
            for timepoint in TIMES:
                n = round(timepoint/DT)
                noise_mean = cumsum[n-1]/(n*CHAINS)
                error = noise_mean[None, :]+mu-original_mu
                error_by_time.append(np.einsum('bi,ij,bj->b', error, A, error)/
                                     np.einsum('bi,ij,bj->b', original_mu, A, original_mu))
            rmses.append(error_by_time)
            rrng = np.random.default_rng(np.random.SeedSequence([20261110, index, seed, 1]))
            transient = sample_ou(h, rrng, 200, np.full(len(h), 5.))
            relax.append(np.r_[10., transient[:, :, :4].sum(axis=2).mean(axis=0)/2])
        analytic_rmse = []
        for timepoint in TIMES:
            cm = exact_mean_cov(h, round(timepoint/DT))[:4, :4]
            bias = mu-original_mu
            analytic_rmse.append(np.sqrt((np.einsum('bi,ij,bj->b', bias, A, bias)+np.trace(A@cm))/
                                        np.einsum('bi,ij,bj->b', original_mu, A, original_mu)))
        analytic_rmse = np.array(analytic_rmse)
        relaxation_time = np.arange(201)*DT
        exact_relaxation = np.exp(-np.outer(relaxation_time, lam))@((v.T@c)*(v.T@np.full(len(h), 5.)))
        own = [r for r in records if r['system'] == name]
        results[name] = dict(
            tau_y_exact_continuous=tau_cont, tau_y_exact_observation_grid=tau_discrete,
            tau_y_exact_window=tau_window, tau_y_measured=ci([r['tau_y'] for r in own]),
            tau_y_squared_exact_window=float(np.trapezoid(rho_exact**2, dx=DT)),
            tau_y_squared_measured=ci([r['tau_y_squared'] for r in own]),
            ess_y=ci([r['ess_y'] for r in own]),
            covariance_own_error=ci([r['covariance_own_error'] for r in own]),
            covariance_original_error=ci([r['covariance_original_error'] for r in own]),
            expected_rmse_final=analytic_rmse[-1].tolist(),
            measured_rmse_final=np.sqrt(np.mean(np.array(rmses)[:, -1, :], axis=0)).tolist(),
            first_grid_time_expected_rmse_5percent={lab: (float(TIMES[np.where(analytic_rmse[:, j] <= .05)[0][0]])
                if np.any(analytic_rmse[:, j] <= .05) else None) for j, lab in enumerate(LABELS)})
        curves[name] = dict(acf_exact=rho_exact.tolist(), acf_measured=ci(y_acfs),
                            acf_squared_measured=ci(square_acfs),
                            rmse_exact=analytic_rmse.tolist(), rmse_empirical=np.sqrt(np.mean(rmses, axis=0)).tolist(),
                            mse_ci=ci(rmses), relaxation_exact=exact_relaxation.tolist(), relaxation=ci(relax))
        print(name, json.dumps(results[name]), flush=True)
    payload = dict(metadata=dict(date='2026-11-10', python=sys.version, numpy=np.__version__,
                   scipy=scipy.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(),
                   command='python -I "'+str(Path(__file__).resolve())+'"',
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
                   wall_seconds=time.perf_counter()-started),
                   config=dict(dt=DT, steps=STEPS, chains=CHAINS, seeds=list(range(SEEDS)),
                               lag_window=LAG*DT, times=TIMES.tolist()),
                   algebra=checks, summary=results, per_seed=records, curves=curves)
    (OUT/'results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plot(curves)
    report(payload)
    print('Saved', OUT, flush=True)


def plot(curves):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = dict(clique='#2563eb', star='#dc2626', cycle='#059669')
    for name, d in curves.items():
        color = colors[name]
        lag = np.arange(LAG+1)*DT
        axes[0, 0].plot(lag, d['acf_exact'], color=color, label=name+' exact')
        axes[0, 0].plot(lag, d['acf_measured']['mean'], '--', color=color)
        axes[0, 0].fill_between(lag, d['acf_measured']['low'], d['acf_measured']['high'], color=color, alpha=.12)
        tt = np.arange(201)*DT
        axes[0, 1].plot(tt, d['relaxation_exact'], color=color, label=name)
        axes[0, 1].plot(tt, d['relaxation']['mean'], '--', color=color)
        axes[0, 1].fill_between(tt, d['relaxation']['low'], d['relaxation']['high'], color=color, alpha=.12)
        for ax, direction in [(axes[1, 0], 0), (axes[1, 1], 1)]:
            ax.loglog(TIMES, np.array(d['rmse_exact'])[:, direction], color=color, label=name)
            ax.loglog(TIMES, np.array(d['rmse_empirical'])[:, direction], '.', color=color)
    axes[0, 0].set(title='Stationary mean-mode autocorrelation', xlabel='Lag (physical time)', ylabel='ACF')
    axes[0, 0].set_xlim(0, 6)
    axes[0, 1].set(title='Relaxation from visible = auxiliary = 5', xlabel='Physical time', ylabel='Mean y = sum(x)/2')
    for ax, title in [(axes[1, 0], 'Solution RMSE: uniform force'), (axes[1, 1], 'Solution RMSE: low-mode force')]:
        ax.set(title=title, xlabel='Horizon per chain (4 chains averaged)', ylabel='Relative A-norm RMSE vs clique')
        ax.axhline(.05, color='gray', linestyle=':', label='5% target')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Gaussian spring compression: exact law does not imply faster sampling\nLines: theory | dashed/dots: measurements | bands: 95% CI over 10 seeds', fontsize=12)
    fig.savefig(OUT/'experiment.png', dpi=170)
    plt.close(fig)


def report(data):
    lines = ['# Keputusan: spring compression', '',
             'Tarikh 2026-11-10. Eksperimen 1 dan 2 selesai; bukan ukuran hardware.', '',
             'Semua invariant algebra lulus (float64, toleransi 1e-10).', '',
             '## Autocorrelation mean mode', '',
             '| Sistem | Integral exact continuous | Anggaran fixed window, 95% CI | Covariance error vs equilibrium sendiri |',
             '|---|---:|---:|---:|']
    for name, r in data['summary'].items():
        c = r['tau_y_measured']
        lines.append(f"| {name} | {r['tau_y_exact_continuous']:.4f} | {c['mean']:.4f} [{c['low']:.4f}, {c['high']:.4f}] | {r['covariance_own_error']['mean']:.4f} |")
    lines += ['', 'Covariance error ialah spectral norm selepas whitening. CI merentas seed; ACF menggunakan window 10, bukan integral infinity. Exact observation-grid/window tersedia dalam JSON.', '',
              '## Ralat solution akhir terhadap clique asal', '',
              '| Sistem | Uniform RMSE | Low-mode RMSE | High-mode RMSE |', '|---|---:|---:|---:|']
    for name, r in data['summary'].items():
        lines.append('| '+name+' | '+' | '.join(f'{x:.3%}' for x in r['measured_rmse_final'])+' |')
    lines += ['', 'RMSE empirikal merentas 10 seed; setiap seed menggabungkan 4 chain, horizon 5000 per chain. Ini bukan CI per run.', '',
              'Jangan tafsir nilai RMSE star yang sedikit lebih rendah dalam run ini sebagai kelebihan. Expected RMSE analitik uniform ialah 1.266% (clique) dan 1.361% (star); variasi 10 seed boleh membalikkan ranking empirikal. Arah b berkongsi centered trajectory, maka error antaranya bukan replicate bebas.', '',
              '## Tafsiran', '',
              '- Star mengekalkan marginal visible tepat, tetapi tau mean mode meningkat 25% secara analitik dengan mobility=I.',
              '- Cycle mempunyai bias low-mode 4/11 dan high-mode -4/19 pada amplitude mean; averaging tidak menghapuskannya.',
              '- Mean dan covariance empirikal dibandingkan dengan equilibrium sistem sendiri. Cycle betul sebagai sampler bagi sistem baharu walaupun bias terhadap masalah asal.',
              '- Relaksasi daripada initial state jauh diperiksa berasingan. Plot tidak digunakan untuk menentukan mixing time melalui crossing noisy.',
              '- Sasaran 5% dinilai melalui expected RMSE stationary pada fixed grid; masa burn-in dan hardware programming tidak termasuk.',
              '- Tempoh fizikal simulasi berbeza daripada wall-clock Python. Tiada bukti joule atau chip speedup.', '',
              '## Artifak', '', '- PROTOCOL.md: pilihan sebelum run.', '- run_experiment.py: implementasi.',
              '- results/results.json: konfigurasi, hashes, versions, algebra, 30 per-seed records, curves dan CI.',
              '- results/experiment.png: plot.', '',
              f"Masa komputasi hingga JSON: {data['metadata']['wall_seconds']:.2f} saat.", '',
              'Sumber teori: [Aifer](https://threeplusone.com/pubs/Aifer2024a.pdf), [Kron](https://arxiv.org/abs/1102.2950), [sparsification](https://arxiv.org/abs/0803.0929).',
              'Angka eksperimen ialah ukuran script tempatan; ramalan OU/star ialah derivasi untuk setup ini.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
