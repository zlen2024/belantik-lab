# Protokol eksperimen spring compression

Tarikh: 2026-11-10. Ditetapkan sebelum run utama.

Skop: eksperimen 1 (algebra) dan 2 (Gaussian OU); tiada latihan Ising,
hardware, tenaga elektrik, atau eksperimen scaling. NumPy/SciPy float64.
THRML Ising tidak sesuai untuk pemboleh ubah berterusan ini. Torx tidak
diperlukan kerana rujukan exact OU kecil boleh dibina terus dengan eigendecomposition.

## Hipotesis dan setup

- Clique K4, edge weight 1, grounding 1 pada setiap visible node.
- Star, edge weight 4, grounding 1 hanya pada empat visible; auxiliary tiada daya.
- Cycle C4, edge weight 4/3, grounding sama.
- kBT=1, mobility=I, friction convention dq=-H(q-mu)dt+sqrt(2)dW.
- Daya unit norm: uniform (1,1,1,1)/2; low (1,0,-1,0)/sqrt(2);
  high (1,-1,1,-1)/2. b=0 hanya untuk centered covariance/ACF.
- Star marginal sepatutnya sama dengan clique; cycle berbeza mengikut eigenmode.
- Integral ACF y=sum(x)/2: clique 1; star 1.25; cycle 1.

## Sampel dan uncertainty

Seeds 0..9, empat chain bebas setiap seed, dt=0.05, 100000 samples/chain
(5000 unit masa), stationary initialisation daripada Gaussian exact.
PRNG SeedSequence mengasingkan sistem, seed, dan eksperimen.
Exact OU: eigen-coordinate u'=rho*u+sqrt((1-rho^2)/lambda)*normal.
Stationary law var(u)=1/lambda. Tiada Euler timestep bias; dt tetap
menentukan resolusi observation dan ESS per retained sample.

Trajectory centered yang sama digunakan untuk tiga b dalam setiap sistem,
kerana covariance bebas b. Ketiga-tiga arah bukan replicate bebas.
Mean/covariance errors dibandingkan dengan equilibrium sendiri DAN clique.
ACF dianggarkan selepas subtract sample mean setiap chain, diintegrasi dengan
trapezoid hingga lag 10 unit masa (fixed window), kemudian dibandingkan dengan
ramalan exact window yang sama serta integral continuous infinity.
Laporkan Student-t 95% CI merentas 10 seed, bukan merentas time samples.
ACF y^2-Var(y) turut diperiksa kerana covariance estimation boleh mempunyai ESS berbeza.

Relaksasi: 10 seeds x 4 chains, mula visible=5, auxiliary=5 jika ada;
dt=0.05 hingga t=10. Bandingkan ensemble mean y dengan exact OU mean dan
95% CI daripada seed means. Jangan anggap crossing noise sebagai mixing time.

## Metrik dan keputusan

Algebra: SPD, Schur complement, visible inverse block, solution directions,
generalised spectral ratios. Toleransi relative 1e-10.
Sampling: relative energy-norm mean error; covariance whitened terhadap target
dan rujukan asal (spectral norm); ACF, ESS, RMSE curve.
Primary solution metric: relative A-norm RMSE terhadap clique asal pada bajet
masa tetap, pooled 4 chains per seed; analytic expected RMSE juga dihitung.
Laporkan masa horizon per chain dan aggregate chain-time=4*horizon.
Time-to-accuracy menggunakan expected RMSE 5%, fixed time grid sehingga 5000;
ini stationary-start Gaussian expectation, bukan hardware runtime atau guarantee
per run. Kos burn-in dikecualikan daripada metrik ini dan diuji berasingan.
Cycles dengan bias floor >5% tidak boleh mencapai sasaran melalui averaging.

Tidak tuntut chip speedup daripada wall-clock simulator. Matrix exponential /
eigendecomposition ialah alat rujukan exact, bukan implementasi sparse hardware.
Simpan source hash, versions, command, konfigurasi, per-seed metrics, curves,
figure, report. Sebarang perubahan selepas run mesti dinyatakan.

## Sumber

- https://threeplusone.com/pubs/Aifer2024a.pdf (Gaussian equilibrium, eq. 2-4)
- https://arxiv.org/abs/1102.2950 (Schur complement/Kron)
- https://arxiv.org/abs/0803.0929 (spectral sparsification)
- Identiti star, OU ACF, dan finite-sample RMSE ialah derivasi untuk setup ini.
