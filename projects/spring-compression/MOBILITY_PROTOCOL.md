# Protokol mobility auxiliary

2026-11-10. Ditetapkan sebelum run mobility; susulan eksperimen pertama.
Tiada perubahan kepada protokol atau hasil eksperimen pertama.

## Model dan hipotesis

H star, grounding, suhu, dan definisi y=sum(x)/2 sama seperti eksperimen pertama.
M=diag(a,a,a,a,b), r=b/a. Dinamik dq=-MH(q-mu)dt+sqrt(2M)dW.
Drift DAN noise berubah bersama untuk mengekalkan covariance H^-1.
Transform q=sqrt(M)u menjadikan stiffness S=sqrt(M)Hsqrt(M) symmetric SPD,
dan u mengikuti OU mobility identity. Sampling eigenmode exact; bukan Euler.

Derivasi untuk setup ini: tau_y=c^T H^-1 M^-1 H^-1 c/(c^T H^-1 c)
=1/a+1/(4b). Baseline clique a=1 mempunyai tau_y=1.

Regime utama: a=1, r=[0.25,1,2,4,16]. Hipotesis tau=1+1/(4r),
sentiasa >1 bagi r finite tetapi mendekati clique apabila r meningkat.

Regime proxy resource: trace(M)=4, a=4/(4+r), b=ar,
r=[0.25,1,4,16]. Ini hanyalah bajet jumlah mobility, BUKAN watt atau area.
Hipotesis tau=(4+r)/4*(1+1/(4r)); minimum 25/16 pada r=1.
Analytic curves boleh merangkumi r=0.1..64 tanpa mendakwa sampling semua titik.

## Measurement dan uncertainty

Sama seperti eksperimen asal: 10 seed, 4 chain, dt=.05, 100000 samples/chain,
stationary initialisation, horizon 5000. PRNG seed family baharu 20261111;
tarikh run kekal 2026-11-10 (integer PRNG bukan tarikh).
Centered b_force=0 cukup untuk covariance dan ACF; perubahan mean force tidak
mengubah centered dynamics. Tiada time-to-solution, burn-in, atau total RMSE
claim baharu; primary observable ialah stationary mean-mode autocorrelation.

Fixed ACF window 40 physical units bagi semua cases, dipilih untuk slowest
sampled regime. Student-t 95% CI merentas seed. Bandingkan estimator dengan
exact trapezoid window sama, dan laporkan continuous infinite tau berasingan.
Ukur covariance visible whitened vs target; simpan centered mean per seed.
Laporkan ESS ratio continuous per unit masa terhadap clique sebagai prediction,
bukan measured general speedup bagi semua observables.

## Resource dan checks

Laporkan trace(M), auxiliary noise amplitude sqrt(b), lambda_max(MH),
dan Euler stability ceiling 2/lambda_max (diagnostik numerik, bukan constraint
exact simulator atau spesifikasi chip). Exact OU boleh menyembunyikan kos
menyelesaikan fast modes dalam simulator/hardware sebenar.

Checks: SPD, exact transition stationarity, continuous Lyapunov equation,
tau formula vs eigen decomposition dan direct inverse, mean-distribution
chi-square reference daripada finite observation covariance. CI ialah marginal,
bukan simultaneous; jangan rerun seeds sehingga semua CI lulus.

Negative control analitik pada r=4: naikkan drift mobility tetapi kekalkan
noise sqrt(2)I. Solve Lyapunov dan ukur covariance yang salah. Tiada claim
bahawa kawalan negatif ialah implementasi yang betul.

Simpan source/protocol hashes, versions, command, per-seed records, plot dan
report. Tiada fitting r berdasarkan noise; optimum proxy ditentukan analitik.
Tidak mengukur joule, circuit bandwidth, calibration, atau hardware layout.
