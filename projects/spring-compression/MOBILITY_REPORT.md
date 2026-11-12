# Keputusan mobility auxiliary

2026-11-10. Eksperimen susulan; keputusan asal tidak diubah.

Model: dq=-MH(q-mu)dt+sqrt(2M)dW; kBT=1. M mengubah drift dan noise bersama. Semua stationary covariance ialah H^-1.

10 seed x 4 chain x 100000 samples per case, dt=.05. Exact OU transitions. ACF window 40. CI Student-t marginal 95% merentas seed, bukan simultaneous.

## Hasil

| Regime | r | a visible | b auxiliary | tau exact | tau measured [95% CI] | ESS rate / clique (theory) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | None | 1.0000 | 1.0000 | 1.00000 | 0.9914 [0.9136, 1.0693] | 1.0000 |
| fixed_visible | 0.25 | 1.0000 | 0.2500 | 2.00000 | 2.0585 [1.9571, 2.1599] | 0.5000 |
| fixed_visible | 1 | 1.0000 | 1.0000 | 1.25000 | 1.2642 [1.2201, 1.3083] | 0.8000 |
| fixed_visible | 2 | 1.0000 | 2.0000 | 1.12500 | 1.1584 [1.0898, 1.2269] | 0.8889 |
| fixed_visible | 4 | 1.0000 | 4.0000 | 1.06250 | 1.1032 [1.0170, 1.1894] | 0.9412 |
| fixed_visible | 16 | 1.0000 | 16.0000 | 1.01562 | 1.0115 [0.9535, 1.0696] | 0.9846 |
| fixed_trace | 0.25 | 0.9412 | 0.2353 | 2.12500 | 2.0959 [1.9255, 2.2663] | 0.4706 |
| fixed_trace | 1 | 0.8000 | 0.8000 | 1.56250 | 1.5311 [1.4534, 1.6089] | 0.6400 |
| fixed_trace | 4 | 0.5000 | 2.0000 | 2.12500 | 2.0334 [1.9238, 2.1430] | 0.4706 |
| fixed_trace | 16 | 0.2000 | 3.2000 | 5.07812 | 5.0424 [4.7672, 5.3176] | 0.1969 |

## Derivasi dan tafsiran

Untuk c=(1/2,1/2,1/2,1/2,0), tau=c^T H^-1 M^-1 H^-1 c/(c^T H^-1 c)=1/a+1/(4b).
Dengan a=1, b=r: tau=1+1/(4r). Pada r=16, tau=1.015625: hanya 1.5625% melebihi clique, dan predicted ESS rate meningkat 23.08% berbanding star r=1. Ini khusus mean-mode stationary dalam physical-time unit yang sama.
Tiada finite r dengan a=1 mengatasi clique dalam metrik ini. CI mungkin tidak dapat membezakan r besar daripada clique; perbezaan kecil ditentukan formula, bukan dakwaan significance Monte Carlo.
Dengan 4a+b=4: tau=17/16+r/4+1/(4r), minimum 25/16 pada r=1. Maka mempercepat auxiliary sambil mengurangkan mobility visible tidak menghasilkan kemenangan dalam proxy ini.
trace(M) bukan watt, chip area, atau harga komponen. Bajet ini hanya ujian sensitivity terhadap cara sumber diperuntukkan.

## Kos yang masih belum diukur

- Untuk r=16 dan a=1, trace(M)=20 berbanding clique 4; amplitude noise auxiliary 4 kali baseline. Ini tuntutan parameter dinamik, bukan tenaga elektrik 5 kali.
- Fast relaxation rate meningkat. Ia boleh menuntut resolusi temporal lebih tinggi dalam model numerik biasa atau sistem fizikal; tiada spesifikasi circuit diukur.
- Exact OU integration menghapuskan Euler bias tetapi tidak membuktikan implementasi hardware dapat mencapai kadar tersebut.
- Friction/mobility, damping, noise bandwidth, auxiliary capacitance/mass, calibration dan energy per answer perlu model litar atau hardware sebelum menilai kos sebenar.
- Tiada pengukuran burn-in atau solution RMSE baharu; eksperimen ini hanya stationary ACF dan covariance.

## Verifikasi

SPD, fluctuation-dissipation Lyapunov identity, exact-step stationarity, tau via independent inverse dan spectral formula, serta finite-sample covariance via explicit lag sum semuanya lulus.
Mean Gaussian chi-square diagnostics serta covariance errors direkodkan dalam JSON; non-rejection bukan bukti semua aspek sampler betul.
Negative control r=4, noise tidak diselaraskan: covariance visible whitening error 4.3478%. Ini menunjukkan drift-only rescaling menukar equilibrium.

## Artifak

- MOBILITY_PROTOCOL.md: setup sebelum run.
- run_mobility.py: kod.
- mobility_results/results.json: metadata, hashes, full per-seed records, exact curves, CI.
- mobility_results/mobility.png: figure.
- Wall-clock sebelum plot: 10.03 saat; bukan physical horizon.

Formula dan angka di atas ialah derivasi serta pengukuran tempatan untuk model yang ditetapkan, bukan benchmark chip.
