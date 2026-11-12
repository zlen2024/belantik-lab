# Scaling dan calibration: protokol sebelum run

2026-11-10. Susulan hanya langkah scaling/calibration. Tiada eksperimen
degree-limited atau implementasi litar. Eksperimen asal tidak diubah.

## Matlamat dan normalisasi

n=[16,64,256], visible grounding g=1, kBT=1. Clique mempunyai weight 1/n
setiap edge, star mempunyai n edge weight 1 ke satu auxiliary tanpa grounding.
Maka A=2I-11^T/n, eigenvalues 1 dan 2; conditioning kekal 2 apabila n berubah.
Ini berbeza daripada fixed edge weight 1 dalam eksperimen n=4 asal dan
dinyatakan secara eksplisit; tiada penggabungan hasil dua family.
Clique reference solve menggunakan struktur O(n), bukan dense naive benchmark.
Pengiraan eigen dense di sini ialah alat audit; wall-clock bukan speedup claim.

## Calibration utama

Bound delta=[0,.005,.01,.03,.05,.10]. Setiap komponen yang diuji didarab
1+delta*u, u iid Uniform[-1,1]. Delta ialah BOUND, bukan standard deviation.
Scopes coupling_only, grounding_only, both; 20 calibration realisations/seed
0..19. Coupling errors bebas setiap edge. Visible grounding errors paired
antara architectures. Draws yang sama digunakan sepanjang delta untuk paired
sensitivity; deltas/scopes bukan replicate bebas. Auxiliary ground kekal 0.
20 seeds mewakili synthetic devices, BUKAN thermal sample chains.
Force set: uniform, contrast e0-e1, dan 32 random unit vectors sama untuk
architectures, scopes, deltas. Force seed tetap 310+n. Tiada tuning force.

Primary metric ialah worst-case relative A-energy-norm solution bias,
untuk SEMUA nonzero b, dihitung exact melalui eigenvalues R=A^-1/2 A_eff A^-1/2:
max |1/lambda(R)-1|. Ia juga sama dengan spectral covariance error selepas
whitening oleh A^1/2; ini satu identiti, bukan dua pengesahan bebas.
Secondary: median/random-force errors, uniform/contrast errors,
tau y=sum(x)/sqrt(n) dengan mobility visible1, auxiliary1 atau16.
Kriteria target worst-case bias <=5%. Laporkan pass count /20 dan Wilson 95% CI,
mean Student-t 95% CI serta empirical 5/95% quantiles; quantiles bukan CI.
Walaupun semua komponen dibatasi delta, tidak boleh generalise synthetic
device pass rate kepada hardware sebenar.

## Controls

Sebelum simulation: Schur complement exact tanpa noise, analytic perturbation
bound delta/(1-delta), mean/covariance checks dengan direct solve pada sampled
audit cases. Auxiliary eliminasi: diag(k+g)-kk^T/sum(k).
Coupling-only mesti mengekalkan uniform solution jika grounding g=1.
Ideal tau clique=1, star=1+1/(n*r). Uji ideal g/w convention juga pada n=4.

Common-mode controls delta .05,.10, tanda +/-: coupling-only, ground-only,
both. Semua komponen dalam scope berkongsi offset; tiada iid averaging.
Auxiliary parasitic grounding eta=[0,.001,.01,.1,1] unit g: auxiliary ground
asal 0, jadi perlu ralat ADDITIVE. Relative uniform mean error dan worst-case
bias exact eta/(n+2*eta) bagi k=1. Tidak tambah parasitic ke baseline.

Pembetulan semasa verification sebelum run lengkap: formula awal parasitic
eta/(1+eta+eta/n) salah. Assertion terhadap Schur complement pada percubaan
pertama menangkap kesilapan. Effective uniform precision ialah
1+eta/(n+eta), jadi relative mean bias eta/(n+2*eta). Tiada seed, threshold,
atau calibration distribution diubah; percubaan pertama tidak menghasilkan
results.json. Pembetulan dicatat sebelum run semula, bukan untuk meluluskan
hasil empirical yang tidak diingini.

## Asingkan sampling error

Tidak jalankan trajectory bersaiz besar. Gunakan exact continuous-time OU
covariance bagi time average stationary untuk reference ideal, semua n,
clique, star mobility1 dan16. T=[10,100,1000,5000,10000,100000], 4 chain
independent averaged, unit norm force uniform dan contrast. Mean noise
covariance = S transformed eigen factor
2*(lambda*T+expm1(-lambda*T))/(4*T^2*lambda^3).
Lapor expected relative A-norm RMSE, bukan measured RMSE atau hardware latency.
Kalibrasi ialah permanent bias; thermal averaging variance berasingan.
Anchor noisy devices pada delta=.1, both, seed0: hitung expected total RMSE
dan exact decomposition bias^2+variance untuk kedua architecture (star r1,r16).
Masa bermula stationary; tidak termasuk burn-in, setup atau readout.

## Evidence dan metadata

Primary calibration metrics dikira secara numerik exact bagi setiap synthetic
device (tiada sampling uncertainty dalam solve). Variability ialah antara
synthetic devices. Source/protocol hash, versions, config, per-device results,
summary JSON, controls, analytical sampling curves, plot dan report disimpan.
Tidak fitting toleransi atau seed berdasarkan hasil. Formula ialah derivasi
untuk setup ini. Physical noise/calibration model sebenar belum disahkan.
