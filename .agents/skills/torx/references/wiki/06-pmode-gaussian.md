# 06. Pmode Gaussian

Sumber: notebook 10, 11, 12, 13 (pengenalan, tajuk dan ringkasan).

## 10: gate Gaussian

`AffineGaussianGate`: `x' = Ax + b + ε`, `ε ~ N(0, diag(exp(log_var)))`. Dalam kod, `theta` ada kunci `A`, `b`, `log_var`. Gate khusus `Displace`, `Scale`, `Mix`, `Diffuse` tetapkan sebahagian parameter.

Gaussian kekal Gaussian di bawah peta afin, jadi cukup hantar **min dan kovarians**:

`μ → Mμ + d`, `Σ → MΣM^T + Δ`.

`AffineGaussianSimulator` hantar momen tepat dan boleh syarat pada satu tapak dengan bentuk tertutup. `MixtureGaussianGate` dengan kawalan diskret keluar daripada keluarga Gaussian.

## 11: model ruang keadaan linear-Gaussian

- Pemacu tersembunyi 3 dimensi, enam saluran pemerhatian bising.
- Dibuka (unrolled) menjadi gate afin dengan parameter homogen masa. Perhatian: pengongsian theta di sini ialah penggunaan semula semasa inferens, bukan pengikatan boleh latih. `log_var = −inf` (saluran deterministik): notebook menasihatkan ganti dengan nombor negatif besar. Ujian kita (`cookbook_7_api.py`): melalui `AffineGaussianSimulator` gradient terhingga (d/dA = 4.0, d/dlog_var = 0.0), tetapi melalui `HybridPCircuit.sample` `d/dlog_var` ialah **NaN** (A dan b terhingga); ganti dengan kira-kira −30 jika membezakan `log_var` secara sampling.
- Syarat atas semua pemerhatian memberi posterior tepat, setara dengan Kalman filter + Rauch-Tung-Striebel smoothing.
- Skor peristiwa luar talian untuk semak.

## 12: Langevin pada Ising lembut

- Spin lembut berniali nyata, tenaga `V`.
- `LangevinGate` tersuai (`examples/helpers/_langevin.py`): `x ← x − ε∇V(x) + √(2Tε) ξ`.
- ULA (tanpa pembetulan) dan MALA (Metropolis-adjusted), diulang dengan `jax.lax.scan` atas `HybridPCircuit.sample`.
- Divalidasi pada kes 1-D dengan quadrature tepat; ULA menunjukkan bias saiz langkah.

## 13: peralihan rejim

- Rejim diskret (`PditCycle`, 3 keadaan) memilih hukum Gaussian untuk tokokan kedudukan (`MixtureGaussianGate`).
- Hanya rejim yang melompat; `X` tidak. `JumpDiffusionGate` ialah model lain.
- Satu langkah Euler dalam `HybridPCircuit`, discan menjadi lintasan.
- Rujukan analitik: hukum bersyarat rejim, hunian pegun, min akhir.

## Untuk kita

Tiada kaitan terus dengan p-bit binari Z1. Mengikut paper, **X0/XTR-0 membawa primitif pmode Gaussian dan pMoG secara fizikal** (Jadual II); Z1 ialah sampler Ising binari yang menilai penjana Glauber 2-setempat, dan program PSC dikompil kepadanya melalui Thermalizers (arXiv:2608.01615).
