# THRML six-spin four-chain audit

**Tarikh:** 15 November 2026  
**Execution tier:** THRML simulator (`cpu_gpu_simulator`), bukan physical hardware

## Tujuan

Semak bahawa THRML block-Gibbs sampler meniru enumeration exact untuk fixture
Ising kecil yang sengaja mempunyai dua basin `bank`: `++` disukai dalam finance
dan `--` disukai dalam river. Ini ialah langkah parity simulator sebelum sebarang
claim tentang laluan ke thermodynamic hardware.

## Apa yang diuji

- 6 spin, 5 edge bukan sifar, max degree 3, dua color block yang sah.
- Bias context `+2` (finance) dan `-2` (river).
- Empat chain dengan permulaan sengaja berbeza: all-minus, all-plus, dan dua
  corak alternating.
- Setiap chain: warmup 500, 2,000 retained sweeps, satu Gibbs step per sample.
- Observable: P(`bank=++`) dan P(`bank=--`) dibandingkan dengan enumeration
  exact 64 state.
- Gate: `R-hat < 1.05`, `ESS >= 400`, `MCSE <= 0.01`, dan
  `absolute error <= max(0.02, 3 × MCSE)`.

## Hasil

| RNG seed | MLflow run | Max R-hat | Min ESS | Max MCSE | Worst error | Gate |
|---:|---|---:|---:|---:|---:|---|
| 17 | `2d743ba7ece54edc8640954b8c299131` | 1.000643 | 3604.4 | 0.00392 | 0.00592 | pass |
| 29 | `134fae32344f4cafbc8136f7263686f1` | 1.000826 | 4004.7 | 0.00362 | 0.00242 | pass |
| 43 | `b85dac0b672b4ced9758a371ef2ad955` | 1.001804 | 3730.3 | 0.00396 | 0.00445 | pass |

## Kesimpulan

THRML simulator melepasi parity dan mixing gate untuk fixture Ising 6-spin ini.
Ini mengesahkan implementasi simulator bagi tenaga, spin encoding, graph coloring
dan observable yang diuji. Ia **tidak** bermaksud Gaussian Word Energy telah
diturunkan ke THRML, model telah belajar natural text, atau program telah berjalan
pada chip Extropic. Langkah selepas ini mestilah sama ada (1) Ising learned model
yang lulus gate sendiri, atau (2) reka bentuk discretization Gaussian yang mempunyai
parity exact sebelum dipaksa ke Ising/THRML.
