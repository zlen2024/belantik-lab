# 00. Peta konsep

## Stack Extropic (blog, Fig. 01)

```
Torx          program stokastik boleh dibezakan (PSC / DFG)      JAX, sumber terbuka
  v
Thermalizers  kompil kernel -> model berasaskan tenaga (EBM)     kertas ada, library belum keluar
  v
THRML         model hipergraf, Gibbs sampling "dekat dengan metal"
  v
Backend       XTR-0 (cip X0, early adopter) | Z1 (tapeout, early access 2027) | simulator GPU
```

Analogi dalam blog: Torx kepada THRML seperti PyTorch kepada CUDA.

## Tiga primitif data

| Primitif | Nilai | Contoh gate |
|---|---|---|
| pbit | {0,1} | PNOT, PCNOT, PSWAP, PJUMP, POR, PReset, PISING |
| pdit | {0..d-1} | PditCycle, PditShift, PditSWAP |
| pmode | R^N | AffineGaussianGate, Displace, Scale, Mix, Diffuse, MixtureGaussianGate |

## Dua lapisan objek

- **PSC** (Parametrized Stochastic Circuit): senarai gate dalam urutan, struktur meniru hardware.
- **DFG** (Directed Factor Graph): graf berarah tanpa kitar bagi factor `P(output | inputs)`. Gate juga ialah factor.

## Satu idea utama

Setiap gate ialah **kernel** `K(y|x)` (lajur jumlah 1). Litar = hasil darab kernel. Parameter (logit θ) kekal dalam pytree berasingan supaya boleh dibezakan dan dilatih.

## Dua cara menjalankan

- Tepat: `StateVectorSimulator.density` (enumerasi, kecil sahaja).
- Sampel: `BranchingSimulator.sample` / `expval_all` (ralat `1/√N`).
- Gaussian: `AffineGaussianSimulator` (momen min dan kovarians, tepat).
