# 01. Primitif dan gate

Sumber: `basic_usage`, notebook 01, dan `torx_learn.py` (kita jalankan).

## Gate cabang (branch gate)

`G(θ) = (1 − σ(θ)) I + σ(θ) B`, dengan `p = σ(θ) = 1/(1+e^{-θ})`.

Torx simpan **logit** θ, bukan p, kerana θ meliputi seluruh R dan sesuai dengan gradient. `θ → ±∞` memilih cabang secara deterministik.

| Gate | Operasi cabang B |
|---|---|
| `PNOT(i)` | balik bit i |
| `PCNOT([c,t])` | balik t jika c = 1 |
| `PSWAP` | tukar dua bit |
| `PJUMP` | `10 -> 01` satu arah (tak boleh balik); lain tetap |
| `POR` | bit pertama = OR, bit kedua = 0 |
| `PReset` | tetap semula bit |

Tiada gate AND atau darab dalam senarai ini.

## PNOT sebagai p-bit (disahkan oleh kita)

Mula pada 0, balik dengan `p = σ(θ)`. Maka `P(1) = σ(θ)` dan min spin `m = 2σ(θ) − 1 = tanh(θ/2)`, jadi `θ = 2·atanh(m)`. Ujian: m = −0.8, 0, +0.5 memberi tepat −0.8000, 0.0000, +0.5000.

## PISING (berasaskan tenaga)

Bukan gate logik. `theta = [J, h1, h2, β, dt]`. Tenaga `E = −J s1 s2 − h1 s1 − h2 s2`, dengan `s = 2σ − 1`. Matriks = `exp(Q·dt)`, `Q` ialah kadar Glauber lompat satu spin. `dt` besar membawa ke keseimbangan Boltzmann. Hanya serasi dengan `StateVectorSimulator`, bukan `BranchingSimulator` (jadual cabang).

Ujian kita (`mₐ=0.6`, `m_b=−0.4`, keseimbangan):

| J | ⟨ab⟩ | sasaran | ⟨a⟩ | ⟨b⟩ |
|---|---|---|---|---|
| 0.0 | −0.2400 | −0.2400 | 0.600 | −0.400 |
| 0.3 | +0.0552 | −0.2400 | 0.520 | −0.242 |
| 1.0 | +0.6383 | −0.2400 | 0.361 | +0.070 |

Kesimpulan: J > 0 menarik spin sama arah (J < 0 antiferomagnet, lawan arah); dalam kedua-dua kes marginal berubah, jadi bukan pendarab.

## Gate pdit dan pmode

- Pdit: `PditCycle(site, dims=d)` ialah jalan rawak stay/forward/backward; gate dengan lebih daripada 2 cabang guna **softmax([0, θ₁, ..., θ_{K−1}])**, θ berbentuk `(K−1,)` (disahkan: θ=[1.0, −0.5] beri [0.2312, 0.6285, 0.1402]). Rumus `(1−σ(θ))I + σ(θ)B` hanya untuk gate dua cabang. Lihat `../gates.md` untuk jadual lengkap.
- Pmode: `x' = Mx + d + ε`, `ε ~ N(0, diag(exp(log_var)))`. Lihat halaman 06.

## Cara baca matriks

Lajur x = taburan output diberi input |x). Contoh `PCNOT` dengan θ = +20: 10 → 11 dan 11 → 10 (disahkan).

## Latihan dasar (basic_usage)

Bina litar → `circuit.init_params(key)` → bina `StateVectorSimulator` → kehilangan `squared error` → Adam (`optax`) 1000 langkah. `build_circuit` untuk `BranchingSimulator` boleh dipanggil dalam atau luar `jit`; mengkompil di luar hanya untuk kelajuan bila theta tetap.
