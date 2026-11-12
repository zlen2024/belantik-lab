# 05. Melatih litar stokastik

Sumber: notebook 07, 08, 09 (pengenalan dan kesimpulan).

## 07: denoising per-piksel MNIST

- Proses hadapan: bit-flip bebas (Austin 2021), CTMC (Campbell 2022).
- UNet 472,545 parameter dilatih **luar talian**, berat disimpan dalam repo.
- Torx lakukan satu langkah: setiap piksel satu `PNOT` dengan logit daripada `p̂_i = P(x0_i = 1 | x_t)`. Satu `PNOT` = Bernoulli(p̂).
- Perbezaan lukisan tunggal Torx dan ensemble 64 sampel yang dibulatkan.
- Nombor FID ialah untuk checkpoint UNet, bukan untuk Torx.

Pelajaran: Torx dijadikan **lapisan sampling** untuk kebarangkalian yang dihitung secara digital. Ini hampir sama dengan cara kita guna p-bit.

## 08: rangkaian konvolusi stokastik

- Bars-and-stripes 4×4: 16 pbit, 30 corak (16 bar, 14 stripe), 24 latihan dan 6 validasi, seed 123.
- Kernel empat gate: `PJUMP`, `PReset`, `PNOT`, `PJUMP`, dua peringkat pooling `PCNOT`.
- Pengikatan berat: `param_map` ialah **tatasusunan indeks numpy dalam notebook**, bukan API Torx (`full_thetas = gate_logits[param_map][:, None]`, 60 gate baca 5 logit). Mekanisme perpustakaan: `Site.param_key`, `TiledFactor(weight_tied=True)`, `ChainFactor`, `reps`. Konvolusi hanya mendatar.
- Latihan menembusi sampler: `BranchingSimulator(diff_method="param_shift_inf", num_samples=256)` + Adam.
- Semakan gradient berbanding finite-difference.

## 09: MaxCut dengan REINFORCE

- Satu pbit per nod, satu `PISING` per tepi (12 tepi), satu sweep = 12 kemas kini berurutan, ulang 20 kali.
- Bacaan: nilai potongan (cut). Diskret, jadi gunakan **REINFORCE**: tolak min kelompok sebagai baseline, kalikan dengan `M/(M−1)`. **Catatan:** notebook mensampel indeks daripada ketumpatan tepat `2^n` (StateVectorSimulator) dan membezakan `log p` tepat; ia tidak boleh diskala dan bukan sampling gaya perkakasan. REINFORCE per-gate (paper Eq. 34) tidak dilaksanakan dalam Torx.
- `StateVectorSimulator` hitung vektor kebarangkalian 256 keadaan tepat, jadi diagnostik "expected cut" tepat.
- Seed tambahan dicetak supaya hasil bukan untuk satu graf sahaja.

## Pola umum

1. Struktur tetap (senarai gate), parameter dalam pytree.
2. Kerugian ditulis atas kebarangkalian tepat atau sampel.
3. Gradient: state-vector tepat, parameter-shift (sampel), atau REINFORCE untuk hasil diskret.
4. Optimizer `optax`.

## Bias dan varians penganggar gradient (ringkasan; nombor rasmi dalam `../gradients.md`)

- `param_shift_inf`: tidak berat sebelah untuk readout linear dan kerugian kuadratik; untuk kerugian dengan `L''' ≠ 0` bias `(1/2) L''' g_j Var(f)/S` (`cookbook_13`).
- `param_shift_single`: varians mengikut paper Eq. 40; untuk kerugian tak linear bias `[L'' + (1/2) L''' g_j] Var(f)/S`, jadi `2 Var(f)/S` pada setiap gate untuk kerugian kuadratik.
- `param_shift_filter`: penganggar nisbah; formula kaedah delta untuk bias dan varians, tanda bias bergantung gate.
- Titik tetap satu gate untuk `(E - t)^2`: `t`, `t - 1/S`, `t/(1 - 1/S)`; litar 3-gate pergi ke had merosot (dua logit pertama ke −∞).
- Remedi: hasil darab dua kelompok hanya untuk kuadratik; polinomial darjah k perlu k kelompok bebas; `exp` tiada remedi tepat.
- Gelung latihan, perangkap Jensen dan Eq. 44 dengan garis dasar dijalankan dalam `cookbook_11`, `cookbook_12`, `cookbook_13`.

