# Paper: A Framework for Stochastic Differentiable Programming

Verdon, Tyrpak, Lockwood, Morton, Neagoe, Sugolov, MacCormack, Amico (Extropic). arXiv:2608.01612v2 [cs.ET], 13 Ogos 2026. Fail asal: `2608.01612v2.pdf` di root projek (21 halaman).

Nota ini ringkasan dalam Bahasa Melayu (istilah Inggeris dikekalkan), satu fail setiap halaman PDF di `page/`. Bukan salinan teks. Persamaan dirujuk dengan nombor dalam paper, contohnya Eq. (17). Teks diekstrak dengan `pypdf`; rajah tidak dibaca, hanya kapsyen.

## Peta halaman

| Halaman | Isi | Bahagian paper |
|---|---|---|
| [01](page/01.md) | Abstrak, pengenalan, motivasi PSC | Abstrak, I |
| [02](page/02.md) | Struktur paper, primitif pbit/pdit/pmode | I, II.A |
| [03](page/03.md) | Notasi Dirac, kernel stokastik, komposisi siri/selari | II.A, II.B |
| [04](page/04.md) | Rantai Markov, CTMC, Directed Factor Graph | II.C, II.D |
| [05](page/05.md) | Definisi PSC, lapisan, gate cabang `(1−σ(θ))I+σ(θ)B` | III, III.A |
| [06](page/06.md) | PNOT, PSWAP, PIsing, gate pmode, hibrid, MoG | III.A |
| [07](page/07.md) | Trotter, Strang, REINFORCE | III.B, III.C |
| [08](page/08.md) | Parameter-shift, kernel berasaskan tenaga, antara muka Torx | III.C, IV.A |
| [09](page/09.md) | Backend, X0/XTR-0, Z1 | IV.B |
| [10](page/10.md) | Z1 2-local Glauber, demonstrasi, jalan rawak graf | IV.B, V.A |
| [11](page/11.md) | Hasil jalan rawak, discrete diffusion | V.A, V.B |
| [12](page/12.md) | SGNN MaxCut, jump diffusion | V.C, V.D |
| [13](page/13.md) | Ising ring, PCD, bias PIsing bond-local | V.D, V.E |
| [14](page/14.md) | Eksperimen perkakasan X0, kesimpulan | V.F, VI |
| [15](page/15.md) | Rujukan [2]–[24] | |
| [16](page/16.md) | Rujukan [25]–akhir | |
| [17](page/17.md) | Appendix A gate tambahan, B panduan untuk pembaca kuantum | A, B |
| [18](page/18.md) | Appendix C: pmode Gaussian, semigroup, SVD | C |
| [19](page/19.md) | Appendix C: Fokker-Planck, gate bukan-Gaussian | C |
| [20](page/20.md) | Appendix D: X0 dan XTR-0 | D |
| [21](page/21.md) | Jadual II: primitif X0 | D |

## Perkara paling berguna untuk eksperimen kita

1. Eq. (17), (37)-(40): sigmoid-mixture gate dan **parameter-shift identity** `∂G/∂θ = G(θ) − G(θ⁻)`.
2. Eq. (41)-(44): estimator gradient untuk kernel berasaskan tenaga, **varians tinggi**.
3. Halaman 10 dan 13: Z1 menilai **2-local Glauber** secara asli; PISING bond-local **bukan** penjana Glauber penuh (sebab bias TV 0.40).
4. Halaman 9 dan 20: pbit X0 ada ralat kalibrasi ~5% dalam kebarangkalian, autokorelasi `τ0 ≈ 100 ns`, DAC 8-bit. Ini fakta X0, bukan Z1.
5. Halaman 14: pbit perkakasan sebagai sumber rawak untuk Metropolis-Hastings dan importance sampling, TV setanding dengan perisian.
