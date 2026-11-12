# 04. Ising, Gibbs dan PISING

Sumber: notebook 06 dan 16, serta apa yang kita uji dalam `onchip_gate.py`.

## Model

Ising = graf dengan gandingan pada tepi dan medan pada tapak. `π(s) ∝ exp(−β H(s))`. Sama dengan Boltzmann machine (Ackley 1985).

## Chromatic Gibbs

Taburan bersyarat satu spin bergantung pada jiran sahaja, jadi tapak tak berjiran (satu warna) dikemas kini serentak. Inilah yang **Z1 buat dalam silikon** (blog dan whitepaper: chromatic Gibbs sweep, degree 16).

## Kemas kini satu tapak (06)

Ideal: `PReset(∞)` kemudian `PNOT`. Boleh laku: `PReset` dengan kekuatan terhingga (12 dalam notebook), atau bekalkan keadaan sifar secara manual dan jalankan `PNOT` sahaja.

## Keputusan notebook 06 (dipetik, tidak dijalankan semula)

| Kaedah | TV ke taburan tepat (cincin 8 tapak) |
|---|---|
| Chromatic Gibbs melalui Torx | 0.0634 (had ditegaskan 0.08); lantai sampel terhingga untuk sasaran dan N notebook tidak dikira semula; rujuk `cookbook_5_gibbs.py`: TV 0.0457 lawan lantai 0.0468 pada cincin kita |
| Host berjalan atas matriks `PISING` per-bond | 0.4020 |

Sebab bias: matriks bond tidak nampak spin jiran luar pasangan, jadi komposisi lapan kemas kini pasangan tidak patuh detailed balance untuk cincin penuh.

Pelajaran: untuk sampling Boltzmann, guna kemas kini **satu tapak berasaskan medan setempat**, bukan kernel pasangan terpencil.

## Pembelajaran parameter

Persistent contrastive divergence (PCD) dalam NumPy hos pulihkan gandingan dan medan dalam 0.05 selepas 300 langkah (ralat terburuk 0.0373 dan 0.0386). Hos sahaja, bukan Torx.

## Notebook 16: Gibbs sebagai DFG

Tapak 4×4 torus, 2^16 = 65,536 keadaan, jadi semua dibandingkan dengan enumerasi tepat. Factor satu spin dengan **medan sebagai input** (bukan berat tertanam), tiling papan dam, graf diulang. Notebook sendiri kata Gibbs bukan kegunaan utama Torx; `thrml` ialah pustaka khusus.

## Kaitan dengan percubaan AND kita

Gate AND = penalti Ising dengan `λ`. Satu kemas kini spin pada `λ` tinggi dihalang oleh penalti, jadi rantai beku (kita belum ukur autokorelasi). Ini selaras dengan pelajaran di atas: kemas kini bersyarat setempat betul, tetapi bergerak antara keadaan sah terhad.
