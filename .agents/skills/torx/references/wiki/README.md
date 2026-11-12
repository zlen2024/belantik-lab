# Wiki Torx dan eksperimen attention Z1

Nota belajar kita, dalam Bahasa Melayu dengan istilah teknikal Inggeris. Sumber utama: repo `torx/` (16 notebook + `basic_usage`), whitepaper arXiv 2608.01612, dan blog Extropic "From One to One Billion". Eksperimen sendiri ada dalam folder induk (`attn_thrml/`).

## Halaman

| # | Halaman | Isi | Notebook |
|---|---|---|---|
| 00 | [Peta konsep](00-peta-konsep.md) | Stack, tiga primitif, gate = kernel | semua |
| 01 | [Primitif dan gate](01-primitif-dan-gate.md) | pbit/pdit/pmode, logit θ, PNOT/PCNOT/POR/PISING | basic_usage, 01 |
| 02 | [Simulator dan readout](02-simulator-dan-readout.md) | exact vs sampled, 1/√N, parameter-shift | 04 |
| 03 | [Graf, Trotter, CTMC](03-graf-trotter-ctmc.md) | PSWAP/PJUMP, splitting bias, reaksi kimia | 02, 03, 05 |
| 04 | [Ising, Gibbs, PISING](04-ising-gibbs.md) | chromatic Gibbs, bias bond-local, PCD | 06, 16 |
| 05 | [Melatih litar stokastik](05-melatih-litar.md) | pengikatan berat, REINFORCE, denoising | 07, 08, 09 |
| 06 | [Pmode Gaussian](06-pmode-gaussian.md) | affine gate, SSM, Langevin, regime switching | 10, 11, 12, 13 |
| 07 | [Boltzmann dan softmax](07-boltzmann-softmax.md) | softmax muncul daripada pdit | 14 |
| 08 | [Factor dan DFG](08-factor-dfg.md) | sample(key, inputs, params) di semua peringkat | 15, 16 |
| 09 | [Kaitan dengan attention kita](09-kaitan-attention.md) | Apa yang kita uji dan apa yang tinggal | eksperimen kita |
| 10 | [Glosari](10-glosari.md) | Istilah ringkas | |

## Tahap bacaan (penting, jujur)

- Aku baca **prose** notebook (markdown cell): pengenalan, tajuk bahagian dan kesimpulan. Notebook `basic_usage`, 01, 02, 03, 04 dibaca lebih lama; 05 hingga 16 dibaca pengenalan, tajuk dan kesimpulan.
- Kod dalam notebook **tidak dijalankan** secara langsung. Skrip kita yang dijalankan: `torx_learn.py`, `cookbook_1..` (cookbook 3 dan 5 membina semula corak notebook 15 dan 06).
- Nombor yang dipetik (contoh TV 0.0634) ialah yang ditulis dalam notebook, bukan hasil larian kita.
- Tiada bukti Z1 melaksanakan gate Torx. Whitepaper kata ia akan dikompil melalui Thermalizers.

## Fail eksperimen kita

- `z1_attention.py`: dot product dengan p-bit bebas, `Z1Config`.
- `onchip_gate.py`: gate AND 2-local, keputusan buruk.
- `torx_learn.py`: PNOT, gate, PISING, dot product tersampel dalam Torx.
