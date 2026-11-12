# Index: wiki/

Peta semua fail dan folder. Kemas kini fail ini setiap kali menambah, mengubah nama atau memadam halaman. Sejarah perubahan dalam [log.md](log.md).

## Folder

| Folder | Isi | Index | Log |
|---|---|---|---|
| [paper/](paper/index.md) | Nota paper arXiv 2608.01612, ikut halaman PDF | [index](paper/index.md) | [log](paper/log.md) |
| [paper/page/](paper/page/index.md) | 21 nota halaman | [index](paper/page/index.md) | [log](paper/page/log.md) |

## Halaman

| Fail | Tajuk | Sumber | Tahap bacaan | Status |
|---|---|---|---|---|
| [README.md](README.md) | Pengenalan wiki | | | semasa |
| [00-peta-konsep.md](00-peta-konsep.md) | Peta konsep | blog Extropic, repo torx | sederhana | semasa |
| [01-primitif-dan-gate.md](01-primitif-dan-gate.md) | Primitif dan gate | basic_usage, nb 01, larian kita | dalam | disahkan dengan larian |
| [02-simulator-dan-readout.md](02-simulator-dan-readout.md) | Simulator dan readout | nb 04, kod sumber | dalam | sebahagian disahkan |
| [03-graf-trotter-ctmc.md](03-graf-trotter-ctmc.md) | Graf, Trotter, CTMC | nb 02, 03, 05, paper hlm 7 dan 10 | sederhana | belum dijalankan |
| [04-ising-gibbs.md](04-ising-gibbs.md) | Ising dan Gibbs | nb 06, 16, paper hlm 13 | sederhana | nombor dipetik |
| [05-melatih-litar.md](05-melatih-litar.md) | Melatih litar | nb 07, 08, 09, paper hlm 7-8, 11-12 | sederhana | corak latihan dijalankan semula dalam `cookbook_6/9/11/12/13`; nombor rasmi dalam `../gradients.md` |
| [06-pmode-gaussian.md](06-pmode-gaussian.md) | Pmode Gaussian | nb 10-13, paper hlm 18-19 | ringkas | gate Gaussian, `condition`, OU tersuai dan JumpDiffusion dijalankan dalam `cookbook_2/7/12`; notebook 11-13 sendiri tidak dijalankan |
| [07-boltzmann-softmax.md](07-boltzmann-softmax.md) | Boltzmann dan softmax | nb 14 | sederhana | hipotesis untuk attention |
| [08-factor-dfg.md](08-factor-dfg.md) | Factor dan DFG | nb 15, 16, kod sumber | sederhana | corak nb 15 dan DFG bersarang dijalankan dalam `cookbook_3/7/12` |
| [09-kaitan-attention.md](09-kaitan-attention.md) | Kaitan dengan attention | eksperimen kita | dalam | eksperimen kita sahaja |
| [10-glosari.md](10-glosari.md) | Glosari | semua | | semasa |

Rujukan teknikal di luar `wiki/` (satu aras di atas): `../gates.md`, `../gradients.md`, `../estimator-math.md`, `../explain-code.md`, `../factors-dfg-hybrid.md`, `../hardware-facts.md`, `../thrml-vs-torx.md`; skrip di `../examples/` (jalankan `check_examples.py`).

Maksud "tahap bacaan": **dalam** = dibaca penuh atau dijalankan; **sederhana** = pengenalan, tajuk dan kesimpulan; **ringkas** = pengenalan dan tajuk.

## Peraturan penyelenggaraan

0. Fail Inggeris `../*.md` ialah sumber rasmi untuk nombor dan formula; wiki ini meringkas dan memaut kepadanya.

1. Setiap folder dalam `wiki/` mesti ada `index.md` dan `log.md`.
2. Setiap ubah fail: tambah satu baris dalam `log.md` folder itu (tarikh, tindakan, fail, sebab, sumber).
3. Setiap fakta: nyatakan sama ada daripada sumber luar, larian sendiri atau andaian.
4. Kemas kini lajur "Tahap bacaan" dan "Status" bila membaca lebih dalam atau menjalankan kod.
