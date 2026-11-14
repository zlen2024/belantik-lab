# Ising collapse: protokol attribution validation sahaja

15 November 2026. Disemak oleh critic GPT-6.1 Sol Ultra sebelum eksperimen.
Tujuan: mengenal pasti tahap latihan yang berkaitan dengan collapse, bukan
mengubah model sehingga seed 29 lulus. Data validation sudah digunakan dalam
pemilihan checkpoint; hasil eksperimen ini bersifat exploratory.

## Perbandingan awal checkpoint sedia ada

| Seed | Random: perbezaan anchor F/R | Original selected | Permuted lexical final | Permuted sense |
|---:|---:|---:|---:|---:|
| 17 | 0.13161 | 0.99766 | 0.97590 | 1.0 |
| 29 | 0.05009 | 0.000807 | 0.19875 | 0.5 |
| 43 | 0.07688 | 0.99710 | 0.27975 | 0.5 |

Nombor ialah total variation, 0 sama dan 1 terpisah. Seed29 bermula dengan
anchor hampir uniform; collapse tajam pada state tunggal belum wujud pada
initialization. Namun initialization masih boleh mempengaruhi arah optimization.
Permuted lexical control masih mempunyai joint supervision dan negatives asal;
ia bukan no-signal control. Accuracy 0.5 juga boleh berlaku tanpa anchor collapse
seteruk original seed29.

Run diagnosis validation MLflow: random seeds 17/29/43 ialah
`c8e9e28f823d40c18b8f39ba4c96386f`, `4d82139099a442cf8ef808698ca43e1c`,
`d871298008a145a58623ddc1e6d4468d`; permuted seeds 17/29/43 ialah
`4152a60334784cb7906580abf51bd801`, `bf707f699a7d45b48f6cea07dd887f97`,
`6eed68cf5d5642128816adc0265881d1`. Config artifact setiap run menyimpan
checkpoint hash dan row-level posterior.

## Eksperimen yang dicadangkan — belum dijalankan

Seeds 17,29,43; data, negatives, schedules, optimizer dan budgets asal. Tiada
test evaluation, parameter sweep atau threshold tuning. Simpan lima keadaan:

| Label | Mula | Latihan |
|---|---|---|
| I | checkpoint random asal | tiada |
| L | I | lexical asal 400 update, simpan selected lexical |
| LJ | L | joint asal 100 update, simpan selected dan fixed step100 |
| IJ | I | joint asal 100 update tanpa lexical pretraining |
| LA | L | auxiliary-only 100 update: `0.25 * lexical_loss` |

Replay LJ mesti menghasilkan semula checkpoint asal: maksimum perbezaan mutlak
parameter <= 1e-10 dan metric validation sepadan. Jika tidak, hentikan branches
dan diagnosis reproduction dahulu. Lexical selected dan joint selected ikut
selector asal, termasuk tie-break joint menggunakan **ranking loss sahaja**;
latihan joint pula menggunakan ranking + 0.25 lexical. Log komponen berasingan.
Bandingan utama gunakan endpoint tetap, bukan memilih checkpoint selepas hasil.

IJ dan LA bermula daripada salinan parameter yang sama dengan control masing-masing;
Adam baharu lr 0.003, clip 5, 100 updates dan auxiliary batch IDs asal. Kekalkan
faktor 0.25 dalam LA. Merekod snapshot setiap 20 update dan L selected. Log qF/qR,
entropy, bank prior, family posterior/gain/counts, dan TV antara purata center
posterior F/R tanpa sense map. Lapor lexical AUC/loss, ranking loss, total loss,
gradients dan saturation. Setiap run simpan hashes, seeds, status dan batasnya.

Had: <=2100 optimizer updates untuk tiga seed, <=30 minit, <=4 CPU threads,
sasaran <=2 GiB; hentikan arm nonfinite. Jika L collapse dan I tidak, lexical
stage terlibat; jika hanya LJ collapse, joint stage terlibat. LA membantu
membezakan pengaruh tambahan lexical daripada joint-ranking, tetapi hasil tetap
tidak membuktikan satu parameter awal tertentu ialah punca. Tidak menganggap
sebarang branch berjaya sebagai lulus gate pilot asal.
