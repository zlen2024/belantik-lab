# Sejarah setiap anchor: Ising seed 29 dan 43

15 November 2026. Analisis deskriptif snapshot yang telah disimpan; **tiada
optimizer update baharu**. Lapan kes merangkumi original, bank swap, full F/R
anchor swap dan anchor-bias swap untuk kedua-dua background. Setiap kes
mengandungi initialization serta lexical updates 20–400 (21 snapshot).
Keputusan lengkap setiap 16 perkataan, empat state probabilities, entropy,
train occurrence count, within-family pairwise TV, family mean, context TV,
dan full-training lexical loss direkod dalam MLflow run
`246f97dfb76d4c91b30bce7940f87f31` (`config.json`, `cases`). Setiap case
merekod hash artifact input. Test split tidak dinilai.

## Apa yang dilihat

| Kes | Update | Finance: bilangan state paling mungkin | River: bilangan state paling mungkin | Weighted F/R TV |
|---|---:|---|---|---:|
| Original 29 | 20 | 8 `--` | 6 `--`, 2 `-+` | 0.1779 |
| Original 29 | 40 | 8 `--` | 8 `--` | 0.0947 |
| Original 29 | 100 | 8 `--` | 8 `--` | 0.0081 |
| Original 29 | 400 | 8 `--` | 8 `--` | 0.0019 |
| Original 43 | 40 | 2 `++`, 6 `-+` | 8 `-+` | 0.2922 |
| Original 43 | 100 | 6 `++`, 2 `-+` | 8 `-+` | 0.7167 |
| Original 43 | 400 | 8 `++` | 8 `-+` | 0.9935 |
| Bias swap 29←43 | 400 | 8 `++` | 8 `--` | 0.9951 |
| Bias swap 43←29 | 400 | 8 `-+` | 7 `-+`, 1 `++` | 0.0214 |

`--`, `+-`, `-+`, `++` ialah label state **dalam** satu model; label tidak
semestinya mempunyai makna yang sama merentas run. Jadual hanya membandingkan
pemisahan F/R dalam setiap model. State paling mungkin hanyalah argmax, bukan
semestinya probability tinggi. Pada update 40 original29, purata probability
state dominan ialah 0.887 untuk F dan 0.781 untuk R; satu token serendah 0.510.
Pada update100 semua 16 anchor original29 mempunyai probability dominan
sekurang-kurangnya 0.932; pada update400 sekurang-kurangnya 0.975.

Jadi pada seed29 **semua 16 anchor**, bukan hanya satu atau dua perkataan,
menumpu kuat ke state yang sama pada akhir lexical. Ini bukti deskriptif,
bukan bukti semua 16 token sama penting untuk menyebabkan outcome. Pada seed43,
kedua-dua keluarga akhirnya menggunakan state berlainan; pada swap bias yang
gagal, `water` menjadi outlier argmax `++` dengan probability 0.464, walaupun
15/16 anchor memilih `-+`.

Critic turut mendapati original43 tidak pulih secara monotonik: pada update40,
hanya `credit` dan `finance` masih memilih `++`, sementara 14 anchor lain memilih
`-+`. Context TV turun daripada 0.1086 pada update20 kepada 0.0269 pada update40,
sebelum meningkat kepada 0.9854 pada update200. Dua perkataan itu ialah petunjuk
untuk diuji, bukan bukti mereka menyebabkan pemulihan. Swap yang berjaya
mencapai F `++` / R `--`, berbeza daripada original43 F `++` / R `-+`;
penderma berkaitan dengan pemisahan, bukan trajectory atau geometri yang sama.

Weighted TV menggunakan bilangan occurrence training seperti sense map asal;
family counts di jadual memberi setiap token satu undi. Dalam reverse bias
swap, TV antara family means kecil (0.0214) tidak bermaksud setiap perkataan
identik: within-family mean pairwise TV F=0.1003 dan R=0.1513, berbanding
0.0049 dan 0.0071 pada original29. Oleh itu outcome gagal yang sama boleh
mengandungi geometri per-token yang berbeza.

Full-training lexical loss turun daripada 1.3104 ke 0.2685 pada original29,
dan 1.3128 ke 0.1396 pada original43. Loss turun tidak cukup untuk membuktikan
bahawa kedua-dua family telah dipisahkan. Analisis terbatas kepada dua seed,
data synthetic, dan validation yang telah digunakan untuk pemilihan; ia tidak
menentukan punca atau menunjukkan performance pada teks sebenar/hardware.

Critic 6.1 Sol Ultra mengesahkan perbandingan dan mencadangkan dua reciprocal
transfer purata bias family sambil mengekalkan residual setiap token. Protokol
terhad yang belum dijalankan berada dalam
[cadangan family mean](ISING_FAMILY_MEAN_BIAS_PROTOCOL.md).

## Verification

Enumeration NumPy per-token pada original29/step40 dibandingkan dengan unary
JAX asal untuk semua token: max absolute difference `5.56e-16`. Metadata MLflow
menyatakan sumber snapshot, hash data dan tier `classical`. Semua keputusan dan
kod tersedia untuk audit; database dan artifact MLflow masih memerlukan backup
lokal, kerana Git tidak menyimpannya.
