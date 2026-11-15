# Protokol validation semula dengan coupling F3 diperbetulkan

15 November 2026. Direka selepas penemuan perbezaan faktor dua; ini eksperimen
**baharu** pada validation synthetic yang sudah digunakan, bukan pengulangan
prospektif preregistered gate lama dan bukan holdout bebas.

Versi legacy menggunakan `lambda*W` bagi setiap spoke F3; plan-conformant
menggunakan `lambda*W/2`. F2 sama tepat dalam kedua-duanya. Simpan default
legacy untuk reproduction sejarah; pilihan `normalize_spokes=True` ialah
versi corrected. Known-answer all-plus E2=-2, E3=-2 corrected vs -4 legacy,
dan ujian bebas 64-state, repeated word, lambda0/1 serta finite differences
perlu lulus sebelum training.

Untuk seed17/29/43, ambil checkpoint `lexical_selected_400` daripada replay
original yang telah disahkan (F2 tidak berubah). Sebelum training, ukur
validation checkpoint sama dengan legacy dan corrected sebagai sensitivity;
bezakan ia daripada keputusan yang dilatih semula. Kemudian mula Adam joint
baharu, 100 update corrected, lr .003, clip5, schedule/negatives/corruption
dan auxiliary .25 lexical asal. Simpan snapshot setiap 20 update, selected
checkpoint dengan selector validation asal, dan fixed joint100. Jumlah 300
update, <=4 thread environment; tiada parameter sweep, initialization baru
atau test evaluation.

Lapor per-seed validation macro sense, family counts/gain, anchor TV,
map-independent context TV, lexical AUC, ranking loss dan source hashes.
Simpan data/config/snapshots pada MLflow dengan nama corrected yang jelas.
Hasil ini menilai effect correction pada baseline kecil, bukan menunjukkan
strategy initialization baru lebih baik. Test lama pernah dibuka; sebelum
claim confirmatory sediakan holdout baru yang benar-benar bebas.

## Hasil validation selepas pembaikan

| Seed | MLflow run | Selected step | Sense selected / fixed100 | Context gain selected | Lexical AUC | Peak working set |
|---:|---|---:|---|---:|---:|---:|
| 17 | `25f97015403c40498623fa2c8c3b8290` | 100 | 1.0 / 1.0 | 0.444752 | 1.0 | 482.5 MiB |
| 29 | `232f088b9f2342c0ba2f03357d259bac` | 100 | 0.5 / 0.5 | −0.000000628 | 1.0 | 486.3 MiB |
| 43 | `91c8470602464d108dc5097820a5c85b` | 100 | 1.0 / 1.0 | 0.446384 | 1.0 | 484.7 MiB |

Seed29 tetap gagal validation sense dan gain; seed17/43 mencapai ambang toy.
Ini konsisten dengan lexical collapse sebelum joint training tetapi tidak
membuktikan faktor dua tidak mempunyai kesan lain. Keputusan ialah eksperimen
development pada validation dan seed yang sudah diperiksa, bukan gate asal.

Sensitivity pada checkpoint lexical yang sama tanpa retrain: legacy vs corrected
context TV untuk seed17 ialah 0.991874 vs 0.877718, seed29 0.00000084 vs
0.000003, dan seed43 0.992376 vs 0.881739. Jadi normalisasi memang mengubah
probability/readout walaupun macro sense sama. Punca collapse tidak dibuktikan.

Semua source hashes, snapshot dan family metrics disimpan pada artifact MLflow.
Peak ialah process working set Windows, bukan jumlah memory seluruh mesin atau
child process; semakan dibuat semasa snapshot dan selepas run. Environment
OMP/OPENBLAS diset ke empat, tetapi peak bilangan thread sebenar tidak diukur.
Masa training direkod sekitar 4.6–4.9s per seed; masa analisis/logging tambahan.

## Correction provenance checkpoint

Dalam tiga artifact run asal, medan diagnosis `checkpoint_sha256` sebenarnya
hash **fail replay lexical**, bukan hash parameter selected/fixed joint. Artifact
asal tidak diubah. Rekod tambahan yang dipautkan kepada run asal menyatakan
hash fail replay dan hash snapshot lexical/selected/fixed secara berasingan:
seed17 `af9346e54b8a4c8ca0a6bf3b196854f9`; seed29
`af3e61da516747e99138ce9a78b82f79`; seed43
`3422d7d51d0448b1809dd06514e5ba32`. Hash snapshot ialah SHA-256 JSON canonical
(`sort_keys=True`, separators comma/colon). Runner telah dibaiki untuk run
seterusnya; runner hash dalam tiga run asal merujuk versi sebelum pembaikan.
Hash versi runner lama mengenal pasti kandungannya, tetapi source snapshot runner
lama tidak disimpan dalam artifact; jangan menganggap hash sahaja membolehkan
fail itu dipulihkan. Training source hash direkod dan perlu dipelihara bersama
Git history dan artifact MLflow semasa backup.

Critic 6.1 Sol Ultra mengira semula diagnostics daripada snapshot dengan
enumeration bebas, termasuk family gains, sense, anchor/context TV dan ranking
loss; hasil dilaporkan sepadan dalam toleransi numerik. Perbezaan hash di atas
ialah kelemahan metadata, bukan perubahan parameter atau keputusan.
