# Liputan kamus dan pendedahan data latihan

21 November 2026. Dua audit deterministik ini menggunakan 600 artikel WikiText training untuk memilih perkataan mengikut kekerapan, dengan syarat sekurang-kurangnya lima kemunculan. Audit liputan menggunakan input SWORDS development tanpa menggunakan judgment label. SWORDS development sudah pernah digunakan untuk penilaian model, jadi keputusan untuk memilih saiz kamus berdasarkan audit ini bersifat pembangunan dan memerlukan penilaian bebas kemudian.

| Saiz kamus | Liputan token WikiText validation | SWORDS target dengan konteks dan ≥2 calon dalam kamus | Calon target-spesifik dalam kamus |
|---|---:|---:|---:|
| 2,000 | 74.06% | 219/370 | 4,049/22,967 |
| 5,000 | 83.74% | 284/370 | 7,025/22,967 |
| 10,000 | 89.37% | 323/370 | 9,795/22,967 |
| 19,871 | 93.27% | 348/370 | 12,378/22,967 |

Penyebut 22,967 merangkumi calon multiword dan calon yang tidak boleh menjadi satu ID ASCII. Ia mengira ID calon bagi target tertentu, bukan jumlah perkataan unik. Dua calon dalam kamus tidak menjamin calon sesuai menurut manusia. Pada korpus ini hanya 19,871 jenis perkataan mencapai kiraan minimum lima, jadi permintaan top 20,000 menghasilkan 19,871. Perkataan baharu di luar kamus masih belum mempunyai representasi terlatih atau fallback. Model tidak dilatih dalam audit liputan.

## Sampel cadangan 10,000 perkataan

Sampler cadangan menghasilkan 200,000 window tiga perkataan, dengan 158,777 window occurrence unik selepas mengabaikan urutan dua konteks. Ini ialah sample prospektif yang eligibility-nya berbeza daripada sample 2,000 asal; ia bukan paired dataset yang mengubah saiz kamus sahaja. Bagi 5,000 perkataan tambahan pada rank 5,001–10,000, **977 tidak muncul sebagai perkataan tengah** dan 861 muncul sekali dalam sampel. Median ialah dua kemunculan sebagai tengah. Bagi 3,000 perkataan pada rank 2,001–5,000, 45 tidak muncul sebagai tengah dan median ialah tujuh.

Critic mengira 3,753 window occurrence unik yang sama antara kedua-dua sampel, menggunakan posisi pusat dan dua posisi konteks yang disusun. Sampel asal mempunyai 141,101 window occurrence unik. Sebanyak 92,456 daripada 200,000 window baharu mengandungi sekurang-kurangnya satu perkataan di luar kamus asal. Untuk rank 5,001–10,000, median hanya satu posisi pusat mentah yang berbeza walaupun median dua kali disampel; 40 perkataan tidak muncul pada mana-mana posisi. Angka ini menguatkan keperluan melaporkan perbandingan pada sokongan bersama dan populasi tambahan secara berasingan.

Jika sampel yang sama diulang tiga kali, bilangan presentasi untuk satu occurrence menjadi tiga, tetapi ia tidak mencipta occurrence unik baharu. Perkataan yang tidak muncul sebagai target atau konteks masih boleh menerima negative-class gradient dalam full-softmax kerana semua kelas dinilai. Statistik pendedahan tidak mengukur sama ada representasi yang dipelajari akan berguna.

Full softmax ke atas 600,000 window menilai kira-kira 1.2 bilion skor calon untuk V=2,000 dan 6 bilion untuk V=10,000. Ini kiraan operasi skor, bukan nisbah masa atau penggunaan tenaga yang diukur. Ising empat-spin dan vector10 mempunyai masing-masing `10V+4` dan `10V` parameter dalam konfigurasi ini. Model sebenar, objective, sampler dan perkakasan mesti dinilai berasingan.

Audit liputan MLflow `b255997743f34449bebac134a19496ea`, commit protokol `aa9ce79`. Audit pendedahan MLflow `32daa45ead0441a2bb1d1d8ac6365fad`, commit `c6749c5`. Prospective sample, posisi, hash dan vocabulary disimpan sebagai artifact lokal dalam run pendedahan. Tiada SWORDS test, penilaian model baharu atau hardware digunakan.
