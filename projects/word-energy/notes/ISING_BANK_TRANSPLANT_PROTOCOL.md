# Protokol pilot pertukaran initialization bank

15 November 2026. Direka sebelum run; semakan critic GPT-6.1 Sol Ultra.
Hipotesis khusus: parameter awal `bank` mungkin mempengaruhi sama ada lexical
training mencapai state finance/river berasingan pada seed asal 29 dan 43.
Ini ujian exploratory pada data synthetic dan validation yang pernah dipakai.

Empat keadaan faktorial: background seed29 atau seed43, dan parameter awal
`bank` daripada seed29 atau seed43. Dua diagonal ialah replay original yang
telah disimpan. Dua keadaan baharu menukar hanya dua raw bias dan satu raw
internal coupling token bank; semua word lain, raw interaction weights, data,
negatives, schedules, Adam, learning rate, clip dan 400 lexical updates kekal.
Kedua-dua arm baharu menggunakan 400 updates; tiada joint training atau test.

Endpoint utama **fixed lexical400**; lapor TV anchor F/R, posterior context TV,
bank prior, lexical validation AUC/loss dan per-family accuracy/gain. Simpan
setiap 20 updates, source checkpoints/hashes dan perbandingan parameter awal;
guna tiga run IDs untuk input asal dan hasil baharu yang boleh ditrace.

Jika kedua-dua swap berubah mengikut bank donor, itu menyokong pengaruh block
bank dalam pasangan seed ini, bukan bukti satu-satunya punca atau generalisasi
kepada initialization lain. Jika tidak, background atau interaksi boleh penting;
dua seed tidak cukup untuk infer population. Tiada claim lulus gate pilot asal.
Berhenti jika input tidak sepadan, nilai nonfinite atau reference replay gagal.
Bajet <=800 optimizer updates untuk dua arm, empat CPU threads; tiada penalaan.

## Keputusan — fixed lexical400

| Background | Bank donor | Anchor TV | Context TV | F / R betul | Lexical AUC | Validation lexical loss | Run MLflow |
|---:|---:|---:|---:|---|---:|---:|---|
| 29 | 29 | 0.001861 | 0.000000837 | 6/6 / 0/6 | 1.0 | 0.26280 | `44af1bc171e340bf8b6d9d4c02c8c191` (replay original) |
| 29 | 43 | 0.002045 | 0.000001386 | 0/6 / 6/6 | 1.0 | 0.26300 | `1f1703f05c2345388878f7e6cf4a941c` |
| 43 | 43 | 0.993516 | 0.992376 | 6/6 / 6/6 | 1.0 | 0.13753 | `b51e07f7fbb4432492fe2a6556049750` (replay original) |
| 43 | 29 | 0.991938 | 0.991562 | 6/6 / 6/6 | 1.0 | 0.13692 | `e542fda97aa0442fa7b390f23c0b72b4` |

Kedua-dua swap bermula daripada parameter yang sama tepat dengan background
kecuali tiga raw parameters `bank`; donor disahkan tepat. Run tambahan mengambil
8.89 dan 8.82 saat masa latihan. Seed29 kekal collapsed, dan seed43 kekal
terpisah: untuk pasangan ini, menukar block bank sahaja tidak memindahkan
kejayaan/kegagalan. Swap 29 menukar arah label family yang sentiasa dipilih;
context TV masih hampir sifar, jadi itu bukan semantic recovery. Hasil ini
mengehadkan hipotesis bank-initialization sahaja, tetapi tidak mengenal pasti
parameter background atau interaksi sebenar yang menyebabkan collapse.
