# Protokol pertukaran initialization anchor F/R

15 November 2026. Cadangan critic GPT-6.1 Sol Ultra sebelum run. Eksperimen
menukar raw bias dan internal coupling untuk 16 token anchor finance/river
antara initialization seed29 dan seed43. Bank, song, music, shared interaction,
data, negative, schedule, optimizer dan bajet 400 lexical updates kekal.

Dua reciprocal swaps, dan replay diagonal sedia ada sebagai control. Ukur fixed
lexical400 anchor/context TV, prior, per-family sense/gain dan lexical AUC/loss;
simpan setiap 20 updates dan hashes. Endpoint validation bersifat exploratory;
tiada joint training, test evaluation atau seed sweep. 800 updates tambahan;
tetapan empat CPU threads, had memory belum diukur.

Jika keadaan berubah mengikuti anchor donor, ia menyokong sumbangan block itu
dalam dua background ini. Jika tidak, interaksi dengan parameter lain masih
mungkin. Pemilihan F/R menggunakan label synthetic sebagai alat diagnosis,
bukan mekanisme latihan tanpa label atau bukti generalisasi/akses hardware.

## Hasil fixed lexical400

| Background | Anchor donor | Anchor TV | Context TV | F / R betul | Lexical AUC | Lexical loss | MLflow |
|---:|---:|---:|---:|---|---:|---:|---|
| 29 | 29 | 0.001861 | 0.000000837 | 6/6 / 0/6 | 1.0 | 0.26280 | `44af1bc171e340bf8b6d9d4c02c8c191` |
| 29 | 43 | 0.994943 | 0.991587 | 6/6 / 6/6 | 1.0 | 0.13749 | `18a3b4c0af9248f58299234ea381f70f` |
| 43 | 43 | 0.993516 | 0.992376 | 6/6 / 6/6 | 1.0 | 0.13753 | `b51e07f7fbb4432492fe2a6556049750` |
| 43 | 29 | 0.067106 | 0.0000499 | 6/6 / 0/6 | 1.0 | 0.36984 | `ec16dbc2d3294b10be99aa19eb9cd2d1` |

Hasil mengikuti donor anchor untuk dua seed ini. Ini menyokong pengaruh block
initialization anchor dalam eksperimen terhad; ia tidak membuktikan setiap satu
daripada 16 anchor penyebab, atau membuktikan pembaikan untuk dataset/seed lain.
Kedua-dua swaps hanya mengubah raw bias/internal anchor F/R. Snapshot 20-update,
hash sumber/checkpoint, thread environment dan result terperinci disimpan dalam
artifact `config.json` setiap run; masa latihan 8.66 dan 8.54 saat. Memory peak
tidak diukur. Hasil validation ini bukan gate baru atau bukti hardware.

Critic 6.1 Sol Ultra memeriksa source/runner hashes, snapshot asal dan donor,
serta mengira semula outcome menggunakan enumeration NumPy. Beliau mengesahkan
32 raw bias dan 16 internal coupling sahaja ditukar, bank/song/M/shared
interaction kekal, dan lexical400 ialah endpoint. Outcome berjaya/gagal
mengikuti donor, tetapi tahap collapse tidak berpindah secara identik.
