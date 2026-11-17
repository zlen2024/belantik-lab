# Keputusan galakan diversity sementara

Protokol dibekukan sebelum training: [protokol](ISING_ENTROPY_INTERVENTION_PROTOCOL.md). Galakan ialah perubahan loss latihan sementara pada corpus toy terpilih, bukan tenaga fizikal.

Status panel: LENGKAP; 48/48 run siap. Kriteria dibaca daripada ringkasan yang telah diaudit.

Percubaan terputus dikekalkan dalam MLflow dan dipautkan kepada run gantian: `{(7043, 269, 'baseline'): 'b9af12890f824290b86ec81b7ac09d56'}`. Lihat [nota sambungan](ISING_ENTROPY_RESUMPTION_20261117.md).

| Permutasi | Baseline | Early bank | Late bank | Early song |
|---:|---:|---:|---:|---:|
| 7019 | 4/4 | 4/4 | 4/4 | 4/4 |
| 7027 | 2/4 | 2/4 | 2/4 | 2/4 |
| 7043 | 0/4 | 2/4 | 0/4 | 1/4 |

Practical rescue: **gagal**; 7043 memerlukan sekurang-kurangnya 3/4 tetapi hanya mencapai 2/4. Manipulation check 7043: lulus; median perubahan entropy bank lexical100 +0.4373 nat. Early-bank mengatasi late-bank pada 7043 sebanyak dua seed, tetapi equal coefficient area tidak menjamin daya gradient yang sama. Early-song mencapai 1/4 berbanding early-bank 2/4 pada 7043; ini tidak membuktikan penargetan bank sebagai satu-satunya punca.

Pada 7027, early-bank menukar seed 251 daripada lulus kepada gagal, dan seed 263 daripada gagal kepada lulus: kedua-dua arm mempunyai 2/4 tetapi bukan seed yang sama. Merentas 12 pasangan, early-bank membantu 3, merosakkan 1 dan tidak mengubah 8. Semua 7019 lulus. Lihat [audit critic akhir](ISING_ENTROPY_PANEL_FINAL_CRITIC.md), [audit interim](ISING_ENTROPY_7043_INTERIM_CRITIC.md) dan [nota sambungan serta gangguan](ISING_ENTROPY_RESUMPTION_20261117.md).

Ringkasan MLflow: `7ae14e229dc44555beb53237a4a8ace8`.

Satu run lulus jika sense >=10/12, AUC>=0.75 dan gain>=0.03 pada joint100. Entropy yang meningkat sahaja bukan bukti dua makna dipelajari. Ini eksperimen posthoc pada satu corpus menggunakan oracle semasa latihan; tiada test, teks sebenar atau hardware diukur.

## Rekod setiap run

| Permutasi | Seed | Arm | Run ID | Status | Sense | AUC | Gain | Lulus |
|---:|---:|---|---|---|---:|---:|---:|---|
| 7043 | 251 | baseline | `89be4d294b7d4a1ca9e0251461fd6c9c` | selesai | 0.500 | 1.000 | 0.000002 | tidak |
| 7043 | 251 | early-bank | `7ea1b55f9e2c497f9857369030f3eb13` | selesai | 1.000 | 1.000 | 0.444184 | ya |
| 7043 | 251 | late-bank | `76ea755cbd274f54b40b003525841a5d` | selesai | 0.500 | 1.000 | 0.000002 | tidak |
| 7043 | 251 | early-song | `cc052b6800964a97a473f4923ce1d614` | selesai | 0.500 | 1.000 | 0.000001 | tidak |
| 7043 | 257 | baseline | `97f0bbf590734eb2bd7e1944eef39cf0` | selesai | 0.500 | 1.000 | 0.000004 | tidak |
| 7043 | 257 | early-bank | `f14cb0a40282461da5b443edfdd9d9c3` | selesai | 1.000 | 1.000 | 0.444561 | ya |
| 7043 | 257 | late-bank | `253982c104a94edfbcf560d5f4549e74` | selesai | 0.500 | 1.000 | 0.000005 | tidak |
| 7043 | 257 | early-song | `9767d0fe78d84802a4aa30b1e0c838e7` | selesai | 0.500 | 1.000 | 0.000003 | tidak |
| 7043 | 263 | baseline | `a14d5a5625974c6e854bc3f5cb47243a` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7043 | 263 | early-bank | `444b622c20614a74afa57930bb5ce3c1` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7043 | 263 | late-bank | `b7787fd74024442b9d144f9d5dca27a1` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7043 | 263 | early-song | `16067844af784ab7806da4b09e29ec08` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7043 | 269 | baseline | `77290528213f4d6cb3a5da8cf7c37188` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7043 | 269 | early-bank | `5f59beaa844b4b4b9f0829832157d9fb` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7043 | 269 | late-bank | `314bd8ad67414170933b0e2bd008d262` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7043 | 269 | early-song | `de18c60df5a54889be72d82aa7f5af89` | selesai | 1.000 | 1.000 | 0.439254 | ya |
| 7019 | 251 | baseline | `d1406ca95fe64615b405fbe4fc670825` | selesai | 1.000 | 1.000 | 0.443736 | ya |
| 7019 | 251 | early-bank | `110d0c9bac7144de9aba5d2af7c9fa53` | selesai | 1.000 | 1.000 | 0.443631 | ya |
| 7019 | 251 | late-bank | `d4690efdb5d041e08293f5d69b65bffd` | selesai | 1.000 | 1.000 | 0.443862 | ya |
| 7019 | 251 | early-song | `17a046d2ee10462f9f2fa19551d7b3f4` | selesai | 1.000 | 1.000 | 0.443319 | ya |
| 7019 | 257 | baseline | `0bff92d4057041fda5ef69d8ef5020e0` | selesai | 1.000 | 1.000 | 0.443748 | ya |
| 7019 | 257 | early-bank | `4e82c7a7355f4b1aac9ccff879c4a084` | selesai | 1.000 | 1.000 | 0.443926 | ya |
| 7019 | 257 | late-bank | `c5a138c779374df598ebd284bdaab367` | selesai | 1.000 | 1.000 | 0.444259 | ya |
| 7019 | 257 | early-song | `a15e0ccb7df349e49a13858138a483a7` | selesai | 1.000 | 1.000 | 0.443947 | ya |
| 7019 | 263 | baseline | `ce1d882873274de69ee8c7a9c2433e4d` | selesai | 1.000 | 1.000 | 0.445014 | ya |
| 7019 | 263 | early-bank | `a9faa9277b6e44c2a25b81ce12a3d95c` | selesai | 1.000 | 1.000 | 0.444255 | ya |
| 7019 | 263 | late-bank | `6917dd2756ab459290e3b2faa805927f` | selesai | 1.000 | 1.000 | 0.445122 | ya |
| 7019 | 263 | early-song | `39b812a7c38e46b0a272b6f1a4685e7f` | selesai | 1.000 | 1.000 | 0.445717 | ya |
| 7019 | 269 | baseline | `e979bc56ff994ee6980b718f8c1df7b3` | selesai | 1.000 | 1.000 | 0.444284 | ya |
| 7019 | 269 | early-bank | `e08a92adcb334f7d9d8e3bef4b10d6eb` | selesai | 1.000 | 1.000 | 0.444019 | ya |
| 7019 | 269 | late-bank | `18962553d7104606807db1ce8a9825c9` | selesai | 1.000 | 1.000 | 0.444404 | ya |
| 7019 | 269 | early-song | `16f3c46c2e4d4037837e8ac0450694f5` | selesai | 1.000 | 1.000 | 0.444384 | ya |
| 7027 | 251 | baseline | `08b46b51146b4b7587347407b50d2e6b` | selesai | 1.000 | 1.000 | 0.445195 | ya |
| 7027 | 251 | early-bank | `62bc877df5664e7284f7a0d134dfbbcd` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7027 | 251 | late-bank | `7d70f6e100284afd8fbf8b6e690e4374` | selesai | 1.000 | 1.000 | 0.445281 | ya |
| 7027 | 251 | early-song | `02fbc80d64814684b1e44347ee325bed` | selesai | 1.000 | 1.000 | 0.445210 | ya |
| 7027 | 257 | baseline | `34b820a345f54937b5406e719f89f42f` | selesai | 1.000 | 1.000 | 0.445237 | ya |
| 7027 | 257 | early-bank | `0d771d72f71a4ecc9fbb9521451d3825` | selesai | 1.000 | 1.000 | 0.443996 | ya |
| 7027 | 257 | late-bank | `df16f830a7ab4043a7df648a643169ee` | selesai | 1.000 | 1.000 | 0.445317 | ya |
| 7027 | 257 | early-song | `da7796899566478299f636f527e9e029` | selesai | 1.000 | 1.000 | 0.444970 | ya |
| 7027 | 263 | baseline | `59824c5e93be439b88136bda15ea2369` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7027 | 263 | early-bank | `c1e785205ddc4a0abfb0edb32acc4334` | selesai | 1.000 | 1.000 | 0.447276 | ya |
| 7027 | 263 | late-bank | `5457889173c848db80a3493f4725f5b8` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7027 | 263 | early-song | `cb8c77a691034f9098d08782f94934ec` | selesai | 0.500 | 1.000 | 0.000000 | tidak |
| 7027 | 269 | baseline | `bba2edfd71cc433d9cfc8432df7087b5` | selesai | 0.500 | 1.000 | 0.000001 | tidak |
| 7027 | 269 | early-bank | `2fc8bd843e284e7abe5ba5dd8891646b` | selesai | 0.500 | 1.000 | 0.000001 | tidak |
| 7027 | 269 | late-bank | `4c42d6a3c8db49dea1ba5439e3d20aea` | selesai | 0.500 | 1.000 | 0.000001 | tidak |
| 7027 | 269 | early-song | `04e65cff2b96466493330c50f34cb29d` | selesai | 0.500 | 1.000 | 0.000001 | tidak |
