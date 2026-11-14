# Eksperimen attribution Ising — keputusan awal

15 November 2026. Eksperimen exploratory pada validation, menggunakan seed asal
17/29/43; tiada retraining pada test atau checkpoint selection menggunakan test.
Latihan menggunakan exact enumeration, bukan simulator hardware.

Replay original menghasilkan checkpoint terpilih asal dengan max absolute
parameter difference masing-masing 1.78e-15, 2.00e-15 dan 1.55e-15; gate
1e-10 lulus. Validation ranking loss sepadan dalam 1.2e-16 dan langkah
checkpoint yang dipilih sama dengan run asal. Kedua-dua cabang kawalan disemak bermula dari parameter yang sama
tepat dengan checkpoint awal atau lexical terpilih masing-masing.

| Seed | Selepas lexical: anchor TV | Original joint100: anchor TV / sense | Joint dari awal: anchor TV / sense | Lexical tambahan: anchor TV / sense |
|---:|---:|---|---|---|
| 17 | 0.99513 | 0.99766 / 1.0 | 0.65735 / 1.0 | 0.99792 / 1.0 |
| 29 | 0.00186 | 0.000828 / 0.5 | 0.29087 / 0.5 | 0.000894 / 0.5 |
| 43 | 0.99352 | 0.99710 / 1.0 | 0.25581 / 0.5833 | 0.99744 / 1.0 |

TV ialah total variation antara distribusi anchor F/R; 0 sama, 1 terpisah.
Validation sense diukur pada endpoint sama (step100); bukan test atau generalisasi.
Pada seed29, anchor TV awal 0.05009 menjadi 0.00186 selepas lexical,
sementara perbezaan purata posterior F/R menjadi 0.00000084. Seed17 dan 43
telah terpisah kuat selepas lexical. Seed29 joint-from-initial menaikkan anchor
TV kepada 0.29087 tetapi sense masih 0.5; ini tidak menyelamatkan pilot.

Keputusan menyokong bahawa collapse pada seed29 sudah wujud pada checkpoint
lexical terpilih dan joint asal tidak memulihkannya. Ini belum membuktikan
initialization atau objective ialah satu-satunya punca. Branch tambahan
bersifat exploratory dan tidak mengubah status gagal gate pilot asal.
Joint dari awal masih mengandungi ranking + 0.25 lexical; ia tidak mengasingkan
kesan ranking sahaja.

| Seed | Replay original | Joint dari awal | Lexical tambahan |
|---:|---|---|---|
| 17 | `c53aaab7073c48fdaf0ecd963ceb0269` | `f9cda07fb2f548509c23aba4989321ff` | `6e10df643b684713b28118ed05b992ed` |
| 29 | `44af1bc171e340bf8b6d9d4c02c8c191` | `8e77572f853f4d1b8a3fd7c86cb3fa6e` | `7870cabfba914082aad61fcbd02740c7` |
| 43 | `b51e07f7fbb4432492fe2a6556049750` | `8bccf4d206714b6f9110f1f4a11b492d` | `4b59cfadea094fba84170fd13bbb55a3` |

Setiap run menyimpan parameter snapshot setiap 20 update sebagai artifact.
Analisis tambahan dengan enumeration NumPy secara bebas merekod setiap snapshot:
anchor distributions/entropy, prior bank, posterior 12 window, family gain dan
accuracy, serta parameter saturation. Untuk seed29, anchor TV meningkat daripada
0.0501 pada awal kepada 0.1779 pada update 20, kemudian turun kepada 0.0339 pada
update 60, 0.00815 pada update 100 dan 0.00186 pada update 400. Ini menunjukkan
corak kemerosotan dalam snapshot yang diperiksa, bukan bila tepat perubahan itu
berlaku di antara dua snapshot.

Run MLflow trajectory untuk seed 17, 29, 43 (susunan original, joint dari awal,
lexical tambahan):

- 17: `0156d21a75174f78bedc70c335e6fd6f`, `fc1fc8000620464b83304ac12d928066`, `8b535cbfb6aa47daa50cfa458da3d6bf`.
- 29: `fecff11a97e445a9a44e2fff1ebf19b1`, `49a3685befc44864807549c5c3f15a32`, `ca1d92a4fb25405c8befe48968c61061`.
- 43: `5d9379802f2c45ebad47e13fdbbb7636`, `b4d3bd881dad4286aa3ad6d320424ccd`, `e241fbd8cca84471acb99bd5465514df`.

Analisis tambahan mengira lexical loss validation dan fixed training batch,
ranking loss, total objective dan gradient L2 mengikut parameter pada fixed
training batches untuk setiap snapshot. Gradient ini ialah probe pada batch
tetap, **bukan** gradient yang digunakan pada update sebenar. Run objective
untuk seed 17, 29, 43 (susunan original, joint dari awal, lexical tambahan):

- 17: `7fa7682b176a4fd79e3898514bf326f7`, `3794018a7ac34b5e93dab3f52ac25a4c`, `2df4a81ff93c49f895e92e5ae983e289`.
- 29: `c8ce2dde79914f61bd2ab518db97a2d8`, `d4ac5f67d54c4dd8bd0d1bdf7cde5fb0`, `94b94a9686764c01a4e2541008506e18`.
- 43: `43174fc8f5a248bdac2a39c907bc8aa1`, `4807a9b2aadb49f0872f160190011994`, `b24173d3c06d4f6d90f3b4cfa4562b3e`.

Lexical AUC dan gradient individu untuk setiap anchor pada setiap snapshot belum
direkod. Resource peak juga tidak diukur; tetapan thread dibatasi kepada empat
melalui environment, tetapi had memory dan masa keseluruhan belum diaudit secara
menyeluruh. Critic 6.1 Sol Ultra mengesahkan kesimpulan endpoint dan membatasi
interpretasi; protocol tidak boleh dianggap lengkap pada butiran tersebut.
