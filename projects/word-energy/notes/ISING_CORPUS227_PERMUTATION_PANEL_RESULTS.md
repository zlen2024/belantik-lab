# Corpus 227: eksperimen lapan permutasi

Susulan eksploratori 16 November 2026. Protokol dan kod latihan dibekukan sebelum panel (commit `49bcd6b`): [protokol](ISING_CORPUS227_PERMUTATION_PANEL_PROTOCOL.md). Corpus ini dipilih selepas keputusan asal dilihat; keputusan di sini tidak mengubah kegagalan gate enam corpus.

Setiap baris ialah satu susunan berbeza dengan empat model seed. Lulus bermaksud sense >=10/12, lexical AUC >=0.75 dan gain >=0.03 pada joint100. Bilangan dalam kurungan ialah seed yang lulus, bukan dataset bebas.

| Seed permutasi | Bilangan lulus daripada 4 |
|---:|---:|
| 7011 | 3/4 |
| 7013 | 3/4 |
| 7019 | 4/4 |
| 7027 | 2/4 |
| 7039 | 3/4 |
| 7043 | 0/4 |
| 7057 | 1/4 |
| 7069 | 3/4 |

Jumlah run lulus ialah 19/32 pada joint100 dan 19/32 pada lexical400. Ini adalah ukuran deskriptif bagi lapan susunan pada satu corpus, bukan 32 replikasi dataset. Susunan 7019 juga 4/4 seperti RNG6061, tetapi 7043 hanya 0/4: hasil bergantung pada susunan dan seed.

Kawalan corpus 227 terdahulu: spectral 4/4, random 2/4, permutasi RNG6061 4/4. Empat model seed ialah 251, 257, 263 dan 269; susunan lama telah dipilih untuk susulan kerana 4/4, jadi bukan sampel tanpa pemilihan bersama lapan susunan baru. Kejayaan beberapa susunan melemahkan idea bahawa hanya RNG6061 boleh berjaya pada corpus ini; ia tidak membuktikan punca atau manfaat susunan spectral secara umum.

Semakan posthoc critic GPT-6.1 Sol Ultra membandingkan 7019 (4/4) dengan 7043 (0/4): bagi 7043, dua keluarga anchor pada akhirnya hampir menggunakan keadaan yang sama (purata anchor TV 0.00550), berbanding 0.99737 untuk 7019. Kedua-duanya mempunyai AUC 1.0. Pada awal latihan, centroid F–R 7043 sebenarnya lebih jauh; ini bermakna jarak awal semata-mata tidak menjelaskan keputusan. Ini pemerhatian terhadap dua susunan terpilih, bukan bukti punca. Lihat [audit dan ukuran semua permutasi](ISING_CORPUS227_PERMUTATION_CRITIC.md).

Data latihan, oracle bagi negatives/corruptions, corrected F3, 400 lexical updates, handoff dan 100 joint updates kekal. Semua run klasik exact enumeration; tiada test, teks sebenar, THRML atau hardware diukur. Lapan permutasi menyiasat satu corpus terpilih dan tidak membuktikan mekanisme sebab-akibat atau generalisasi ke corpus lain.

MLflow: `word-energy-corpus227-permutation-panel-v1`, summary run `a405b027a563409288b7ace0e20b7196`. Artifak setiap run menyimpan konfigurasi, hash, titik semakan dan metrik.

## Rekod setiap run

| Permutasi | Model seed | Run ID | Lexical pass | Sense joint | AUC | Gain | Joint pass |
|---:|---:|---|---|---:|---:|---:|---|
| 7011 | 251 | `6f3e25c2982a4f7ebac744d1118f23fa` | tidak | 0.500 | 1.000 | 0.000004 | tidak |
| 7011 | 257 | `a81e9ed97fe64d9593034c58d734f925` | ya | 1.000 | 1.000 | 0.445459 | ya |
| 7011 | 263 | `c0cb14fd898e412093fa262c3f1f5394` | ya | 1.000 | 1.000 | 0.492421 | ya |
| 7011 | 269 | `175649710c044f128c6dc27d7860bf30` | ya | 1.000 | 1.000 | 0.444160 | ya |
| 7013 | 251 | `0e9c28c482664c809414895200fd804c` | ya | 1.000 | 1.000 | 0.446199 | ya |
| 7013 | 257 | `48cbf0e294b94a9d86fe05682425bd3e` | ya | 1.000 | 1.000 | 0.444325 | ya |
| 7013 | 263 | `5f9da26155cb4f9488e5b5eef491f2e4` | ya | 1.000 | 1.000 | 0.444253 | ya |
| 7013 | 269 | `f5eadaaa72da434d94a025b6b93476f7` | tidak | 0.500 | 1.000 | 0.000009 | tidak |
| 7019 | 251 | `c85048259435451d8303e69fa2fe5f71` | ya | 1.000 | 1.000 | 0.443736 | ya |
| 7019 | 257 | `59c5490c538344eb9b0dd4a0ae9ee8bd` | ya | 1.000 | 1.000 | 0.443748 | ya |
| 7019 | 263 | `597c28b5431a4ddd8b574c2465d07ef3` | ya | 1.000 | 1.000 | 0.445014 | ya |
| 7019 | 269 | `32edfff4f08c4cec96ff40015ab3c39b` | ya | 1.000 | 1.000 | 0.444284 | ya |
| 7027 | 251 | `2e4790f1b9fc42f59bf8b75409f79937` | ya | 1.000 | 1.000 | 0.445195 | ya |
| 7027 | 257 | `1c2ccacdcc5b43ad935b8488112a730e` | ya | 1.000 | 1.000 | 0.445237 | ya |
| 7027 | 263 | `e39942d30222472581a05a50e423e89b` | tidak | 0.500 | 1.000 | 0.000000 | tidak |
| 7027 | 269 | `12305b83df764827b12ae7f7c8e4f476` | tidak | 0.500 | 1.000 | 0.000001 | tidak |
| 7039 | 251 | `53ffaa650f3842aea9aea783b9082300` | tidak | 0.500 | 1.000 | 0.000344 | tidak |
| 7039 | 257 | `fb608ce0e9df4ce3b32f96772f3ee02f` | ya | 1.000 | 1.000 | 0.443959 | ya |
| 7039 | 263 | `d611f8f17fe04abebfa042f24eefb8b6` | ya | 1.000 | 1.000 | 0.442267 | ya |
| 7039 | 269 | `dcef8c14a3184c4180da24e92f076b33` | ya | 1.000 | 1.000 | 0.443598 | ya |
| 7043 | 251 | `6a6cfd03d1c24e8a99392e1fa0bfabac` | tidak | 0.500 | 1.000 | 0.000002 | tidak |
| 7043 | 257 | `244111db332549e89d94990f2ffcc18e` | tidak | 0.500 | 1.000 | 0.000004 | tidak |
| 7043 | 263 | `0e5325a527f44bd08e7d74fc92991a5c` | tidak | 0.500 | 1.000 | 0.000000 | tidak |
| 7043 | 269 | `ba66d296c3894d528198e6c573ae3393` | tidak | 0.500 | 1.000 | 0.000000 | tidak |
| 7057 | 251 | `434a62e0dcbf45b6a67f53d1f8ff36dd` | tidak | 0.500 | 1.000 | 0.000006 | tidak |
| 7057 | 257 | `49f2fcffc0d14cb7b44e8dc59953936f` | ya | 1.000 | 1.000 | 0.445863 | ya |
| 7057 | 263 | `d98bce67ae374ba39cbf2d631bc68344` | tidak | 0.500 | 1.000 | 0.000000 | tidak |
| 7057 | 269 | `0436d758c6f144ff9700f336b324ea95` | tidak | 0.500 | 1.000 | 0.044860 | tidak |
| 7069 | 251 | `f623830ca3bd4c9db239e59eea6d7e1b` | ya | 1.000 | 1.000 | 0.444489 | ya |
| 7069 | 257 | `952894de9bfc4ec4a2d6d9d97d730285` | ya | 1.000 | 1.000 | 0.445213 | ya |
| 7069 | 263 | `632560e2ca464b87be2659faa4fa89fb` | tidak | 0.500 | 1.000 | 0.000113 | tidak |
| 7069 | 269 | `7302f2084b2e418293be5e4e2b277eee` | ya | 1.000 | 1.000 | 0.444338 | ya |
