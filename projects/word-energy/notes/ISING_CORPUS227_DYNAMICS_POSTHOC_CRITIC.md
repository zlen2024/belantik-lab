# Audit posthoc: mengapa susunan 7043 gagal?

16 November 2026. Critic GPT-6.1 Sol Ultra. Analisis ini menggunakan snapshots
yang telah disimpan: 32 permutation runs dan 12 controls corpus 227, lima
checkpoints setiap run (220 checkpoints). Tiada training tambahan, test scoring,
perubahan checkpoint/protokol atau penulisan kepada MLflow database dilakukan.
Keputusan gate enam corpus kekal gagal.

Sumber boleh diulang:

- `scripts/diagnose_corpus227_permutation_dynamics.py`
- `experiments/diagnostics/corpus227-permutation-dynamics-20261116.json`

Jalankan dari project root dengan
`.venv/Scripts/python.exe scripts/diagnose_corpus227_permutation_dynamics.py`.
Skrip membaca SQLite menggunakan `mode=ro`, merekod hash setiap source config
dan hash skrip sendiri, kemudian menulis JSON diagnostik baharu. Ia tidak
memanggil trainer atau optimizer. Runtime numerical terakhir sekitar lima saat.

## Kesimpulan yang disokong

7043 mengikuti trajectory yang menumpukan bank, F dan R pada hampir satu state
yang sama. Ia tetap dapat memisahkan bank-associated words daripada music,
sebab itu lexical AUC ialah 1.0, tetapi tidak dapat memilih F/R mengikut context.
7019 membentuk states F/R yang berlainan dan mengekalkan lebih daripada satu
state bank dengan probability besar. Ini menerangkan *keadaan representasi yang
menghasilkan kegagalan*, bukan mengenal pasti satu ciri initial permutation
sebagai sebab yang telah dibuktikan.

| Ukuran, purata empat seeds | 7019: 4/4 | 7043: 0/4 |
|---|---:|---:|
| Initial raw F–R centroid distance | 0.04772 | 0.11609 |
| Initial anchor TV | 0.08342 | 0.13470 |
| Initial full-train F2 loss | 1.32679 | 1.31811 |
| Lexical20 F2 loss | 1.07564 | 0.95754 |
| Lexical20 bank entropy, nats | 1.24647 | 0.84477 |
| Lexical100 bank entropy | 0.81101 | 0.09465 |
| Lexical400 bank entropy | 0.77883 | 0.03114 |
| Joint100 bank entropy | 0.77606 | 0.01489 |
| Joint100 anchor TV | 0.99737 | 0.00550 |
| Joint100 full-train F2 loss | 0.13026 | 0.25974 |
| Joint100 cross-F/R-negative loss | 0.20873 | 1.30900 |

7043 bermula dengan centroid separation lebih besar dan loss sedikit lebih
rendah. Pada lexical20 loss masih lebih rendah, tetapi bank entropy sudah
menurun jauh lebih cepat. Maka initial separation atau initial aggregate loss
sahaja bukan penjelasan yang mencukupi.

## Apa yang tersembunyi oleh aggregate loss

Negatives synthetic masih menggunakan family/grammar oracle. Daripada 1,440
negative comparisons canonical, 271 (18.82%) ialah cross-F/R comparisons.
Pada joint100, comparisons ini menyumbang 0.24635 daripada total F2 loss
0.25974 untuk 7043, kira-kira 94.84%. Model menyelesaikan kebanyakan masalah
lain sambil mengekalkan masalah pemisahan F/R. 7019 mencapai cross-F/R loss
0.20873, dengan contribution 0.03928 daripada total 0.13026.

Ini bukan kerana softplus telah mengabaikan comparisons yang salah. Mean
ranking slope `sigmoid(margin + Fpos - Fneg)` bagi cross-F/R ialah 0.72990
untuk 7043, lebih besar daripada 0.18685 bagi 7019. Namun norm contribution
gradient cross-F/R kepada full F2 objective cuma 0.001355 bagi 7043, berbanding
0.008393 bagi 7019. Probability concentration dan perbezaan/cancellation
moments boleh memberi sensitivity parameter kecil walaupun ranking loss besar.

Full F2 gradient menggunakan exact unary-minus-joint moments; ia berbeza
daripada actual next minibatch gradient. Mean norm pada lexical400 ialah
0.007190 (7043) berbanding 0.017937 (7019). Pada joint100, norm full joint
objective (`F3 ranking + 0.25 F2`) ialah 0.009881 berbanding 0.029151.
Gradient tidak sifar: jangan menamakan checkpoint ini stationary local minimum.
Saved Adam moments tidak tersedia, dan full canonical gradient tidak menentukan
actual next Adam update. Diagnostics menyokong small-gradient concentrated
regime dalam fixed budget, bukan bukti bahawa model tidak boleh pulih dengan
training lain atau budget lebih panjang.

## Bank local fields dan global spin symmetry

Contoh model seed 251 pada joint100:

- 7043 bank prior ialah `[0.000725, 0.000256, 0.998231, 0.000789]`.
  Mean F/R anchors memberi 0.98444/0.99025 kepada state dominan yang sama.
- 7019 bank prior ialah `[0.017074, 0.534081, 0.001703, 0.447142]`.
  F dan R anchors menumpu kepada dua states berlainan.

Pada dominant unary bank state `s*`, sorted single-spin flip costs ialah
`2*s*_a*(b_a + k*s*other)`. Ini dimensionless beta-E differences, bukan joules.
Untuk seed251, kos tersebut ialah `[7.1432, 7.2282]` bagi 7043, berbanding
`[0.1777, 3.4430]` bagi 7019. Jadi 7019 mempunyai adjacent bank state yang
hampir seimbang, sementara kedua-dua adjacent flips bank 7043 amat tidak
digemari. Mean minimum cost empat seeds ialah 7.1368 berbanding 0.1483.

Absolute bias sign bukan penjelasan: ia berubah di bawah global spin relabeling.
Global two-axis flip menegasikan semua biases dan mengekalkan internals;
one-axis flip juga mesti menegasikan setiap internal coupling. Axis swap turut
menukar interaction coordinates. Semua lapan signed-axis symmetries bagi dua
saved endpoints diperiksa: loss dan transformed gradients sepadan sehingga
1.11e-16. Tiada flipped runs dilatih.

Nilai bank awal sahaja juga tidak mencukupi. 7027 dan 7043 mempunyai bank bias
awal yang sama, serta bank internal draw yang sama bagi paired seed, tetapi
hasilnya 2/4 dan 0/4. 7039 dan 7057 juga sama pada initial bank parameters,
tetapi hasilnya 3/4 dan 1/4. Relasi dengan susunan token lain, song center,
train-pair schedule dan interaction dynamics masih calon penjelasan.

## Batasan dan competing explanations

Perubahan row arrangement memang diasingkan dalam panel, tetapi audit posthoc
tidak mengasingkan ciri row arrangement yang menentukan trajectory. Correlated
initial alignment, center/anchor interactions, fixed negative/batch schedule
dan stochastic initialization lain belum dipisahkan oleh intervention khusus.
Bounded parameterization juga tidak membuktikan hard tanh saturation: bank
effective biases seed251 7043 ialah `[-2.0460, 2.0885]`, dengan internal coupling
`-1.5256`; combined local fields menjelaskan concentration tanpa mencapai
bias bounds `+/-2.5`.

Across 32 new runs, semua 19 passing endpoints mempunyai bank entropy
0.7675–0.9022, manakala 13 failing endpoints 0.01265–0.11387. Ini association
pada observed endpoints sahaja, bukan threshold baharu. F/R anchor collapse
bukan universal: 7057 seed269 gagal walaupun anchor TV 0.8143, kerana satu
family masih dipilih pada semua contexts. Jangan menggeneralisasi failure
pattern khusus 7043 kepada setiap run yang gagal.

Analytic directional gradients diuji terhadap central finite differences pada
dua endpoints dengan dua step sizes; maximum error 1.82e-11. Dua label-aware
F/R separation directions mempunyai positive directional curvature pada dua
step sizes yang sepadan, tetapi pemeriksaan dua directions ini bukan Hessian
audit menyeluruh dan tidak membuktikan local minimum atau loss barrier.

Bounded next action: simpan dan gunakan audit ini untuk memperincikan penjelasan
kepada finite-budget collapse/sensitivity terhadap keseluruhan initial geometry.
Sebarang training intervention seterusnya perlu menjadi eksperimen baharu;
audit ini tidak mengubah gate atau menambah run latihan.
