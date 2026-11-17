# Factorial development: frekuensi dan peranan target/context

17 November 2026. Pelan ini dipilih selepas hasil pilot dan analisis frequency
strata dilihat. Critic GPT-6.1 Sol Ultra menyemak reka bentuk. Ini eksperimen
**posthoc development** pada validation yang sama, bukan confirmatory test.
Jangan ubah coefficient, memilih epoch lain atau membuka reserved test selepas
melihat hasil.

## Soalan dan empat arm

Gunakan sample, vocabulary, q, five training negatives per pair, orders,
validation20 noises dan data hashes daripada dataset run
`6cf76591c3fa49c686a3d81d7f225607`. Tiada labels/family oracle. Tiga seed
17/29/43; fixed tiga epoch, batch256, Adam lr0.01, clip5 dan objective ranking
`mean softplus(1-score_pos+score_neg)` yang sama. Endpoint epoch3. Empat arm:

| Arm | Unary landscapes | Candidate offset |
|---|---|---|
| `tied` | satu table dikongsi target/context | tiada |
| `tied-frequency` | satu table dikongsi | `b(v)=0.25 log(count_train(v))` |
| `role` | table target dan context berasingan | tiada |
| `role-frequency` | table target dan context berasingan | b(v) sama |

Offset coefficient dibekukan kepada 1 dan digunakan pada training ranking
serta validation: `score(u,v) = -F2(u,v) + b(v)`. Ia berasal dari unigram
dan q∝count^.75; constant normalisasi tidak mempengaruhi ranking. Ia **outer
candidate prior/readout pada host**, bukan automatic intrinsic physical energy:
menambah constant pada unary energy yang dinormalisasi akan saling membatalkan.

Role-specific bermula dengan dua salinan table awal tied yang identik dan
shared interaction yang sama, kemudian tables dilatih berasingan. Ia
meningkatkan trainable parameters dari `3V+2` kepada `6V+2`, serta melanggar
andaian satu table Q/K pelan asal sebagai *diagnostic ablation*. Jadi perubahan
result tidak boleh diatribusi kepada simetri sahaja: tambahan capacity dan
optimizer dynamics juga berubah. Graph aktif F2 kekal empat spins dengan empat
potential edges dan max degree2; ini tidak membuktikan adjacency atau precision
chip. Tied-frequency juga menjadi directed **pada readout** kerana b(v),
walaupun normalized F2 asas masih symmetric.

## Keputusan dan numerical controls

Evaluasi dengan canonical unique-pair scorer; unordered keys hanya untuk
`tied` tanpa offset. Baki arm memakai ordered keys supaya offset/peranan tidak
terpadam. Analytic zero-coupling tanpa offset mesti AUC0.5 dan MRR `H21/21`;
zero-coupling dengan offset mesti menghasilkan frequency-only control,
bukannya chance. Check original 16-state F2 vs stable log1p/expm1 serta
gradients; same-ID/reverse/identical-unary/zero/frequency fixtures, batch-size
and candidate-order invariance, expected MRR ties dan no OOV bridging.

Primary descriptive effects per seed pada epoch3 ialah frequency effects
`tied-frequency − tied`, `role-frequency − role`; role-parameterization
effects `role − tied`, `role-frequency − tied-frequency`; interaction ialah
`role-frequency − role − tied-frequency + tied`. Report setiap seed, mean,
parameter count, timing/memory, MRR@21 secondary dan strata frequency
deskriptif. Frequency-only/old vector controls kekal contextual references,
bukan arm yang diklaim parameter-matched.

Tandakan satu effect `material dalam panel development ini` hanya jika
ketiga-tiga paired seeds positif **dan** mean >=0.005 AUC; ia bukan p-value
atau bukti corpus-independent. Laporkan sifar/negatif jika muncul. Dataset
dan semua seeds berkongsi corpus/validation; candidates noise mungkin benar
secara semantik dan text8 tidak mempunyai article boundaries. No sentence or
word-sense claim. Tiada THRML sampler parity atau physical hardware diukur.

Empat arms × tiga seeds =12 sequential runs, ≤4 requested CPU threads,
2 GiB peak process working set check pada epoch, global 45-minute deadline.
Log semua models/checkpoints, pair/noise hash, source, data, runtime dan
failure dalam MLflow experiment baharu. Kekalkan raw pilot dan correction.
