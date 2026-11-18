# WikiText-2: lexical development dipisahkan mengikut artikel

18 November 2026. Eksperimen development pada [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext/tree/c937478d5e4c5aca1b3f9f12de797092d0fdfdef/wikitext-2-raw-v1); [protokol](WIKITEXT2_DOCUMENT_DEVELOPMENT_PROTOCOL.md) dibekukan dalam commit `1f58962`. 18 run = 3 model × 3 seed × pasangan asal/acak. Train dari kosong menggunakan 200,000 pasangan; validation 10,000 pasangan. Prior frequency, loss, sampel, noise dan optimizer dikongsi.

## Hasil epoch3

| Model | Parameter | Train | AUC micro | AUC macro artikel | MRR@21 |
|---|---:|---|---:|---:|---:|
| tied-frequency | 6002 | original | 0.67471 | 0.67405 | 0.30518 |
| tied-frequency | 6002 | shuffled | 0.64750 | 0.64805 | 0.28374 |
| role-frequency | 12002 | original | 0.67725 | 0.67927 | 0.31505 |
| role-frequency | 12002 | shuffled | 0.64607 | 0.64646 | 0.28360 |
| vector-frequency | 8000 | original | 0.67842 | 0.68044 | 0.32077 |
| vector-frequency | 8000 | shuffled | 0.64585 | 0.64727 | 0.28316 |
| frequency-only | 0 | tanpa model | 0.64772 | 0.64816 | 0.28352 |

Micro mengumpul semua pasangan; macro mengira AUC setiap artikel dahulu kemudian mengambil purata 60 artikel. MRR@21 ialah kedudukan konteks sebenar antara 20 calon noise. Metrik-metrik ini menilai noise discrimination, bukan accuracy makna.

## Perbandingan berpasangan

| Model | AUC micro asal − acak, seed 17 / 29 / 43 | AUC macro asal − acak | Interval bootstrap artikel 95% |
|---|---|---:|---|
| tied-frequency | +0.02861 / +0.02541 / +0.02760 | +0.02599 | [+0.01998, +0.03224] |
| role-frequency | +0.02964 / +0.03456 / +0.02934 | +0.03281 | [+0.02728, +0.03936] |
| vector-frequency | +0.03229 / +0.03111 / +0.03432 | +0.03317 | [+0.02587, +0.04140] |

Bootstrap 1,000 kali, seed 22140: sampel semula 60 artikel validation selepas purata 3 seed dikira bagi setiap artikel. Interval bersyarat pada model, sampel dan satu shuffle ini; artikel mungkin masih berkaitan, bukan bukti pengulangan pada corpus bebas atau pengujian confirmatory. Semua original seed juga mengatasi frequency-only pada AUC micro. Perbezaan model asal terhadap vector pada micro AUC:

| Model | Seed 17 | Seed 29 | Seed 43 | Perbezaan macro purata | Bootstrap artikel 95% |
|---|---:|---:|---:|---:|---|
| tied-frequency − vector | -0.00270 | -0.00364 | -0.00480 | -0.00639 | [-0.01214, -0.00110] |
| role-frequency − vector | -0.00169 | +0.00196 | -0.00379 | -0.00117 | [-0.00435, +0.00204] |

Tied berada di bawah vector untuk tiga seed pada panel ini; role hampir sama dan tanda perbezaannya berubah antara seed. Bilangan parameter berbeza (6,002/12,002/8,000), jadi ia bukan perbandingan capacity-matched dan bukan bukti seni bina Ising lebih baik secara umum. Semua shuffled menghampiri frequency-only; perubahan tidak semestinya semantik dan boleh dipengaruhi marginal atau kemerosotan kawalan.

## Semua run

| Model | Seed | Train | Run ID | AUC micro | AUC macro | MRR | Masa (s) |
|---|---:|---|---|---:|---:|---:|---:|
| tied-frequency | 17 | original | `d848a6d5a711462795a4476589346b33` | 0.67563 | 0.67343 | 0.30588 | 12.3 |
| tied-frequency | 17 | shuffled | `a1c77e6702254bb9870b85b68fdfd660` | 0.64702 | 0.64838 | 0.28370 | 11.1 |
| tied-frequency | 29 | original | `00139333a47d4afb959bbc2c849e9a47` | 0.67385 | 0.67524 | 0.30519 | 13.3 |
| tied-frequency | 29 | shuffled | `12eff0d722114b319f8d07737e909092` | 0.64843 | 0.64886 | 0.28369 | 12.9 |
| tied-frequency | 43 | original | `5bd444057a9c49b9a2df5f5ad933b035` | 0.67464 | 0.67347 | 0.30448 | 12.1 |
| tied-frequency | 43 | shuffled | `192ddf26c856418790d27931ec5b06a5` | 0.64704 | 0.64691 | 0.28385 | 11.3 |
| role-frequency | 17 | original | `e6f37d07e0fa42d98185d5882ee50bb2` | 0.67664 | 0.67825 | 0.31375 | 11.9 |
| role-frequency | 17 | shuffled | `31f66892a93f497ea5150808d3b4b395` | 0.64700 | 0.64758 | 0.28384 | 11.9 |
| role-frequency | 29 | original | `5fcbf994b7ef4e0892e7c97366c964c8` | 0.67944 | 0.68105 | 0.31831 | 14.9 |
| role-frequency | 29 | shuffled | `09df0500e0324c45998e886a349f7f08` | 0.64488 | 0.64594 | 0.28291 | 14.6 |
| role-frequency | 43 | original | `435407d0b8554c80b9eb46fecfff4e2c` | 0.67566 | 0.67851 | 0.31308 | 12.7 |
| role-frequency | 43 | shuffled | `705a9a9bc7d0407fa0faf82135ec5a5b` | 0.64631 | 0.64586 | 0.28403 | 13.2 |
| vector-frequency | 17 | original | `2f3e3780af564fec875967d211939007` | 0.67834 | 0.67990 | 0.32240 | 8.4 |
| vector-frequency | 17 | shuffled | `b23fd4c9745942bca2d0f2f50236cfc5` | 0.64605 | 0.64764 | 0.28350 | 9.9 |
| vector-frequency | 29 | original | `4366495bee144bb0921efcfc0687b883` | 0.67748 | 0.67902 | 0.31785 | 10.0 |
| vector-frequency | 29 | shuffled | `ee91a78cd5324dc6875470a1dda2df99` | 0.64637 | 0.64753 | 0.28315 | 9.0 |
| vector-frequency | 43 | original | `554ff0148e8e4120b3b6c7646a5b8202` | 0.67944 | 0.68239 | 0.32206 | 9.9 |
| vector-frequency | 43 | shuffled | `50879dc9a05c4fb7ae905d3d44fb0e6e` | 0.64512 | 0.64664 | 0.28282 | 10.1 |

## Data dan batas

- **600 artikel train dan 60 validation** berdasarkan tajuk top-level dikelilingi baris kosong; tiada normalized title, whole-document hash atau exact long-paragraph yang sama antara split. Ujian ini tidak mengesan semua near-duplicate atau overlap dengan text8.
- Tokenizer projek lowercase `[a-z]+`, heading dikeluarkan dan pasangan tidak menyeberang perenggan, artikel atau OOV di tengah. Data train: 1,681,017 token, coverage 74.35%; validation 175,628 token, coverage 74.06%. Vocabulary 2,000 dan prior dibina daripada train sahaja. 60/60 artikel validation disampel.
- Satu context shuffle dikongsi oleh semua seed dan mengekalkan kekerapan calon. Positive/noise collisions dibenarkan untuk kedua-dua keadaan. Checkpoints awal/epoch1–3, hashes, sumber, shuffle, noise dan orders direkod. Zero-coupling/zero-vector memulangkan frequency-only.
- Corpus, tokenizer, bilangan pasangan dan vocabulary berubah berbanding text8, jadi beza skor tidak mengenal pasti kesan scaling secara berasingan. WikiText dan text8 sama-sama sumber Wikipedia; independent-domain dan historical disjointness belum dibuktikan. Noise mungkin konteks yang sah.
- Tiada labels sense, THRML sampler parity atau cip fizikal diuji. Test WikiText tidak dimuat turun; reserved text8 tidak diakses untuk latihan atau penilaian.

MLflow experiment `word-energy-wikitext2-doc-development`: dataset `b37da8076fd54f4da389082278e9733a`, summary `5b8b2958f2e249ea834766de478bba56`. Dataset sumber train/validation dan kod disimpan dalam Git; array eksperimen dan checkpoints disimpan dalam MLflow tempatan.

Semakan bebas dan tafsiran: [laporan critic](WIKITEXT2_DOCUMENT_DEVELOPMENT_CRITIC.md).
