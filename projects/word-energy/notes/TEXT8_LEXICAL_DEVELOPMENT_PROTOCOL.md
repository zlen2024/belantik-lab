# Text8: pilot lexical tanpa label keluarga

17 November 2026. Pelan disemak GPT-6.1 Sol Ultra sebelum latihan. Tujuan
pilot ialah membandingkan *word–context compatibility* pada teks sebenar,
bukan membuktikan polysemy, prestasi bahasa umum atau execution hardware.

## Data dan pemisahan

Sumber tempatan `word2energy/landscape_ebm/text8_data/text8.zip`; satu entry
100,000,000 byte, MD5 kandungan `3bea1919949baf155f99411df5fada7e`.
Sumber asal: [Matt Mahoney](https://mattmahoney.net/dc/textdata.html).
Bytes train `[10_000_000,11_000_000)`, validation `[16_000_000,16_500_000)`;
reserve `[18_000_000,20_000_000)` tidak dinilai. Kawasan ini tidak muncul
dalam julat yang ditemui dalam skrip historical yang diperiksa, tetapi semua
provenance lama tidak diketahui. Text8 tidak mengekalkan sempadan artikel:
ini **development** sahaja, bukan held-out document benchmark yang pristine.
Hash penuh archive/kandungan boleh dibaca untuk integriti tetapi reserved
bytes tidak digunakan untuk sampling, latihan atau metrik.

Token ASCII text8 dipisahkan mengikut ruang, potong perkataan separa pada
boundary. Train-only vocabulary: 1000 token kerap, minimum count5, susunan
count menurun kemudian alphabet untuk tie. Simpan posisi asal; jangan buang
OOV lalu rapatkan jirannya. Sample 50,000 pasangan train dan 5,000 validation
dengan replacement: satu center layak dipilih kemudian satu konteks dalam
offset -2,-1,+1,+2 secara uniform antara context dalam vocabulary. Data
positions dan pasangan boleh berulang; report coverage dan unique counts.
Tiada subsampling perkataan kerap untuk pilot ini.

Train-only `q(token) ∝ count^0.75`, lima noise per pasangan train dan 20 per
pasangan validation. Noise unconditional dibenarkan sama dengan positif atau
berkaitan secara semantik: ia bukan label bahawa perkataan itu salah. Seeds
sampler dibekukan; simpan pasangan, posisi, noise, urutan dan hash. Tiga epoch
setiap arm menerima contoh, per-epoch negative draws dan batch order sama.
Tiada toy grammar, family labels atau holdout-based initializer digunakan.

## Empat arms, tiga initialization seeds

1. `ising-ranking`: 2 spins/token, normalized pair F2, skor `-F2`; loss
   `mean softplus(1-score_pos+score_neg)` atas lima negatives.
2. `vector-ranking`: dua embedding table 2D, skor dot product, **loss ranking
   yang sama**. Ini bukan SGNS.
3. `sgns`: dua embedding table 2D, dot score, loss standard
   `softplus(-positive)+sum_5 softplus(negative)`; konsep [Mikolov et al.](https://arxiv.org/abs/1310.4546).
4. `ising-shuffled`: sama seperti Ising ranking tetapi context column train
   dipermutasikan secara tetap; original validation dan noise dikekalkan.
   Ini kontrol corruption, bukan kaedah pembelajaran yang disyorkan.

Seeds model 17,29,43. Batch256, tiga epoch, Adam lr0.01 dan global clip5.
Initial dan selepas setiap epoch diukur; **epoch3 tetap** ialah endpoint,
tanpa checkpoint selection atau tuning pada validation. Model Ising mempunyai
`3V+2` parameter; dua-table vector 2D mempunyai `4V`. Laporkan perbezaan ini:
tiada claim fair parameter-count exact atau comparison semua hyperparameter.
Ising F2 symmetric dengan dua shared positive couplings; SGNS untied dan
asymmetric. Noise dan data setara tidak bermaksud objective atau architecture
setara. Losses antara objectives berbeza tidak dibanding secara langsung.

## Metrik dan batasan

Per model/seed: pairwise positive-vs-noise AUC (ties 0.5), MRR@21 atas satu
positif dan 20 noise tetap dengan expected reciprocal rank bagi ties, positive
dan noise score spread, train/validation objective masing-masing, run time,
initial → epoch1/2/3, dan OOV/coverage. Pecahkan AUC kepada tiga kumpulan
frequency center menurut train counts bagi validation occurrences, dengan
stable sort dan tiga kumpulan hampir sama besar; occurrences dan tiga
seeds berkongsi corpus dan bukan independent documents. Sebagai kontrol murah
tanpa training laporkan skor frekuensi `0.25*log(train_count[context])`,
yang boleh mendapat signal disebabkan q^0.75. Ising zero-coupling memberi
skor tie dan AUC 0.5 secara matematik; semak numerical fixture.
Validation objective menggunakan lima noise pertama agar bentuk objective
sama dengan training; semua 20 noise digunakan untuk AUC/MRR@21.

Data dan run disimpan sebagai MLflow artifacts (dataset run berasingan, semua
seed merujuk dataset hashes); no test evaluation. Batas batch: satu proses
run pada satu masa, target <=2 GiB working set, OMP/OPENBLAS 4 dan 45 min
global deadline; checks are snapshots rather than hard OS caps. Stop jika
nonfinite atau resource/provenance failure dan rekod run tidak lengkap.

Baseline hanya menguji **noise discrimination**, bukan word similarity
manusia atau bank-dua-makna. Penilaian contextual seperti [WiC](https://aclanthology.org/N19-1128/)
memerlukan adapter dan protokol berasingan. Classical exact four-spin F2
training ialah rujukan; tiada CD/PCD, THRML sampler parity, thermodynamic chip
atau hardware speed/energy diukur. Lebih banyak token vocabulary tidak
menambah bilangan state unary bagi setiap perkataan.
