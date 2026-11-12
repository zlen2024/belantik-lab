# Pelan Word Energy Embedding v2

11 November 2026. Status: reka bentuk untuk pelaksanaan; belum dilatih atau divalidasi. Pelan ini berasaskan [teks asal](TEKS_ASAL_BIKAR_ENERGY_EMBEDDING.txt), [rumusan yang disemak](RUMUSAN_IDEA_BIKAR_ENERGY_EMBEDDING.md), kod tempatan dan sumber primer di bawah. Formulasi baharu dalam dokumen ini ialah cadangan reka bentuk, bukan formula yang pengguna telah berikan.

## 1. Keputusan reka bentuk

Satu perkataan memetakan kepada satu fungsi skalar dalam ruang semantik bersama:

\[
w\longmapsto U_w(z;\theta_w),\qquad z=(z_1,\ldots,z_D)\in\mathbb R^D.
\]

Parameter masih disimpan sebagai nombor/vektor. Yang berubah ialah cara menggunakannya: nombor tersebut menentukan bentuk satu landscape, bukan hanya satu lokasi. Q dan K menggunakan jenis representasi dan jadual parameter perkataan yang sama. Apabila Q bertukar, kita membaca baris/perspektif yang berbeza daripada sistem bersama.

**Dimensi dan alun ialah dua perkara berbeza.** D menentukan jumlah koordinat bebas; bentuk fungsi menentukan lembah, bukit, lebar dan penghalang. D besar sahaja tidak menjamin banyak minimum. \(x,x^2,\ldots,x^N\) ialah ciri kuasa bagi satu pemboleh ubah, bukan N koordinat bebas. Kita tidak menetapkan dimensi tertentu sebagai makna sungai atau kewangan.

Pengguna menjelaskan bahawa bayangan asal melibatkan seluruh kamus, tetapi menyerahkan pilihan untuk bermula dengan konteks ayat. Keputusan pelan:

- **Seluruh kamus:** semua perkataan mempunyai landscape dalam satu ruang koordinat yang dipelajari bersama.
- **Sistem aktif:** setiap kemunculan token dalam ayat/window mempunyai state tersendiri. Semua state dalam window berinteraksi melalui satu fungsi tenaga bersama.
- **Tanpa ayat:** bandingkan keserasian landscape asal menggunakan ukuran overlap; ini bukan equilibrium seluruh kamus.

Ini ialah pelaksanaan awal yang lebih kecil daripada menjadikan seluruh kamus sistem dinamik serentak. Ia membolehkan konteks mengaktifkan makna tertentu dan mengelakkan kos interaksi semua V perkataan. Dua kemunculan `bank` berkongsi parameter perkataan tetapi boleh mempunyai state berbeza. Dalam ayat kewangan, kita tidak menambah token `river` sebagai peserta hanya kerana ia wujud dalam kamus.

## 2. Jurang model semasa

| Kod semasa | Apa sebenarnya dikira | Perubahan v2 |
|---|---|---|
| `models/torx_semantic_ebm.py` | DFG `target → latent categorical → context`; marginal kebarangkalian dikira melalui logsumexp | Kekalkan sebagai baseline. Ia belum merupakan sistem continuous landscape yang berinteraksi bersama. |
| `landscape_ebm/general_energy.py`, `WordEnergy` | Dua embedding tables; energy `−dot − MLP + 0.025||x||²`; context ialah satu titik | Ganti kontrak utama dengan satu `energy(word,z)` untuk kedua-dua peranan, dan bacaan landscape–landscape. |
| `models/energy_metrics_torx.py` | Antaranya purata dua conditional NLL | Jangan gunakan nama atau tafsiran ini sebagai bukti free energy satu sistem equilibrium bersama. |

[Run skala terdahulu](landscape_ebm/scale_extended_results/REPORT.md) menemui calon minima `bank` kira-kira 43.7 unit daripada context embedding terdekat. Ini pemerhatian tempatan bagi model lama, bukan bukti kegagalan semua EBM. Ia menunjukkan perlunya menguji geometri di luar titik latihan serta kaitan minima dengan data. Menambah dimensi pada model lama tidak menyelesaikan pertukaran representasi yang diminta.

## 3. Landscape v2.0 yang boleh diperiksa

Mulakan dengan campuran Gaussian sebagai bentuk tenaga yang terkawal:

\[
p_w(z)=\sum_{r=1}^{M}\pi_{wr}\,\mathcal N(z;\mu_{wr},\operatorname{diag}(\sigma_{wr}^2)),
\qquad U_w(z)=-\log p_w(z).
\]

Gunakan `logsumexp`. \(\pi_w=\operatorname{softmax}(a_w)\); lokasi \(\mu\), skala \(\sigma\), dan logit \(a\) semuanya boleh dilatih. Batas norma lokasi dan julat varians positif ditetapkan dalam config supaya komponen tidak mengecil tanpa had atau lari jauh. Batas dipilih melalui validation dan dilaporkan, bukan dianggap jaminan semantik.

M ialah bilangan komponen/capacity, **bukan bilangan makna yang diketahui atau jaminan bilangan minima**. Komponen boleh bertindih, tidak digunakan, atau mewakili variasi selain makna. Komponen juga menggunakan vektor pusat secara dalaman: ini satu fungsi campuran, bukannya representasi yang bebas daripada vektor atau parameter berbilang pusat. Tiada label `river=-2`, `money=+2` diberi dalam latihan teks.

Cadangan pilot: D=16, M=4, covariance diagonal, tiga seed. Ini hyperparameter permulaan, bukan hasil optimum. Bandingkan M=1,2,4 dengan D tetap; kemudian D=16,32,64 dengan M tetap. Anggaran parameter per perkataan ialah M(2D+1); contohnya V=10,000, D=16, M=4 memberi 1,320,000 nombor sebelum W. Bilangan parameter, masa dan data mesti dilaporkan bersama apabila membandingkan dengan vektor tunggal.

### Bentuk lebih bebas selepas asas lulus

\[
U_w^{\mathrm{flex}}(z)=-\log p_w(z)+A\tanh f_\phi(e_w,z),\qquad A<\infty.
\]

Residual neural yang berkongsi weight boleh mengubah bentuk lembah dan bukit; pilihan Fourier features dengan frekuensi terkawal ialah ablation kemudian. Residual terbatas mengekalkan kebolehnormalan kerana asas Gaussian mengawal ekor. Namun density baharu memerlukan normalizer perkataan baharu, dan formula overlap analitik campuran tidak lagi tepat. Jadi ini **v2.1 selepas audit**, bukan tambahan percuma kepada pipeline v2.0. Jangan gunakan energy tidak dinormalisasi untuk membandingkan perkataan tanpa mengurus perbezaan offset/normalizer.

## 4. Bacaan lexical dan satu sistem aktif

### Bacaan asal tanpa konteks

\[
O(w,v)=\int p_w(z)p_v(z)\,dz
=\sum_{r,s}\pi_{wr}\pi_{vs}\,
\mathcal N(\mu_{wr};\mu_{vs},\Sigma_{wr}+\Sigma_{vs}),
\qquad C(w,v)=-\log O(w,v).
\]

Nilai C rendah menunjukkan overlap lebih besar dalam koordinat yang sama; ia bukan kebarangkalian semantik, boleh negatif, dan bergantung pada skala density. Covariance dibatasi dan sistem koordinat dikongsi. Ia ukuran simetri. Overlap memberi cara yang tepat dan murah untuk membandingkan v2.0 sebelum membina inference bersama. Ia tidak boleh dilabel sebagai hasil dari bikar bersama. `bank` berpotensi overlap dengan dua kelompok tanpa memerlukan `money` overlap besar dengan `river`.

### Tenaga untuk window bersama

Untuk L token aktif, setiap token mempunyai \(z_i\). Pilot menggunakan satu head:

\[
\mathcal E(z_{1:L}\mid w_{1:L})=
\sum_i U_{w_i}(z_i)
-\lambda\sum_{(i,j)\in G}a_{ij}\,k_W(z_i,z_j),
\]

\[
k_W(x,y)=\exp\left[-\frac{(x-y)^T W(x-y)}{2\ell^2}\right].
\]

W positif definit dibina daripada \(B^TB+\epsilon I\), kemudian dinormalisasi trace untuk mengawal skala; \(\ell>0\) ditetapkan dahulu. Untuk graf penuh L token, \(a_{ij}=1/(L-1)\) apabila L>1; L=1 tiada interaksi. \(\lambda=1\) dahulu, sebelum eksperimen kekuatan yang berasingan. Komponen interaksi terbatas untuk window terhingga, jadi normalizer joint wujud. Ini hipotesis interaksi tarikan yang paling mudah; jangan anggap ia memadai untuk sintaks atau hubungan berarah.

W menentukan cara state berinteraksi, bukan hubungan yang dicetak manual bagi setiap pasangan perkataan. Tensor parameter perkataan yang sama digunakan walaupun token dibaca sebagai Q atau K. Jangan memasukkan label relevan ke dalam graf G. Untuk pilot, semua token aktif berinteraksi; ablation graf jarak posisi boleh dibuat kemudian.

Gunakan unit suhu T=1 bagi versi pertama:

\[
P(z_{1:L}\mid w_{1:L})=Z(w_{1:L})^{-1}\exp[-\mathcal E].
\]

Ini taburan equilibrium yang ditakrifkan, bukan jaminan bahawa run sampler pendek sudah mencapainya. Jangan menggantikan semua \(z_i\) dengan satu z yang wajib sama; itu akan memaksa semua token berkongsi satu keadaan, dan boleh memadam perbezaan makna.

### Bacaan Q–K daripada sampel sistem yang sama

\[
R_{ij}=\mathbb E_{P(z_{1:L}\mid w_{1:L})}[k_W(z_i,z_j)],
\qquad E^{\mathrm{read}}_{ij}=-\log(R_{ij}+\varepsilon).
\]

Satu kumpulan sampel joint menghasilkan seluruh matriks R dalam window; **tiada equilibrium baharu bagi setiap pasangan Q–K**. R simetri dalam versi ini. Jika diperlukan, attention ialah penormalan baris \(A_{ij}=R_{ij}/\sum_{k\ne i}R_{ik}\), dengan self-mask. A boleh tidak simetri akibat denominator berbeza; itu tidak membuktikan hubungan berarah sudah dipelajari. \(E^{\mathrm{read}}\) ialah skor bacaan, bukan jumlah tenaga fizikal atau free energy seluruh sistem.

Landscape asal `bank` boleh mempunyai beberapa kawasan low-energy, sementara marginal `bank` dalam satu konteks mungkin menumpukan pada satu kawasan. Kita menilai kedua-duanya secara berasingan. Dalam konteks yang benar-benar mencampurkan dua makna, jangan menuntut satu state tunggal menyelesaikan semua hubungan serentak.

## 5. Training dan inference yang diuji berperingkat

**A. Pretraining lexical:** gunakan pasangan window positif dan negatif tersampel dengan objective ranking

\[
\mathcal L_{\mathrm{lex}}=\operatorname{softplus}(m+C(w,v^+)-C(w,v^-)).
\]

Ini fixed-negative contrastive ranking, bukan likelihood/NCE terkalibrasi. Negatif menahan collapse semua perkataan pada landscape sama; parameter bounds menahan varians hampir sifar. Monitor komponen mati/bercantum, false negatives dan kata kerap. Jangan memaksa setiap perkataan mempunyai empat makna. Default negatives unigram^0.75 daripada train, dengan collision positif tepat dikecualikan; negatif masih boleh berkaitan semantik. Subsampling dan definisi window train/eval ditetapkan serta dilaporkan konsisten.

**B. Joint training kecil:** untuk vocabulary kecil dan L=3, gunakan grid/quadrature untuk mengira

\[
F(x)=-\log\int\exp[-\mathcal E(z\mid x)]dz.
\]

Train `softplus(m+F(x_pos)-F(x_corrupt))` bersama lexical loss. Positif dan korup mesti mempunyai panjang serta graf sama. Ubah satu token bagi corruption; integrasi melibatkan seluruh sistem dalam kedua-dua konfigurasi, bukan sekadar menilai token baharu pada equilibrium lama. Gradien melalui jumlah grid boleh diperiksa secara tepat untuk model grid; kesilapan truncation/quadrature mesti diperiksa berasingan. Ini global ranking terhadap konfigurasi teks, bukan MLE bahasa penuh. Bila λ=0, unary density yang dinormalisasi memberi F=0: ujian penting untuk membezakan pengaruh interaksi daripada offset token.

**C. Sampler bersama:** cadangkan pertukaran satu state token menggunakan proposal \(z_i'\sim p_{w_i}\). Metropolis–Hastings menerima dengan `min(1, exp(−ΔE_interaction))` kerana unary dan proposal saling membatalkan. Setiap langkah tetap mengemas kini sistem joint yang sama. Proposal boleh melompat antara komponen, namun mixing masih mesti diukur. Audit melawan grid, beberapa chains, occupancy, ESS dan perbezaan run pendek/panjang. Marginal uncertainty dikekalkan; hanya mengambil satu argmin bukan bacaan default.

**D. Scaling joint:** pretraining lexical boleh dibesarkan dahulu. Joint training pada D besar memerlukan anggaran F dan gradien yang diuji. Importance sampling dengan proposal produk unary ialah pilihan untuk window pendek; log-anggaran Z biased pada sampel terhingga, mixture sampling membawa isu gradien pilihan komponen, dan varians boleh meningkat. Audit terhadap grid sebelum digunakan. Jangan `autograd` melalui pilihan diskret sampler dan anggap gradien sah. Jika estimator gagal, hentikan scaling joint; laporkan lexical model sahaja sebagai komponen yang siap, bukan seluruh seni bina.

## 6. Eksperimen yang menjawab idea pengguna

| Tahap | Hasil yang perlu ditunjukkan sebelum lanjut |
|---|---|
| 1: matematik, D=1–2 | Overlap analitik sepadan quadrature; density ternormal; penambahan konstanta energy tidak menukar taburan ternormal; gradien finite difference; grid diperhalus sehingga stabil. |
| 2: corpus sintetik berkonteks | Perkataan ambigu dilatih daripada dua keluarga konteks tanpa memberi koordinat makna. Nilai kedua-dua sense, bukan hanya yang dominan. Bezakan demonstrasi bentuk yang ditetapkan manual daripada makna yang dipelajari. |
| 3: sistem aktif L=3 dahulu | Dalam konteks sungai, marginal `bank` beralih ke kelompok sungai; dalam konteks kewangan ia beralih ke kelompok kewangan. R diperoleh daripada sampel joint sama. Banding λ=0 dan λ=1. |
| 4: lexical text8 | Baseline SGNS, single-Gaussian dan multi-Gaussian pada split/data/compute setara; tiga seed. Tiada dakwaan two-sense tanpa contoh konteks yang diuji. |
| 5: joint lebih besar dan residual neural | Hanya jika estimator, mixing dan konteks menunjukkan manfaat. Tambah D atau kapasiti bentuk satu pemboleh ubah pada satu masa. |
| 6: Torx | Diskretkan toy dan bandingkan energy/taburan/bacaan dengan rujukan asal; laporkan ralat pemetaan serta kos. |

Metrik utama toy ialah ketepatan pemilihan sense pada konteks ditahan dan retrieval per sense; bukan bilangan minima atau graf cantik. Tiga seed, template train/validation/test berasingan, tiada token label sense yang diberi pada inference. Benchmark teks sebenar memerlukan split dokumen atau standard yang belum digunakan untuk tuning. Test text8 dan analogi yang telah dilihat sebelum ini dianggap development/exploratory. Bekukan protokol sebelum membuka holdout baharu.

Ablation perlu membezakan kapasiti daripada dynamics: SGNS tied/untied tables; satu Gaussian; mixture lexical; mixture dengan sistem bersama λ=0/λ>0; kemudian residual neural. Laporkan jumlah parameter, masa, konteks yang tersedia dan corruption sama. Model vektor juga perlu baseline berkonteks untuk dakwaan manfaat konteks yang adil. Kejayaan synthetic sahaja tidak membuktikan semantik teks sebenar.

## 7. Pemetaan kerja dalam repo

Tambah modul baharu di `word2energy/energy_embedding/`:

- `landscape.py`: parameter perkataan, `energy(word,z)`, `log_density`, proposal unary.
- `overlap.py`: formula analitik dan reference quadrature.
- `system.py`: global energy, W, graf dan pembinaan window tanpa label bocor.
- `reference.py`: grid kecil, F, gradien dan bacaan joint.
- `sampler.py`: update MH joint, diagnostik dan reproducible seeds.
- `readout.py`: matriks R daripada sampel yang sama; bezakan lexical dan contextual API.
- `train_lexical.py`, `train_joint_toy.py`: objective serta checkpoint dengan config.
- `evaluate.py`: sense/context, retrieval, minima, controls dan run manifest.

Kekalkan kod dan checkpoint lama sebagai baseline, bukan menulis ganti sejarah. Data loader text8 boleh digunakan semula selepas split/protokol disemak. Checkpoint lama dua embedding tables tidak terus serasi dengan landscape baharu; sebarang pemindahan parameter mesti dinamakan sebagai initialization ablation.

Torx lama ialah directed categorical model. Global joint v2 ialah objek berbeza. Untuk reference berterusan gunakan PyTorch dahulu, kemudian kontrak energy yang sama diuji pada JAX/Torx. Memilih K keadaan pdit atau bilangan pbit ialah keputusan representasi/ketepatan, bukan `D = bilangan pbit`. State-vector exact bertambah eksponen. Grid 2D atau Gaussian-mixture sampler yang dijalankan dalam perisian belum membuktikan implementasi cip. Pemetaan arbitrary neural potential kepada Ising tidak automatik.

Pelaksanaan pertama yang dicadangkan: **landscape + overlap + corpus synthetic + global reference kecil**. Had pilot tempatan: tiga seed dan maksimum 30 minit CPU untuk keseluruhan validation awal; semak runtime smoke sebelum memperbesar. Tiada cloud atau run text8 besar termasuk dalam dokumen pelan ini. Jika ujian asas gagal, simpan keputusan dan baiki sebabnya dahulu.

## 8. Sumber dan batas tuntutan

- **Teks pengguna:** [transkrip](TEKS_ASAL_BIKAR_ENERGY_EMBEDDING.txt); sumber maksud satu sistem bersama dan Q/K sejenis.
- **Kajian sedia ada:** [Athiwaratkun & Wilson, Multimodal Word Distributions, §3](https://aclanthology.org/P17-1151/) memperkenalkan word Gaussian mixtures, overlap dan objective ranking. Oleh itu asas mixture embedding bukan tuntutan kebaharuan kita. Sistem joint aktif dan bacaan di atas ialah cadangan untuk diuji, bukan hasil yang paper itu membuktikan bagi model kita.
- **Latar:** [Vilnis & McCallum, Word Representations via Gaussian Embedding](https://arxiv.org/abs/1412.6623) menggunakan density sebagai representasi perkataan.
- **Training EBM:** [Song & Kingma, How to Train Your Energy-Based Models, §2–5](https://arxiv.org/html/2101.03288v2) membezakan energy, normalizer dan kaedah latihan. Formulasi ranking di sini tidak didakwa sebagai MLE daripada paper itu.
- **Word2Vec:** [Mikolov et al., §2](https://arxiv.org/html/1310.4546) untuk skip-gram/window dan negative sampling; seluruh vocabulary disimpan tidak bermaksud setiap perkataan berinteraksi pada setiap langkah.

Belum ada eksperimen v2 dijalankan. Pelan menetapkan hipotesis, model awal, ukuran bacaan, laluan latihan dan syarat sebelum scaling; ia tidak menyimpulkan bahawa landscape berbilang lembah semestinya mengatasi embedding vektor.
