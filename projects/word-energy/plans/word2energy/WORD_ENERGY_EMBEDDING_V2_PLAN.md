# Pelan Word Energy Embedding v2

11 November 2026. Status: reka bentuk untuk pelaksanaan; belum dilatih atau divalidasi. **Keutamaan telah dikemas kini:** gunakan [pelan Ising/Z1 dahulu](WORD_ENERGY_Z1_FIRST_PLAN.md) sebagai laluan utama; reka bentuk Gaussian continuous di sini ialah baseline. Skor critic terdahulu tidak mengesahkan pelan Z1 baharu. Pelan ini berasaskan [teks asal](TEKS_ASAL_BIKAR_ENERGY_EMBEDDING.txt), [rumusan yang disemak](RUMUSAN_IDEA_BIKAR_ENERGY_EMBEDDING.md), kod tempatan dan sumber primer di bawah. Formulasi baharu dalam dokumen ini ialah cadangan reka bentuk, bukan formula yang pengguna telah berikan.

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

Pilot lexical teks pada peringkat kemudian: D=16, M=4, covariance diagonal, tiga seed. Pilot joint pertama jauh lebih kecil: D=1, M=2, L=3. Ini hyperparameter permulaan, bukan hasil optimum. Bandingkan M=1,2,4 dengan D tetap; kemudian D=16,32,64 dengan M tetap. Anggaran parameter per perkataan ialah M(2D+1); contohnya V=10,000, D=16, M=4 memberi 1,320,000 nombor sebelum W. Bilangan parameter, masa dan data mesti dilaporkan bersama apabila membandingkan dengan vektor tunggal.

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

W positif definit ditetapkan sebagai \(W=(1-\rho)D B^TB/\operatorname{tr}(B^TB)+\rho I\), dengan \(\rho=0.05\), B bukan sifar, dan \(\operatorname{tr}W=D\). Berhenti dengan ralat jika norma B mendekati sifar; jangan membahagi dengan sifar. Ini menetapkan batas eigen minimum \(\rho\). \(\ell=1\) pada toy. Untuk graf penuh L token, \(a_{ij}=1/(L-1)\) apabila L>1; setiap edge i<j dikira sekali, dan L=1 tiada interaksi. \(\lambda=1\) dahulu, sebelum eksperimen kekuatan yang berasingan. Komponen interaksi terbatas untuk window terhingga, jadi normalizer joint wujud. Ini hipotesis interaksi tarikan yang paling mudah; jangan anggap ia memadai untuk sintaks atau hubungan berarah.

**Had yang boleh diterbitkan:** \(F(x)\in[-\lambda L/2,0]\). Perubahan log-odds antara dua kawasan bagi satu token akibat konteks paling besar ialah \(\lambda\), kerana jumlah berat edge yang menyentuh token itu ialah 1. Untuk dua kawasan yang bersama-sama meliputi ruang dan mempunyai prior mass 0.5/0.5, posterior occupancy sesuatu kawasan tidak melebihi \(\operatorname{sigmoid}(\lambda)\); bagi λ=1 batasnya sekitar 0.731. Ini bukan batas accuracy klasifikasi argmax. Margin dan sasaran occupancy mesti menghormati batas ini. Dalam D=1, W=1 selepas normalisasi, jadi ujian 1D tidak boleh menunjukkan pembelajaran W; ujian anisotropi D=2,L=2 diperlukan secara berasingan.

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

Ini fixed-negative contrastive ranking, bukan likelihood/NCE terkalibrasi. Negatives menghukum collapse dari segi objective tetapi tidak menjamin optimizer mengelakkannya. Semua means yang sama boleh menjadi keadaan stationary walaupun loss buruk. Parameter bounds menahan varians hampir sifar. Monitor komponen mati/bercantum, false negatives dan kata kerap; initialization means bebas bagi setiap token/komponen diperlukan. Jangan memaksa setiap perkataan mempunyai empat makna. Default negatives teks sebenar unigram^0.75 daripada train, dengan collision positif tepat dikecualikan; negatif masih boleh berkaitan semantik. Subsampling dan definisi window train/eval ditetapkan serta dilaporkan konsisten. Negative rules synthetic ditentukan berasingan dalam kontrak pilot di bawah.

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

Pelaksanaan pertama yang dicadangkan: **landscape + overlap + corpus synthetic + global reference kecil**. Had pilot tempatan ialah 30 minit masa berlalu, maksimum empat CPU threads dan working set sasaran 2 GiB. Ini had kerja, bukan jaminan semua checks selesai. Semak runtime smoke sebelum memperbesar. Tiada cloud atau run text8 besar termasuk dalam dokumen pelan ini. Jika ujian asas gagal, simpan keputusan dan baiki sebabnya dahulu. Kontrak operasional berikut mengatasi cadangan umum jika terdapat perbezaan detail.

## 8. Kontrak pilot yang dibekukan sebelum run

### 8.1 Config, training dan bajet

| Tetapan | Nilai pilot pertama |
|---|---|
| Reference utama | D=1, M=2, L=3; W=1 tetap; float64 |
| Audit W berasingan | D=2, M=1, L=2; B awal `[[1,.2],[.1,.7]]`; mu1=(-1,-.5), mu2=(.8,1.2), diagonal variance1=(.3,.8), variance2=(.7,.2); λ=ell=1. Tiada syarat noncollinearity untuk dua means. |
| Parameter unary | Means diprojeksi ke bola norma ≤3 selepas optimizer; varians `.16 + 1.28*sigmoid(raw_variance)`; logits mixture softmax |
| Initialization | Means bebas `Normal(0,.8²)` kemudian projection; varians awal .36; logits 0. Tiada koordinat ditetapkan menggunakan label sense. |
| Interaksi | λ=1, ell=1, rho=.05, trace W=D; tolak B jika `||B||F<1e-8` |
| Seeds model | 17, 29, 43; data/split tetap sama untuk semua model |
| Lexical loss | Margin 1; lima negatives per positif; mean atas negatives dan batch |
| Lexical optimizer | Adam lr=.01, 400 updates, batch64 pasangan train disampel dengan penggantian, clip grad norm5 |
| Joint loss | `mean softplus(.25+F_pos−F_corrupt) + .25*L_lex`; dua corruptions bagi setiap window |
| Joint optimizer | Adam baru lr=.003, 100 updates, batch4 windows train dengan penggantian; clip5; bermula dari checkpoint lexical dipilih |
| Validation | Setiap20 updates; lexical pilih macro-AUC validation tertinggi, tie loss lexical terendah; joint pilih macro sense accuracy tertinggi, tie loss joint terendah; tie akhir pilih checkpoint lebih awal |
| Tuning pertama | Tiada sweep bounds/lr/seed; λ=2,4 ialah kajian susulan yang ditandakan baharu, hanya selepas pilot λ=1 disimpulkan |
| Termination | Stop pada bajet updates atau had masa; NaN/Inf berhenti serta simpan diagnostic. Tiada claim convergence hanya kerana updates habis. |

Jangan gunakan test untuk pilihan hyperparameter atau checkpoint. Target posterior occupancy >0.731 tidak digunakan bagi λ=1 dengan prior dua sense seimbang. Jika kajian susulan mensasarkan occupancy≥.9, λ≥log9 hanyalah syarat perlu, bukan jaminan; λ lebih besar turut mengubah mixing dan concentration. Margin joint .25 berada di bawah jurang maksimum1.5 untuk L=3,λ=1.

Batch/negative schedule dikongsi semua model dan seeds: RNG4040 menghasilkan400 batch lexical ×64 dari senarai pasangan train canonical; RNG5050 menghasilkan100 batch joint ×4 windows. RNG6060 menghasilkan lima lexical negatives per occurrence dalam schedule menggunakan rules synthetic §8.2. Untuk loss validation dan tie-break, pasangan val canonical serta negatives tetap RNG7070 digunakan. Fail schedule disimpan sekali dengan hash; modelseed hanya mengawal initialization, bukan contoh latihan. Semasa joint training, gunakan batch lexical pertama100 dalam schedule yang sama sebagai auxiliary loss. Model vector bermula dengan setiap parameter `Normal(0,.1²)` daripada modelseed; initialization ini berbeza daripada mixture dan dinyatakan dalam perbandingan.

### 8.2 Corpus dan split synthetic

Vocabulary pilot 26 token: `bank`, `song`, dan tiga pool lapan anchor berikut:

- F: money, loan, credit, deposit, finance, lender, payment, cash.
- R: river, stream, shore, flood, water, creek, sediment, current.
- M: melody, rhythm, singer, guitar, piano, chorus, concert, lyric.

Grammar toy menghasilkan `(bank,F_i,F_j)`, `(bank,R_i,R_j)`, `(song,M_i,M_j)` untuk i<j. Ada 28 unordered anchor pairs per pool, 84 contoh keseluruhan. **Ini data buatan dengan maksud yang ditetapkan oleh generator**, bukan bukti pembelajaran bahasa daripada korpus semula jadi. Setiap contoh memberi semua enam pasangan berarah antara tiga token untuk lexical training. Model menerima IDs token sahaja, tiada label family atau koordinat makna.

Bahagikan indeks 28 pasangan menggunakan permutation `numpy.random.default_rng(1730)`: 16 train,6 validation,6 test. Tolak permutation dan ambil yang seterusnya daripada RNG sama jika mana-mana anchor muncul kurang dua kali dalam train. Gunakan pembahagian indeks sama untuk ketiga-tiga pool: 48 train,18 val,18 test; dua sense bank seimbang. Simpan pair lists dan hash sebelum training. Sorting bag-of-token memastikan tiada window val/test ialah permutasi window train. Semua anchor muncul dalam train: ini ujian komposisi konteks pada vocab dikenali, bukan generalisasi kata baharu atau pasangan bank–anchor yang belum pernah dilihat.

Corruption joint: bagi setiap window, hasilkan dua negatif tetap, masing-masing mengganti slot anchor1 atau anchor2 dengan anchor daripada pool lain yang dipilih uniform; RNG202, urutan canonical mengikut pool dan pasangan. Tiada token baru ditambah ke input positif. Grammar toy menganggap satu domain per window; ia tidak menyatakan ayat manusia yang mencampurkan domain sentiasa salah. Semua model menggunakan fail corruption yang sama.

Lexical negatives synthetic: unigram^.75 train, K=5; tolak self dan pasangan yang sah menurut **grammar generator**, bukan menurut hasil model/test yang diperhatikan. Nyatakan bantuan generator ini dalam laporan; ia ialah constructed-negative task. Untuk natural text tiada oracle sedemikian. Tiada subsampling pada toy. Laporkan histogram frekuensi, collision dan coverage. Frequency baselines wajib kerana bank muncul dua kali lebih kerap daripada song.

### 8.3 Reference numerik dan gauge

Reference utama menggunakan tensor Gauss–Hermite terhadap expectation produk Gaussian, dengan semua assignment komponen dijumlahkan, bukan sample pilihan komponen. Bagi setiap component, transform `z=mu+sqrt(2)*sigma*node`, dan normalized quadrature weights `weight/sqrt(pi)` per koordinat. Maka `Z=E_product_unary[exp(lambda*S)]`; tiada tail truncation pada reference ini. Ia tepat untuk model quadrature, bukan integral continuous yang tepat pada n terhingga. Autodiff mengambil kira lokasi node, variance dan mixture weights.

Untuk D=1,L=3,M=2: n=8 per koordinat/component ialah 4,096 joint terms/window. Semak n=12,16 dan sehingga32 jika perlu. Untuk D=2,L=2,M=1: n=8 ialah 4,096 terms, n=16 ialah65,536. Jangan membuat grid D=2,L=3 dalam pilot. Proses window/terms berkelompok, maksimum65,536 joint terms per chunk; smoke memeriksa peak working set dan runtime. Training menggunakan n8 hanya jika gradient/refinement gate lulus pada initialization dan checkpoint setiap20updates; jika gagal cuba n12/16 dalam bajet yang sama. Jika tiada n cukup tepat dalam bajet, status **tidak konklusif: reference belum cukup tepat**, bukan gagal seni bina.

Jika refinement gagal selepas satu blok20updates, rollback model, optimizer dan RNG ke checkpoint terakhir yang lulus; naikkan n dan replay batch/negative IDs sama bagi blok itu. Semua kerja berulang tetap dikira dalam had masa. Resolusi reference32 untuk audit tidak bermaksud training32 dibenarkan secara automatik; jika training16 masih tidak tepat, hentikan pilot sebagai tidak konklusif. Rekod trajectory resolusi dan jangan menyebut gradient training quadrature sebagai gradient integral continuous yang tepat.

Cross-check bebas: density/overlap 1D dengan Simpson domain[-12,12], n513 kemudian1025; ulang domain[-15,15] pada spacing serupa untuk membezakan ralat domain dan spacing. Bounds mean3 dan sd≤1.2 membolehkan semakan Gaussian tail secara analitik. Semak F/R joint GH dengan Simpson berketul untuk satu konfigurasi D1,L3 jika bajet memadai; sehingga dilakukan, nyatakan bahawa joint reference mempunyai convergence check tetapi belum cross-check integral bebas. Untuk indicator basin yang tidak licin, jangan menganggap GH8 mencukupi: gunakan marginal 1D, cari boundary qF=qR dan integrasikan interval berasingan dengan adaptive quadrature.

Training F mesti menggunakan unary ternormal `−log p_w`. Jika raw unary ditambah c_w, P dan R tidak berubah selepas normalisasi, tetapi raw F bertambah `sum_i c_wi`; ranking F merentas teks tidak invariant kepada offset perkataan. Ujian gauge menyemak kedua-dua fakta, dan generic energy API mesti mengembalikan normalizer unary sebelum digunakan untuk ranking. Label component dan translation global tidak identifiable; jangan tafsir component0 atau paksi tertentu sebagai sense. Bounds parameter memilih satu koordinat yang praktikal, bukan semantik kanonik.

### 8.4 Metrik operasional dan baseline

**Sense region:** untuk setiap checkpoint yang dinilai pada validation, bentuk qF dan qR sebagai purata density anchor train masing-masing daripada checkpoint itu menggunakan algoritma yang tetap. Region `AF={z:qF(z)>qR(z)}` dan `AR` pelengkapnya; kes sama diberi separuh mass dalam pengukuran. Map menggunakan label generator pada train untuk evaluation sahaja, bukan input model/objektif atau penetapan μ. Region tidak bergantung pada index komponen dan bergerak bersama geometri jika seluruh model diterjemah. Selepas checkpoint dipilih menggunakan validation, **bekukan checkpoint dan map sebelum test**. Jangan memilih region/threshold berdasarkan test. Untuk ablation λ0/λ1, gunakan checkpoint dan map sama supaya perubahan occupancy benar-benar daripada dynamics.

Untuk bank dalam setiap window, ukur `P(z_bank in A_true | window)`, prior mass `P_bank(A_true)` pada checkpoint sama dan pertambahan mass. Classification memilih region dengan mass terbesar; tie=0.5 credit. Primary metric ialah macro accuracy pada12 bank test windows (6 F+6 R) per seed. Laporkan count tepat, confusion matrix, semua seed dan CI Wilson deskriptif atas12 contoh setiap seed; jangan pool seeds seolah-olah mereka36 dokumen bebas. Occupancy dan accuracy dilaporkan berasingan.

**Lexical retrieval:** query bank, candidates24 anchor, exclude bank/song; score `−C`. Lapor pairwise AUC F versus M dan R versus M, macro purata; ties0.5. Ini menilai hubungan vocab dikenali, bukan contextual retrieval. Untuk checkpoint lexical selection, positives ialah anchor occurrences daripada val windows, negatives pool M dengan kekerapan seimbang; jelaskan bahawa val combinations baharu tidak menjadikan setiap bank–anchor association baharu.

**Active-window R:** ukur ketiga-tiga edges daripada sampel joint yang sama, bersama uncertainty; uji perubahan apabila konteks ditukar. Jangan dakwa retrieval seluruh kamus daripada R: kandidat luar belum aktif dan tiada scorer contextual external-candidate dalam pilot ini.

Baselines: frequency-only; lexical-only mixture yang sama; same checkpoint dengan λ=0 dan λ=1 tanpa retraining (asingkan manfaat dynamics); M1 dan M2 dengan D sama (kapasiti berbeza dilapor); SGNS tied D6 dan untied D3 memberi156 embedding parameters, sama dengan26*2*(2*1+1) untuk mixture toy. Untuk contextual vector baseline, gunakan `h=v_bank+mean(v_context)` dan cosine kepada centroid F/R anchor train; tiada classifier tambahan atau labels pada inference. Ia menerima context sama dan boleh mendapat accuracy tinggi pada toy mudah ini. Tetapan Adam/lr/batch/update budget sama seperti lexical stage, negative files sama; skor yang berbeza skala dilaporkan dengan objective masing-masing dan tiada tuning tambahan tersembunyi.

Kontrak vector: `s(w,v)=u_w^T v_v`; tied bermaksud table u=v, untied dua table berasingan. Loss lexical SGNS ialah `mean_batch[softplus(−s_pos)+sum_5neg softplus(s_neg)]`. Untuk contextual readout dan centroid, **semua token menggunakan input table u**, termasuk bank dan context; output table untied digunakan hanya untuk lexical training score. Lexical retrieval menggunakan s(w,v) secara berarah dan candidate set yang sama. Zero-norm h/centroid memberi cosine0 dan tie0.5. Checkpoint vector dipilih dengan rule val lexical yang sama seperti mixture; tiada joint adaptation vector dalam pilot, maka perbandingan joint-trained mixture lawan vector tidak mengasingkan architecture/training objective. Claim dynamics khusus menggunakan ablation checkpoint mixture sama.

Untuk audit identifiability joint, tambah baseline **frozen lexical geometry** (freeze semua unary, hanya inference λ) sebelum joint adaptation. Check permutation komponen dan translation bersama pada model serta map tidak mengubah metrik; lakukan translation dalam bounds parameter untuk check ini. Semua map berasal dari identiti anchor train yang sama dan algoritma tetap, bukannya pemilihan lembah paling menguntungkan selepas melihat test.

### 8.5 Gates dan controls sebelum scaling

Gates berikut dipilih sebelum run, bukan dianggap hukum universal:

| Check | Lulus |
|---|---|
| Normalisasi unary dan overlap 1D | Integral error≤1e-6; analytic vs quadrature overlap rel≤1e-5 atau abs≤1e-8 |
| Autodiff vs central difference | Step1e-5 pada raw parameters interior; abs≤1e-6 atau rel≤1e-4; uji means, variance, logits, B; sekurang-kurangnya satu gradien B bukan sifar dalam D2 audit |
| Joint refinement | Max perubahan F/R≤1e-4 dan gradient norm difference≤1e-3 relatif (floor1e-6) antara dua resolusi tertinggi; check init dan setiap checkpoint |
| Marginal region integration | Normalisasi/mass error≤1e-4 pada refinement; integrasi boundary sepadan dalam1e-4 |
| Gauge dan λ=0 | P/R unchanged≤1e-8 untuk exact quadrature selepas offsets; raw F shift=sum offsets≤1e-8; F(λ0)=0≤1e-10 dalam GH expectation |
| Sampler audit | Empirical R dan basin mass dalam `max(.02,3*MCSE)` daripada reference; MCSE≤.01; rank-normalized split R-hat<1.05 dan multi-chain ESS≥400 bagi basin indicators dan kernel scores |
| Toy semantic gate | Setiap seed macro accuracy≥10/12, macro lexical AUC≥.75, dan mean correct-region posterior gain (λ1−λ0 pada checkpoint sama)≥.03. Laporkan kedua-dua senses dan variability. |

Sampler schedule: empat dispersed chains bermula pada komponen berbeza, satu sweep mengunjungi semua token dalam permutation rawak; burn-in500 sweeps kemudian2,000 sweeps direkod tanpa thinning. Semua R edges menggunakan sampel joint tersebut. Jika precision gagal, double burn-in/production maksimum dua kali jika bajet cukup. Indicator yang constant kerana chain terperangkap tidak mendapat ESS tak terhingga; diagnosis gagal atau tidak konklusif bergantung reference/precision. Sampler MCSE bukan variability antara model seeds.

Gate semantik hanya menunjukkan toy sanity, **bukan model lebih baik daripada contextual vector baseline**. Claim improvement memerlukan primary metric pradaftar pada data lebih besar, paired uncertainty yang mengambil kira unit data, dan CI improvement di atas sifar; jika vector baseline100%, tiada dakwaan improvement daripada toy ini. Numerical gate lulus tetapi semantic gate gagal bermaksud hipotesis/kapasiti/training belum mencapai kriteria pada bajet itu. Precision atau runtime tidak cukup bermaksud tidak konklusif. Jika gagal, simpan keputusan sebelum ubah config atau dataset; jangan ubah threshold untuk menukar label gagal kepada lulus.

Collapse diagnostics: std pair scores, inter-word mean spread, within-word pairwise centre distances, normalized mixture entropy/effective count `exp(H(pi))`, occupancy setiap component, serta pecahan R>.99. Flag semua means spread<1e-3 bersama score std<1e-3; flag component mean occupancy<.01 sebagai unused, bukan automatik kegagalan kerana bilangan sense tidak diketahui. Run satu diagnostic bermula semua means sama untuk menunjukkan ranking tidak menjamin escape; ia tidak perlu berjaya.

**Control sumbangan positive co-occurrence sahaja:** dalam tahap lexical, permute kolum context pada senarai pasangan train canonical menggunakan RNG8080; target dan counts marginal kekal. Negatives grammar-filtered asal serta joint windows/corruptions asal dikekalkan. Rule yang sama untuk model/budget; mula daripada initialization asal, bukan checkpoint terlatih. Ini sengaja mengekalkan negative supervision dan joint supervision yang mengandungi family information. Oleh itu, jika prestasi tidak turun, ia **bukan** bukti leakage atau kegagalan validiti seluruh pipeline; ia hanya menunjukkan sumbangan tambahan lexical positives belum ditunjukkan. Jangan menamakannya control tanpa maklumat semantik.

Tambahkan baseline model rawak sebelum training (seeds sama), dengan algoritma sense-map dan contextual readout yang sama. Ini mengukur berapa banyak signal datang daripada geometri rawak, context anchors dan evaluation readout. Jika trained model tidak mengatasi baseline rawak pada metrik sasaran, laporkan bahawa manfaat training belum terbukti; jangan menuntut penemuan semantik hanya kerana posterior boleh digerakkan oleh context. Semua labels generator yang digunakan untuk negatives atau evaluation mapping didedahkan.

### 8.6 Artifacts dan pintu masuk tahap seterusnya

Setiap run menyimpan config lengkap, versi dependencies, source snapshot/hash, split/corruption hashes, device, threads, seeds, checkpoints lexical/joint, losses, reference convergence, semua gate, runtime/memori, sampler MCSE dan task metrics. Tempoh30min dikongsi semua pilot/checks; prioritikan reference, satu seed smoke, kemudian tiga seed dan controls. Yang tidak siap dinyatakan jelas dan menghalang claim lulus penuh. Rerun pengesahan hanya apabila perubahan/failure membenarkannya.

Sebelum real-text confirmatory: wajib isi sumber/version dataset, license yang sesuai, unit dokumen, deduplication train/val/test, train-only vocab, preprocessing serta candidate protocol dan declaration test belum dibuka. Character-region text8 lama tidak menyediakan document split; ia kekal development. Semua ini gate masuk, bukan syarat untuk toy synthetic yang sudah ditakrifkan.

Kos v2.0: overlap satu pasangan O(M²D); retrieval lexical V kandidat O(VM²D); proposal unary O(D) selepas component dipilih, evaluasi unary O(MD). Kernel W dense naif O(D²), bukan O(D); precompute transformed positions menggunakan factor W memberi O(LD²+L²D) per sweep (termasuk state baru), ditambah proposal/unary costs. Grid seragam O(n^(DL)); mixture GH O(M^L n^(DL)). Joint importance scaling boleh enumerasi M^L assignments sebelum Gaussian MC untuk mengelakkan gradien kategori, tetapi masih mempunyai log-Z bias/variance dan perlu audit.

Torx pertama hanya reference diskret D1,L2 dengan7 state per token,49 joint states. Gunakan cell weights positif yang diketahui, energy grid dan conditional transition law; compare exact normalized enumeration, selected marginals/R dan transition invariance. Bezakan grid discretization error daripada sampler error. Runtime/gradient parity mesti direkod; tiada claim speedup atau chip mapping. Pbit encoding 7 state memerlukan peraturan explicit untuk state kelapan yang tidak sah; pdit7 mengelakkannya dalam reference tetapi tidak membuktikan perkakasan menyokongnya. Stage ini tidak dijalankan sebelum core gates lulus.

## 9. Sumber dan batas tuntutan

- **Teks pengguna:** [transkrip](TEKS_ASAL_BIKAR_ENERGY_EMBEDDING.txt); sumber maksud satu sistem bersama dan Q/K sejenis.
- **Kajian sedia ada:** [Athiwaratkun & Wilson, Multimodal Word Distributions, §3](https://aclanthology.org/P17-1151/) memperkenalkan word Gaussian mixtures, overlap dan objective ranking. Oleh itu asas mixture embedding bukan tuntutan kebaharuan kita. Sistem joint aktif dan bacaan di atas ialah cadangan untuk diuji, bukan hasil yang paper itu membuktikan bagi model kita.
- **Latar:** [Vilnis & McCallum, Word Representations via Gaussian Embedding](https://arxiv.org/abs/1412.6623) menggunakan density sebagai representasi perkataan.
- **Training EBM:** [Song & Kingma, How to Train Your Energy-Based Models, §2–5](https://arxiv.org/html/2101.03288v2) membezakan energy, normalizer dan kaedah latihan. Formulasi ranking di sini tidak didakwa sebagai MLE daripada paper itu.
- **Word2Vec:** [Mikolov et al., §2](https://arxiv.org/html/1310.4546) untuk skip-gram/window dan negative sampling; seluruh vocabulary disimpan tidak bermaksud setiap perkataan berinteraksi pada setiap langkah.

Belum ada eksperimen v2 dijalankan. Pelan menetapkan hipotesis, model awal, ukuran bacaan, laluan latihan dan syarat sebelum scaling; ia tidak menyimpulkan bahawa landscape berbilang lembah semestinya mengatasi embedding vektor.
