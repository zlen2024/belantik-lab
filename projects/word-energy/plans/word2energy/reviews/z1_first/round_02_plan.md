# Word Energy Embedding: laluan Ising / Z1 dahulu

11 November 2026. Keutamaan pengguna: eksperimen mesti mempunyai laluan yang munasabah menuju thermodynamic chip seperti Z1. Dokumen ini mengubah urutan [pelan v2 sebelumnya](WORD_ENERGY_EMBEDDING_V2_PLAN.md): **Ising/p-bit menjadi laluan utama**, Gaussian continuous menjadi baseline rujukan. Skor critic 9.6 terdahulu ialah untuk pelan lama, bukan pengesahan bagi perubahan ini.

## Apa yang diketahui tentang sasaran

[Extropic menerangkan](https://extropic.ai/writing/z1t/) Z1 sebagai programmable Ising sampler dengan graf fizikal tetap, darjah 16 bagi setiap p-bit dan chromatic Gibbs sampling. Bukan setiap graf matematik berdegree≤16 semestinya subgraf wiring fizikal tersebut. Ketersediaan SDK, akses hardware, graph adjacency yang boleh diprogram, bias/coupling ranges, resolusi, precision dan readout bagi projek ini masih belum disahkan. [Laman hardware Extropic](https://extropic.ai/hardware) menyenaraikan early access Z1 stick/card pada 2027; ia tidak membuktikan akses kita sendiri.

Bezakan empat tahap: (1) fungsi Ising secara matematik, (2) disimulasikan dalam THRML/Torx pada CPU/GPU, (3) embedded/compiled pada graf dan parameter hardware sebenar, (4) benar-benar dieksekusi dan diukur pada chip. Tahap 1–2 menyokong feasibility algoritma sahaja. Sama ada operasi loading, clamping dan parameter updating di bawah benar-benar boleh dilaksanakan pada chip memerlukan semakan API/backend sebenar.

## Reka bentuk: landscape setiap perkataan, satu sistem bersama

Perkataan mempunyai parameter spin yang sama apabila digunakan sebagai Q atau K. Bagi kemunculan token i, \(s_i\in\{-1,+1\}^d\), dengan kemungkinan spin tersembunyi \(h_i\). Satu window aktif menggunakan satu joint energy:

\[
E(s,h\mid w_{1:L})=-\sum_i\left[b_{w_i}^{T}s_i+\sum_{(a,b)\in H}J_{w_i,ab}s_{ia}s_{ib}+\text{coupling tersembunyi}\right]
-\sum_{(i,j)\in G}\sum_{(a,b)\in K}W_{ab}s_{ia}s_{jb}.
\]

Semua terma di sini linear atau pairwise selepas semua hidden/auxiliary diperkenalkan; struktur H dan K **ditetapkan dan sparse**. Q dan K ialah label bacaan; tiada dua jadual perkataan berasingan untuk dua peranan. Parameter boleh bergantung pada word ID, tetapi embedding lookup dan penyediaan parameter dijangka berlaku pada host pada versi awal. Landscape bermaksud energy atas state diskret; jika hidden wujud, landscape efektif atas visible ialah free energy \(F_w(s)=-\beta^{-1}\log\sum_h e^{-\beta E_w(s,h)}\), bukan semestinya sama dengan nilai energy pada satu hidden state. Ia boleh mempunyai beberapa kawasan kebarangkalian tinggi, namun bilangan dan tafsiran makna mesti diuji.

Dalam pelan awal, semua vocabulary disimpan sebagai parameter; token dalam window merupakan graf aktif. Ini penyederhanaan yang pengguna benarkan, bukan bukti seluruh kamus muat dan berinteraksi serentak pada satu chip. Perkataan sama pada dua posisi berkongsi parameter tetapi mempunyai spin aktif berlainan. \(W\) dikongsi dan dilatih; context menukar sampel joint. Pilih contohnya bacaan \(R_{ij}=d^{-1}\sum_a\langle s_{ia}s_{ja}\rangle\) daripada **sampel joint yang sama**, dengan tafsiran relevance dan normalisasi dipilih pada validation. Ini skor korelasi yang boleh negatif, bukan kebarangkalian atau tenaga pair automatik. Korelasi sama untuk i,j dan j,i; untuk attention berarah perlu mekanisme tambahan yang dinyatakan dan diuji.

## Contoh tepat: satu perkataan dengan dua lembah

Contoh buatan ini menggunakan 3 token aktif × 2 spin = 6 p-bit; dua token lain ialah konteks. Energy tanpa beta ialah:

\[
E=-s_{b1}s_{b2}-h\sum_{i=1}^{2}(s_{i1}+s_{i2})
-0.5\sum_{i=1}^{2}\sum_{a=1}^{2}s_{ba}s_{ia}.
\]

Perkataan `bank` sendiri mempunyai dua minimum yang sama, `++` dan `--`; context finance menggunakan h=+2, context river h=−2. Nilai dan label ini **ditetapkan manual** untuk memeriksa matematik, bukan dipelajari daripada bahasa. β=1. Enumerasi tepat kesemua 64 konfigurasi: h=+2 memberi P(bank=++)≈0.9421 dan P(bank=--)≈0.0204; apabila h=−2 nilai bertukar. Graf mempunyai lima edges dan degree maksimum tiga. [Kod fixture](reviews/energy_embedding_v2/ising_six_spin_fixture.py), [hasil](reviews/energy_embedding_v2/ising_six_spin_fixture.json). Ini bukti konsep matematik kecil; ia belum melatih model, membuktikan makna atau membuktikan embedding fizikal pada Z1.

## Urutan eksperimen utama

1. **Exact reference:** bina graph sama, enumerasi semua spin, semak tanda energy, nilai β, conditional distribution, normalizer, marginals dan gradien terhadap finite differences. Perlu membandingkan konfigurasi identik, bukan hanya gambar. Ujian 6 spin di atas menyediakan jawapan asas.
2. **THRML simulation:** replika graph tersebut menggunakan `SpinNode`, `IsingEBM`, coloring sah, `IsingSamplingProgram`, dan `sample_states`. Bezakan bool `True` daripada spin +1. Bandingkan kedua-dua konteks terhadap enumerasi dengan key bebas, multi-chain, Monte Carlo SE, effective sample size dan occupancy kedua-dua mode. Hitung finite-sample TV floor; semak warmup, sweep dan mixing. Jangan panggil simulation ini chip.
3. **Training toy:** buang biases manual berlabel sebagai solusi, gunakan data artificial konteks kewangan/sungai dan kontrak Ising yang ditetapkan di bawah, tanpa memberi sense label pada inference. Train biases/couplings perkataan dan W dalam batas struktur; lakukan baselines dan controls yang sama. Gunakan exact ranking gradient pada graph kecil dahulu. Word ID menentukan parameter dan bukan spin-state positif yang diperhatikan: jangan terus memanggil `estimate_kl_grad` seolah-olah input kita ialah data spin. Sebarang estimator melalui sampel mesti diturunkan untuk objective ini dan diaudit terhadap exact gradient. Simpan 3 seeds, negative controls dan metrik per sense. Dua lembah yang ditetapkan tangan tidak membuktikan latihan berjaya.
4. **Hardware mapping audit:** bina manifest node/edge, sign convention, β, graph coloring, maximum degree dan actual adjacency embedding. Audit physical subgraph, overhead ancilla/replication, precision/quantisation dan coupling range berdasarkan spesifikasi sebenar jika tersedia. Jika spesifikasi belum boleh diakses, tandakan unknown; jangan menganggap graph degree3 auto-fits atau meneka precision. Bagi subgraph mapping, cari injection daripada setiap logical node ke physical node yang berbeza dan pastikan setiap logical edge ialah physical edge yang dibenarkan. Bagi minor embedding, perlukan connected chains yang tidak berkongsi physical nodes, edges inter-chain untuk semua logical edges, chain couplings/penalty, kaedah decoding dan ujian decoded distribution pada suhu terhingga. Laluan berasingan bagi setiap edge sahaja tidak membuktikan mapping sah. Bandingkan taburan, correlation readout dan sense ranking sebelum/selepas mapping.
5. **Scale hanya selepas gate:** tambah dimensi diskret dan token secara berperingkat dengan sparse structure. Lapor bilangan pbit sebenar, edges, degree, jumlah parameters, chain steps, data loading, wall time, memory, dan jika hardware ada, host/chip communication serta energy end-to-end. Baseline vector dan Gaussian mesti menggunakan data/budget dan tugasan penilaian yang setanding. Pemeriksaan perbandingan chip hanya dibuat apabila akses benar-benar ada.

THRML ialah laluan Ising primary untuk graph toy. Torx ialah laluan tambahan untuk stochastic circuit/DFG yang benar-benar dipetakan dan diuji; Torx directed word→latent→context sedia ada bukan secara automatik Ising joint ini. Semakan discretization Gaussian boleh digunakan untuk membandingkan landscape, tetapi tidak menganggap arbitrary continuous function atau neural network secara automatik menjadi coupling Z1. Tiada tuntutan speedup, p-bit count produksi atau tenaga sebenar sehingga diukur.

## Kriteria keputusan

- **Lulus matematik:** energy dan gradient exact/finite differences sepadan dalam toleransi float64 yang ditetapkan sebelum run; probabiliti taburan ternormal dan kedua-dua konteks mengikuti fixture.
- **Lulus simulation:** marginals/readout THRML sepadan exact dalam uncertainty Monte Carlo yang dilaporkan, termasuk kedua-dua mode; gagal mixing/precision dikategorikan tidak konklusif sehingga sampel mencukupi.
- **Lulus toy learning:** guna gate sanity dan baseline dalam kontrak di bawah. Toy mudah mungkin membolehkan contextual vector mendapat 100%; itu tidak menuntut Ising lebih baik. Dakwaan superiority memerlukan benchmark relevan yang lebih besar dengan paired uncertainty pradaftar. Jika label semantik gagal, jangan menyimpulkan bahawa bentuk minima semata-mata ialah makna.
- **Lulus mapping:** injection subgraph atau minor embedding sah seperti langkah 4, overhead/kekangan diaudit, parameter boleh diprogram, serta ralat decoded distribution/readout memenuhi toleransi pradaftar. Degree≤16 hanyalah syarat perlu, bukan bukti cukup.
- **Lulus hardware:** jalankan pada backend dan chip yang dikenal pasti serta ukur hasil/sumber end-to-end. Selagi belum ada akses, status ialah `hardware mapping/execution not verified`.

Training/fixture pada CPU dan simulator ialah langkah untuk menyediakan eksperimen chip. Tidak semua operasi seperti pengindeksan vocabulary, pemilihan konteks, pengiraan loss/gradient dan bacaan mesti berlaku pada chip; profil hybrid host–chip mesti dinyatakan secara jujur.

## Kontrak pilot Ising yang boleh dilaksanakan

Kontrak ini khusus untuk model Ising dan mengatasi butiran Gaussian yang tidak serasi dalam pelan v2. Setiap word type w mempunyai **satu** jadual parameter `(b_w1,b_w2,J_w)`; setiap occurrence mempunyai dua spins tersendiri. Tiada hidden/ancilla untuk pilot. Unary dan joint physical energy, dengan β=1:

\[
e_w(s)=-b_{w1}s_1-b_{w2}s_2-J_ws_1s_2,\quad
E_{\rm phys}=\sum_{i=0}^2e_{w_i}(s_i)
-{\lambda\over2}\sum_{j=1}^2\sum_{a=1}^2 W_a s_{0a}s_{ja}.
\]

Graf aktif G ialah star dengan posisi pusat tetap 0 dan jiran 1,2; H ialah edge internal (1,2) setiap token, K ialah sambungan koordinat sama sahaja, W dikongsi semua spokes. Simpan juga edge dengan coupling0 supaya struktur tidak bergantung pada perkataan; ada tujuh potential edges dan max degree3. Fixture sebelumnya mempunyai hanya lima **nonzero** edges kerana J context=0. Q dipilih selepas satu joint sampling dengan membaca baris yang dikehendaki; menukar Q tidak menukar pusat/graph atau menjalankan semula inference. Graf star mempunyai keistimewaan posisi pusat, maka symmetry terhadap permutasi tiga posisi **tidak** dijamin.

Untuk generalisasi struktur star dengan L token, setiap spin pusat mempunyai degree `1+(L−1)` dan setiap peripheral degree2 untuk L>1. Hipotesis degree≤16 memberi L≤16, tetapi ini bukan bukti graph dapat di-embed; dengan hidden/ancilla atau graph lain kira semula semua edges dan overhead. Versi pilot L=3. W_1 dan W_2 ialah interaction weights yang boleh dilatih bersama parameters perkataan; mereka dikongsi, bukan jadual per-word. Awalan: `b=2.5*tanh(raw_b)`, `J=2*tanh(raw_J)`, `W=1.5*sigmoid(raw_W)`, dan λ=1. Nilai bounds ini ialah cadangan simulator, **bukan** julat hardware. Seeds17,29,43; `raw_b,raw_J` Normal(0,.1²), `raw_W=log(.8/(1.5-.8))` memberi W=.8. λ=0 ialah inference ablation pada checkpoint sama, bukan checkpoint retrain. Had ini tidak memaksa bank menjadi dua basin.

### Objective, normalisasi dan gradien

Untuk setiap type, \(Z_w=\sum_{s\in\{-1,1\}^2}e^{-e_w(s)}\). Gunakan unary ternormal \(U_w=e_w+\log Z_w\). Dengan \(Z_{\rm phys}(x)=\sum_{s_{0:2}} e^{-E_{\rm phys}}\):

\[
F_3(x)=-\log Z_{\rm phys}(x)+\sum_{i=0}^2\log Z_{w_i}.
\]

Jumlah ini hanya empat state bagi setiap unary dan 64 state bagi joint; `logsumexp` untuk kestabilan. F3 ialah **free energy ternormal yang digunakan sebagai ranking score**, bukan raw chip energy. Penambahan \(\sum\log Z_w\) tidak mengubah sampel joint untuk window tetap, tetapi mengubah skor antara window: hitung offset ini pada host kecuali hardware/backend terbukti mampu. Pada λ=0, F3=0 tepat secara matematik.

Untuk perbandingan lexical F2, guna graph 2 token dengan H sama dan satu edge inter-token setiap koordinat, pekali `−λ W_a` (a_ij=1); F2 didefinisikan dengan normalisasi dua unary yang sama. Objective ialah `mean softplus(1+F2(w,v_pos)−F2(w,v_neg))` untuk lima negatives per positif. Joint: `mean softplus(.25+F3(x_pos)−F3(x_corrupt)) + .25*L_lex`, dua corruptions per window. Ini supervised-by-cooccurrence contrastive ranking pada data synthetic, bukan MLE bahasa. `F2` dan `F3` ialah fungsi Ising sendiri, bukan Gaussian overlap.

Untuk apa-apa parameter θ, exact gradient β=1 ialah

\[
\nabla_\theta F_3=\mathbb E_{P_{\rm joint}}[\nabla_\theta E_{\rm phys}]
-\sum_i\mathbb E_{p_{w_i}}[\nabla_\theta e_{w_i}].
\]

Derivasi yang sama digunakan untuk F2; repeated word occurrences menyumbang setiap kali. Gunakan differentiable enumerasi bagi kedua-dua inner F dan outer softplus. Audit dengan central finite differences step1e−5, abs error≤1e−6 atau relative≤1e−4 pada semua jenis parameter, termasuk W dan word berulang. Untuk proof estimator kemudian, bandingkan unary dan joint moments tersampel dengan exact serta gradient itu; fixed-temperature Gibbs sample tidak boleh dipakai sebagai observed data positif. Menyisip anggaran F tersampel ke outer softplus menghasilkan bias nonlinear yang perlu diaudit, dan `jax.grad` melalui sample diskret bukan gradient sah. Jangan menamakan latihan ini CD/PCD atau THRML MLE sebelum objective/estimator masing-masing dibuktikan sepadan.

### Dataset, schedule dan baseline

Gunakan grammar 26 token, 84 window dan split 48/18/18, corruption, negatif buatan serta manifest tepat yang ditetapkan dalam [v2 §8.2](WORD_ENERGY_EMBEDDING_V2_PLAN.md); ini oracle-assisted synthetic negatives dan tidak menggambarkan penyeliaan natural text. Enam ordered lexical pairs per train window, lima negatives, rules dan RNG tetap sama. Early lexical: Adam lr .01, 400 updates, batch64, grad clip5; joint: Adam baru lr .003, 100 updates, batch4, clip5, dan .25 auxiliary lexical. Schedules batch RNG4040/5050, negatives6060, validation7070 dan corruption202, seperti v2; snapshot semua IDs/hash sebelum runs. Seeds model sahaja17,29,43. Setiap20 updates pilih checkpoint menggunakan validation; lexical pilih macro-AUC lexical, tie loss rendah; joint pilih macro sense accuracy, tie loss joint rendah; tie akhir checkpoint lebih awal. Test hanya dibuka selepas checkpoint/protokol dibekukan.

Baselines: frequency-only, random Ising initialization, Ising unary-only coupling W=0, same Ising checkpoint λ0 vs λ1, tied SGNS D3 (78 lexical parameters; Ising 26×3+2=80), dan untied SGNS D1 (52). SGNS loss `softplus(−u_w·v_pos)+sum_5 softplus(u_w·v_neg)`, sama data, schedule dan optimizer; vector contextual readout menggunakan input table sahaja seperti v2 §8.4. Jangan menyamakan parameter count atau training objective secara palsu; model Ising joint mendapat latihan tambahan, maka bandingkan stage lexical dengan lexical dan lapor kos tambahan. Contextual vector ialah baseline sebenar, bukan sasaran yang mesti diatasi pada toy.

### Sense, retrieval dan distractor

Map sense berdasarkan *anchor train sahaja*: bagi setiap state s dalam empat state, \(q_F(s)\) dan \(q_R(s)\) ialah mean unary probability daripada anchors family F dan R bagi checkpoint yang dinilai. Tetapkan region F jika qF>qR, R jika qR>qF; equality dikongsi separuh oleh kedua-dua region. Gunakan label family generator untuk evaluation map sahaja; ia bukan input model pada inference. Validation boleh menggunakan map setiap checkpoint menurut algoritma ini; bekukan checkpoint dan map sebelum test. Ini invariant kepada pertukaran nama states yang konsisten dan bukan component-index atau label diberi pada model.

Sense classification: exact/sample joint P(bank state dalam region F/R), argmax dengan tie half credit; lapor prior bank sahaja, conditional mass, gain λ1 vs λ0 pada checkpoint sama, macro accuracy atas12 bank test windows serta counts dua family. Lexical retrieval: query bank, candidates24 anchors, rank −F2; macro pairwise AUC F versus M dan R versus M dengan ties setengah. R ialah correlation bersama active tokens sahaja; tidak menyokong retrieval seluruh vocabulary tanpa tambahan scorer. Lapor semua tiga edges serta Monte Carlo error. Tiada claim bahawa correlation sama dengan energy atau makna.

**Distractor fixtures manual:** b_finance=(2,2), b_river=(−2,−2), bank b=(0,0), J_bank=1, J_others=0, W=(1,1), λ=1. Untuk 3 token dengan satu finance dan satu river yang sama kuat, exact enumeration mendapati kedua-dua R(bank,context)=0.040733022: model memberi tie, selaras dengan symmetry apabila tiada bukti tambahan untuk memilih satu sense. Ia tidak boleh digunakan sebagai bukti model membezakan relevance dalam konteks ambigu ini. Menambah satu finance lain (4 token, 8 spin, a=1/3) menghasilkan R≈0.511641386 kepada setiap finance dan R≈−0.461382914 kepada river; konfigurasi bertukar family memberi keputusan cerminan. Ini contoh yang direka tangan, tidak membuktikan pembelajaran atau generalisasi. Rekod ujian [script](reviews/z1_first/distractor_fixture.py) dan [hasil](reviews/z1_first/distractor_fixture.json). Semak kedua-dua contoh serta invariance apabila peripheral ditukar; jika model learned tidak membezakan relevance, laporkan hasil, bukan ubah threshold selepas melihatnya.

### Precision, controls dan gate

Exact enumeration float64: semak Σp=1 dalam1e−12, fixture margins dalam1e−10, F(λ0) abs≤1e−10 dan analytical/finite-difference gradients seperti atas. Gauge: menambah pemalar kepada e_w tidak mengubah unary normalized, joint distribution atau F2/F3; raw F sahaja berubah mengikut bilangan occurrence. THRML replica: empat chains dengan permulaan tersebar, burn-in500 sweeps, kemudian2,000 sweeps/chain; double dua kali jika perlu dalam bajet. Compare marginal bank dan semua R edges dengan exact dalam max(.02,3×MCSE), MCSE≤.01, multi-chain ESS≥400 dan split R-hat<1.05 untuk indikator bank-basin dan setiap R edge; bincangkan indikator constant berbanding exact, bukan lapor ESS palsu. Kiraan error finite-sample TV vs exact ialah tambahan; mixed-time state tidak disebut independent.

Sebelum mengukur semantic test, toy sanity gate pradaftar: setiap seed macro sense accuracy≥10/12, lexical macro AUC≥.75, correct-region posterior gain λ1−λ0≥.03; lapor kedua-dua keluarga dan uncertain intervals dengan 12 contoh per seed, tanpa pooling seeds sebagai 36 dokumen bebas. Gagal gate bermaksud pilot belum mencapai sasaran pada config tersebut, bukan kegagalan seluruh pendekatan. Jika runtime/MC precision tidak memadai, status tidak konklusif. Bandingkan random/frequency dan contextual vector; tuntutan lebih baik memerlukan data lebih besar, split bebas dan paired CI. Toy yang vector100% tidak boleh membuktikan kelebihan Ising.

Collapse checks: std F2 seluruh pairs, inter-word bias/J spread, bank occupancy kedua-dua regions, R saturation, dan state/chain mixing. Compare pretraining random model dengan trained; jika training tidak menambah signal, jangan dakwa belajar. Run control mengacak lexical-positive contexts sahaja (RNG8080) sambil kekalkan negatives grammar-filtered dan joint supervision; interpretasi terhad kepada tambahan lexical positives, **bukan** control yang membuang seluruh supervision. Lapor juga bahawa grammar oracle digunakan untuk memilih negatives/corruptions. Jangan memaksa setiap word mempunyai dua makna.

Had keseluruhan awal ialah30 minit masa berlalu, ≤4 CPU threads, sasaran≤2 GiB; dahulukan exact math, satu-seed smoke, kemudian tiga seeds dan simulator checks. Simpan config, source hash, vocab/split/negative hashes, OS/library versions, runtime, checkpoint, validation choice, MC diagnostics dan keputusan yang belum sempat. Tiada janji semua gate selesai dalam bajet. Plan tidak menggantikan historical snapshot critic lama.
