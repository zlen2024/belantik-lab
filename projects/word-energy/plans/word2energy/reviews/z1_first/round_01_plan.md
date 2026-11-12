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
3. **Training toy:** buang biases manual berlabel sebagai solusi, gunakan data artificial konteks kewangan/sungai pada split tetap seperti pelan lama, tanpa memberi sense label pada inference. Train biases/couplings perkataan dan W dalam batas struktur; lakukan baselines dan controls yang sama. Gunakan exact gradient pada graph kecil dahulu; THRML positive/negative phase mesti dibandingkan dengan gradient enumerasi sebelum CD/PCD. Simpan 3 seeds, negative controls dan metrik per sense. Dua lembah yang ditetapkan tangan tidak membuktikan latihan berjaya.
4. **Hardware mapping audit:** bina manifest node/edge, sign convention, β, graph coloring, maximum degree dan actual adjacency embedding. Audit physical subgraph, overhead ancilla/replication, precision/quantisation dan coupling range berdasarkan spesifikasi sebenar jika tersedia. Jika spesifikasi belum boleh diakses, tandakan unknown; jangan menganggap graph degree3 auto-fits atau meneka precision. Bandingkan taburan, correlation readout dan sense ranking sebelum/selepas mapping.
5. **Scale hanya selepas gate:** tambah dimensi diskret dan token secara berperingkat dengan sparse structure. Lapor bilangan pbit sebenar, edges, degree, jumlah parameters, chain steps, data loading, wall time, memory, dan jika hardware ada, host/chip communication serta energy end-to-end. Baseline vector dan Gaussian mesti menggunakan data/budget dan tugasan penilaian yang setanding. Pemeriksaan perbandingan chip hanya dibuat apabila akses benar-benar ada.

THRML ialah laluan Ising primary untuk graph toy. Torx ialah laluan tambahan untuk stochastic circuit/DFG yang benar-benar dipetakan dan diuji; Torx directed word→latent→context sedia ada bukan secara automatik Ising joint ini. Semakan discretization Gaussian boleh digunakan untuk membandingkan landscape, tetapi tidak menganggap arbitrary continuous function atau neural network secara automatik menjadi coupling Z1. Tiada tuntutan speedup, p-bit count produksi atau tenaga sebenar sehingga diukur.

## Kriteria keputusan

- **Lulus matematik:** energy dan gradient exact/finite differences sepadan dalam toleransi float64 yang ditetapkan sebelum run; probabiliti taburan ternormal dan kedua-dua konteks mengikuti fixture.
- **Lulus simulation:** marginals/readout THRML sepadan exact dalam uncertainty Monte Carlo yang dilaporkan, termasuk kedua-dua mode; gagal mixing/precision dikategorikan tidak konklusif sehingga sampel mencukupi.
- **Lulus learning:** pada holdout, bacaan sense lebih baik daripada frequency, random dan baseline vektor berkonteks dengan protokol dan uncertainty pradaftar. Jika label semantik gagal, jangan menyimpulkan bahawa bentuk minima semata-mata ialah makna.
- **Lulus mapping:** setiap logical edge mempunyai laluan ke graf fizikal dan overhead/kekangan berjaya diaudit, parameter boleh diprogram, serta ralat taburan/readout memenuhi toleransi pradaftar. Degree≤16 hanyalah syarat perlu, bukan bukti cukup.
- **Lulus hardware:** jalankan pada backend dan chip yang dikenal pasti serta ukur hasil/sumber end-to-end. Selagi belum ada akses, status ialah `hardware mapping/execution not verified`.

Training/fixture pada CPU dan simulator ialah langkah untuk menyediakan eksperimen chip. Tidak semua operasi seperti pengindeksan vocabulary, pemilihan konteks, pengiraan loss/gradient dan bacaan mesti berlaku pada chip; profil hybrid host–chip mesti dinyatakan secara jujur.
