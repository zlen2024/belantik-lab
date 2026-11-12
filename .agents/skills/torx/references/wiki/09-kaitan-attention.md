# 09. Kaitan dengan eksperimen attention kita

Ringkasan apa yang diuji dan apa yang masih terbuka. Semua nombor dari larian kita sendiri kecuali dinyatakan.

## Fakta Z1 (sumber luar)

- Blog Extropic: 269,568 p-bit, 8 core, 215,904 parameter gandingan, >50 MHz, <1 W, degree 16, early access 2027.
- Whitepaper: Z1 hardwire graf degree-16, jalankan chromatic Gibbs, hos hantar `h_i` dan `J_ij`, terima sampel. Tiada adder atau accumulator disebut.
- 215,904 parameter gandingan jauh kurang daripada bilangan tepi (~2.16 juta). Belum jelas sama ada setiap `J` boleh diprogram sendiri.

## Keputusan kita

| Percubaan | Keputusan |
|---|---|
| Kaedah A (288 coin dikongsi, `u·v`) | ralat 74.6 → 4.2 untuk N 10 → 10⁴ |
| Kaedah B (576 p-bit bebas, bias = q,k) | ralat 56 → 2.4; lebih tepat daripada A |
| Gate AND on-chip (`onchip_gate.py`) | λ=4 terbaik: RMS 8.0 unit Σmₐm_b (133 dalam unit q·k), N=1000, 6 larian (±30%); B pada N=1000 ialah 6.35 dalam unit q·k, jadi kira-kira 20× lebih teruk (bukan 14×) |
| PISING pasangan | J ≠ 0 mengubah marginal; bukan pendarab |
| `torx_learn.py` | PNOT sebagai p-bit disahkan; ralat ikut 1/√N |

## Reka bentuk yang kekal terbaik setakat ini

Kaedah B: p-bit bebas, bias dari hos. Produk per-dimensi dan jumlah dikira oleh hos. Z1 hanya menjana sampel.

## Softmax melalui pdit / spin satu-panas (sebahagiannya diuji)

Notebook 14: `p(h=e_k | v) = softmax_k(θ_k(v))`, `θ_k(v) = v·μ_k/σ² − ‖μ_k‖²/(2σ²) + log π_k + (terma bebas-k)`. Peta ke attention: `v = q`, `μ_k = k_j`. Untuk mendapat `softmax(q·k_j/√d)` perlu `σ² = √d` **dan** bias per kunci `log π_j = ‖k_j‖²/(2√d)` untuk membatalkan terma norma (`c_k` bukan berat campuran). Hos mengira satu nombor setiap kunci. Persampelan indeks kunci = sampel pdit.

Soalan terbuka:
1. Skor `θ` tetap dikira dengan hos (dot product digital), jadi Z1 hanya membantu persampelan. Adakah itu berbaloi?
2. Pdit pada Z1 mungkin dikodkan sebagai kumpulan spin dengan kekangan satu-panas (winner-take-all). Kekangan itu perlukan gandingan antara spin; degree 16 membataskan K.
3. Berapa banyak kunci boleh dimuatkan: dengan degree 16, kumpulan satu-panas besar perlu pembahagian.

## Keputusan pusingan 2 (THRML, `cookbook_8_attention_thrml.py`)

- Attention THRML dengan DAC bias: TV 0.059 (4-bit), 0.021 (6), 0.009 (8), 0.012 (16) pada N=10,000: 8-bit tidak dapat dibezakan daripada 16-bit.
- Softmax satu-panas dengan K=8 spin dan gandingan semua-ke-semua: taburan bersyarat satu-panas tepat `softmax(θ)`; Gibbs THRML: `lam=4` kebocoran 8.5%, TV 0.019; `lam=6` hampir beku (97% sampel sama berturutan).
- Sampel berkorelasi dengan `τ0 = 100 ns`: `N_eff/N` = 0.46 jika disampel setiap 100 ns, 0.05 setiap 10 ns.

- Pensampelan indeks kunci hujung-ke-hujung (`cookbook_10_key_sampling.py`, 12 benih): TV 0.0079 ± 0.0028 pada 5 langkah Gibbs setiap sampel, dekat lantai multinomial 0.0073; kebocoran 0.197 (dibuang); DAC bias perlu julat ±10 dan 8-bit memberi TV 0.0170. Nombor rasmi: `../estimator-math.md`.

## Cadangan seterusnya

1. (Selesai) Persampelan softmax satu-panas K=8 dalam THRML, dibandingkan dengan softmax tepat (`cookbook_8`, `cookbook_10`).
2. (Selesai) Bilangan sampel untuk TV tertentu dan kesan sampel berkorelasi (`cookbook_4`, `cookbook_8`).
3. Masih terbuka: kira-kira masa dan tenaga dengan andaian tersurat (>50 MHz, <1 W), dan penyematan satu-panas pada graf sebenar Z1 (topologi tidak diketahui).

## PoC: attention sebagai mesin Boltzmann yang dilatih (THRML, `attn_thrml/ebm_attention_poc.py`)

Visible = query `q` (d=32 spin, di-clamp), label = K=8 spin satu-panas (posisi yang dirujuk), gandingan `J[d,K]` dipelajari, `y-y = -lam/2` (lam=3) tetap. Tenaga posisi `j` rendah bila `q` sepadan dengan kunci `j`: `p(j|q) ~ exp(2 J_j . q)`. Teacher: `softmax(1.5 q.k_j/sqrt(d))` (berat maksimum purata 0.45, tidak tepu). Kunci statik, tiada had degree 16. Dijalankan dari luar `master/`.

| Peringkat | Hasil (skrip yang sama) |
|---|---|
| S0, `J = beta_t k/(2 sqrt d)` tanpa latihan | TV tepat 0.0000; THRML disampel 0.0075 +/- 0.0034 (lantai 0.0069); kebocoran 0.22 |
| S1, CD bersyarat (`q` di-clamp dalam kedua-dua fasa), lr 0.003 susut kosinus, 4000 langkah | TV tepat 0.0120, THRML disampel 0.0153 (lantai 0.0062); garis dasar softmax-regresi parameter sama 0.0102 |
| S1, kosinus `J[:,j]` dengan `k_j` | 0.937 (garis dasar 0.957): sedikit di bawah sasaran 0.95 |

Pemerhatian: (1) logit EBM = `2 J.q`, jadi skala `J` 0.116 setara skala garis dasar 0.237 (sasaran 0.265); (2) lr tetap 0.02 menyangkut TV pada 0.11 (hingar), lr yang disusutkan memberi 0.024 pada 1500 langkah dan 0.012 pada 4000; (3) lapisan hidden gaya XOR **tidak diperlukan** untuk kunci statik kerana `q.k` linear dalam `q`; (4) tiada had degree 16: setiap label berkait dengan 32 query dan 7 label lain (39 jiran), jadi S3 akan gagal tanpa pembahagian; (5) kunci dinamik (S2) belum dicuba. Semua emulasi THRML.

### S3, S2 dan S2+S3 (skrip `attn_thrml/ebm_attention_s3.py`, `ebm_attention_s2.py`, `ebm_attention_s2s3.py`)

Semua THRML emulasi; garis dasar ialah softmax-regresi tak terkekang. Degree dikira daripada graf yang dibina.

| Peringkat | Konfigurasi | TV tepat | TV disampel (lantai) | Rujukan |
|---|---|---|---|---|
| S3a, kunci statik, sambungan terus, degree label 16 | K=8, d=9 | 0.0085 | 0.0112 (0.0060) | garis dasar 0.0057 |
| S3a | K=4, d=13 | 0.0097 | 0.0121 (0.0039) | garis dasar 0.0068 |
| S3b, kunci statik, hidden perantara 3 kumpulan, degree 8/12/10 | d=32, 6000 langkah (cloud) | 0.0586 | 0.0688 +/- 0.0325 (0.0062) | garis dasar 0.0118; mendatar selepas langkah 4500 |
| S2, kunci dinamik, tiada had degree (label 39) | d=16, permulaan xnor (cloud) | 0.0166 | 0.0209 +/- 0.0068 (0.0090) | seragam 0.441 |
| S2, permulaan rawak (setempat) | d=16 | 0.0254 | 0.0309 +/- 0.0116 (0.0090) | cerun logit 0.348 lawan sasaran 0.375 |
| S2+S3, kunci dinamik, degree maksimum 16 | d=16, 3000 langkah, lr 0.003, warmup 150 (cloud) | 0.0599 (terbaik 0.0394 pada langkah 1125) | 0.0925 +/- 0.0290 pada 5 langkah/sampel; 0.0563 +/- 0.0207 pada 20 langkah/sampel (lantai 0.0091) | seragam 0.441 |

Pemerhatian: (1) S2 menggunakan seni bina yang kita reka (pengesan XNOR dengan berat dikongsi merentas posisi dan bit); CD menemuinya dari permulaan rawak dalam kelas itu, tetapi ini bukan bukti hasil darab dipelajari dari kosong; (2) S2+S3 dengan lr 0.01 meletup (kebocoran 0.88, TV disampel `nan`); dengan lr 0.003 dan warmup 30 TV drift naik selepas 0.0655 (setempat), warmup 150 mengurangkannya tetapi TV tepat masih naik daripada 0.0394 ke 0.0599 pada cloud, jadi bias CD jangka pendek belum hilang sepenuhnya; (3) jurang TV disampel dengan tepat pada S2+S3 ialah rantai yang lambat: 20 langkah Gibbs setiap sampel memberi 0.0563, hampir 0.0599; (4) kekangan degree 16 pada d=32 menambah kira-kira 0.047 TV berbanding garis dasar (S3b), kerana spin kumpulan binari meringkaskan 10-11 bit; (5) tidak diuji pada d=288. Kos cloud: kira-kira $0.31 untuk semua job (job S2+S3 lr 0.01 tersasar dan tidak boleh dibatalkan kerana disekat semakan keselamatan).

### Purata syiling dengan berat sebagai gandingan (`attn_thrml/coin_average_z1.py`, THRML emulasi)

Setiap dimensi: syiling adil `z_i` bergandingan dengan dua p-bit bacaan `a_i` (berat `atanh(m_q,i)`) dan `b_i` (berat `atanh(m_k,i)`); 3N = 864 p-bit, degree 2, dua blok Gibbs. Diberi `z_i`, `E[a_i] = m_q,i z_i`, jadi `E[a_i b_i] = m_q,i m_k,i` dan sebutan silang `E[a_i b_j] = 0` kerana syiling bebas. Dua cara baca daripada sampel yang sama, d=288 dan data sama seperti `z1_attention.py` (q.k tepat 156.26), 6 ulangan, RMS ralat:

| N sampel | R1: jumlah `a_i b_i` | R2: (jumlah `a`) x (jumlah `b`) | kaedah B (2N p-bit bebas) |
|---|---|---|---|
| 10 | 47.87 | 1348.73 | 56.18 |
| 100 | 28.60 | 492.53 | 26.55 |
| 1000 | 11.31 | 133.25 | 8.20 |
| 10000 | 2.53 | 25.56 | 1.42 |

Pemerhatian: (1) kedua-dua cara baca memberi q.k secara purata, tetapi R2 (cara tambah semua lalu darab) mempunyai ralat kira-kira 10 kali lebih besar pada N=10000 kerana varians `N^2` berbanding `N` pada R1, jadi memerlukan kira-kira 100 kali lebih banyak sampel; (2) R1 sepadan kaedah B dalam hingar 6 ulangan, jadi gandingan tidak memberi kelebihan statistik dan menggunakan 3N bukan 2N p-bit; (3) auto-korelasi lag-1 ialah -0.001 pada 1 langkah Gibbs, jadi sampel hampir bebas dan tidak perlu penjarangan; (4) hos masih menjumlah N bit setiap sampel (Z1 tiada adder), jadi kerja digital ialah N x M operasi, jauh lebih banyak daripada satu dot product biasa (N operasi). Tiada tuntutan tentang cip sebenar: julat DAC gandingan (8-bit, +/-3) ialah andaian.

#### Attention K kunci dan kos (`attn_thrml/coin_attention_cost.py`, THRML emulasi)

Satu query dan K=8 kunci berkongsi syiling: `z_i` bergandingan dengan satu bacaan query `a_i` dan K bacaan kunci `b[j,i]` (2880 p-bit, degree syiling 9). TV kepada softmax tepat (berat maksimum 0.36), purata 8 ulangan, auto-korelasi lag-1 skor +0.001:

| N sampel | bintang R1 | bintang R2 | kaedah B |
|---|---|---|---|
| 100 | 0.0971 | 0.7967 | 0.1178 |
| 1000 | 0.0386 | 0.5120 | 0.0381 |
| 10000 | 0.0112 | 0.1641 | 0.0126 |

Kos satu baris attention (1 query x 8 kunci, d=288) pada CPU ini (numpy): `q.K^T` tepat ialah 2304 darab-tambah dalam 1.9 us. Hos mengurangkan bit sampel: N=100 ambil 0.029 ms (16 kali masa tepat), N=1000 0.408 ms (217 kali), N=10000 6.776 ms (3600 kali) untuk TV 0.0112; bilangan operasi bit ialah 100, 1000 dan 10000 kali operasi tepat. Masa cip pada N=10000 ialah 0.4 ms **dengan andaian** satu sapuan = 2 kemas kini blok pada 50 MHz (maksud ">50 MHz" dalam blog tidak diketahui). Pemerhatian: (1) R1 sepadan kaedah B, R2 jauh lebih teruk (0.164 pada N=10000); (2) walaupun cip dianggap serta-merta, kerja hos lebih besar daripada q.K^T tepat; einsum numpy pada float bukan pengurangan popcount bit yang dioptimumkan, jadi masa hos mungkin lebih rendah, tetapi bilangan operasi tidak; (3) GPU tidak diukur. Kesimpulan: tiada kelebihan kos untuk dot product atau attention melalui purata p-bit.

### Blok attention EBM pada dataset bayi (`attn_thrml/ebm_attn_baby/`, THRML emulasi, degree <= 16)

Tugasan: recall kunci-nilai sintetik (8 pasangan, kunci berbeza, soalan satu kunci). Token menjadi kod +-1 (16 bit); bit query dan kunci di-clamp; pengesan padanan (degree 3) menyuap spin kumpulan (degree 15) menyuap label (degree 10); query degree 16. Berat dikuantiti 8-bit (julat DAC andaian). Dengan hidden dijumlahkan tepat, perhatian = `softmax(skor_j + 2 b_j)`, jadi latihan ialah backprop biasa; pada inferens perhatian dianggar daripada sampel THRML. Skrip: `train.py`, `eval_chip.py`, `try_model.py`, README dalam folder.

| Model (3000 langkah, 5 benih) | Ketepatan, hingar bit query 0 / 0.1 / 0.2 / 0.3 | Benih berjaya |
|---|---|---|
| EBM degree <= 16, 8-bit, permulaan xnor | 0.995 / 0.943 / 0.815 / 0.608 | 5/5 |
| EBM permulaan rawak | 0.646 / 0.609 / 0.538 / 0.438 | 2/5 |
| EBM tanpa had degree (rujukan) | 0.986 / 0.958 / 0.837 / 0.624 | 5/5 |
| dot-product softmax, kod sama | 1.000 / 0.980 / 0.867 / 0.646 | 5/5 |
| tanpa attention | 0.322 | 0/5 |

Emulasi cip pada model chip-aware (128 contoh): hingar 0, 30 sampel beri 1.000 dengan 97% sampel satu-panas (model biasa 34%); hingar 0.1, 100 sampel beri 0.984 (perhatian tepat 0.977). Pemerhatian: (1) blok EBM sepadan dengan dot-product pada tugasan mudah ini, tidak mengalahkannya; (2) sukar dilatih: perlu permulaan xnor dan lr kecil, dari permulaan rawak hanya 2/5 benih; (3) kebocoran (keadaan bukan satu-panas) membazirkan sampel; fine-tune dengan penalti menurunkan kebocoran purata daripada 0.31 kepada 0.17 (satu benih guna sandaran); (4) `K x R <= 16` menghadkan K kepada 8; (5) masa cip 120 us untuk 30 sampel ialah andaian 50 MHz, berbanding kira-kira satu mikrosaat untuk attention digital biasa; (6) tidak diuji: teks sebenar, banyak kepala, d=288, perkakasan.

## Peringatan jujur

Dot product melalui `1/√N` hampir pasti kalah kepada GPU untuk matmul biasa. Potensi Z1 ialah sampling model tenaga, bukan aritmetik. Tuntutan 10,000× tenaga dalam blog ialah daripada syarikat, untuk "workload yang sesuai", dan belum disahkan pihak bebas.
