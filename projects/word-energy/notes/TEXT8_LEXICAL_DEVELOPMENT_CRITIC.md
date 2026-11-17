# Semakan critic: pilot lexical text8

17 November 2026. GPT-6.1 Sol Ultra merancang pilot dan mengaudit hasil secara
read-only. Teks, vocabulary, counts, pasangan/posisi, noise dan epoch orders
dibina semula daripada sumber; semuanya sepadan dengan hashes. Dua belas run
FINISHED, 48 checkpoint sepadan dengan hash, source versions konsisten dan
paired initialization betul. Loss yang dikira secara bebas menggunakan
NumPy sepadan sehingga 4.45e-16. Tiga model seed berkongsi satu corpus dan
sample, bukan tiga dataset bebas.

Critic menemui 10,847 false non-ties merentas 48 checkpoint: positive dan
noise ber-ID sama boleh menghasilkan perbezaan pembundaran apabila skor
dikira dengan batch shapes berbeza. Tiada sign disagreement ditemui di luar
candidate ID yang sama. Exact-ID correction menghasilkan AUC dan MRR yang
sepadan dengan independent canonical score lookup bagi semua checkpoint.
Metrik run asal tidak diubah; lihat [audit correction](TEXT8_IDENTICAL_CANDIDATE_METRIC_REPAIR.md)
dan corrected MLflow audit `e14d3b8e72e1467597b3826575be896f`.
Skrip independent NumPy critic tidak disimpan; skrip correction reproducible
dan source snapshots disimpan dalam MLflow dan Git.

| Arm | Corrected AUC mean | Corrected MRR@21 mean |
|---|---:|---:|
| Ising ranking | 0.639893 | 0.274456 |
| Vector ranking | 0.678610 | 0.334231 |
| SGNS | 0.689757 | 0.327954 |
| Ising shuffled | 0.622033 | 0.242373 |
| Frequency only | 0.660195 | 0.265787 |

Ising meningkat daripada initialization dan mengatasi paired shuffled control
pada AUC untuk semua seed, namun AUC-nya di bawah frequency dan kedua-dua
vector arms. Mean MRR Ising sedikit di atas frequency, walaupun seed 29 tidak;
vector-ranking mempunyai mean MRR lebih tinggi daripada SGNS meskipun AUC
SGNS lebih tinggi. Jangan membuat satu susunan global yang mengabaikan metrik.
Vector-ranking mengatasi Ising dengan ranking loss yang sama, maka perbezaan
tidak boleh dijelaskan semata-mata sebagai SGNS loss yang berbeza. Ia belum
memisahkan effect parameter count, symmetry, bounds dan optimization.

Ising mean unary entropy sekitar 0.678 nat; hanya 1.1–1.5% token mempunyai
dominant-state probability >0.99. Dakwaan seluruh vocabulary collapse tidak
disokong. Shared couplings mendekati 1.43–1.45 pada bound 1.5, menunjukkan
scale pressure tetapi bukan bukti puncanya. Missing candidate-frequency offset
atau distinct target/context roles ialah hipotesis bersaing, bukan hasil yang
telah dibuktikan.

Zero-coupling memberikan semua scores sifar secara analitik, tetapi evaluator
floating semasa memberikan |score| <=1.78e-15 dan boleh membuat false non-ties
merentas ID berlainan. Exact-ID repair membetulkan 48 checkpoint sebenar di
atas, tetapi **tidak** membetulkan semua structural ties dalam zero-coupling
atau bentuk simetri lain. Ujian analytical zero menyokong AUC0.5 dan expected
MRR `H21/21≈0.1735885`. Sebelum eksperimen baharu, wujudkan scorer canonical
atau enforce struktur tie yang diketahui; jangan guna tolerance global yang
boleh menyamakan skor berlainan secara salah.

Interpretasi kekal pada satu blok text8 development, validation token coverage
66.19%, pasangan bertindih/repeated, q^0.75 dan noise yang boleh menjadi false
negatives. Text8 tiada article boundaries; historical provenance tidak lengkap.
Tiada reserved-test scoring, human sense labels, THRML parity atau chip run.
Next action yang disyorkan: simpan pembetulan dan audit ini; pilih eksperimen
baharu berdasarkan competing hypotheses, tanpa retune atau membuka test pada
panel sedia ada.
