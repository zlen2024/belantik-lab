# Audit feasibility model dan latihan pada corpus 227

17 November 2026. Audit eksploratori ini menjawab soalan terhad: adakah
parameter bagi seni bina Ising dua-spin yang sama **wujud** dan mencapai
ambang toy pada corpus 227, dan adakah permulaan 7043 yang sama boleh
berakhir di keadaan yang lulus atau gagal? Bukti sedia ada sudah menunjukkan
kedua-duanya; audit ini mengira semula dan mengawal pembacaan konteks. Ia
bukan ujian unseen-data, bukti kebolehcapaian dari semua initial states,
bukti optimum global, atau pengesahan hardware.

## Checkpoint dan protokol

Ambil fixed joint100 untuk lapan run asal corpus 227: permutasi 7019 dan
7043, masing-masing seed 251, 257, 263, 269. Tambah dua witness 7043
early-bank seed 251 dan 257 yang sudah diketahui lulus; ia menggunakan initial
parameters/data yang dipadankan kepada baseline 7043 tetapi loss latihan
berbeza buat sementara. Pilihan ini posthoc dan dibekukan sebagai witness;
jangan memilih seed lain berdasarkan hasil baharu. Semak sumber, tensor
shapes, initial hashes berpasangan, data/vocab/schedule hash, corrected F3 dan
fixed handoff. Kira semula sense, lexical AUC, context gain dan per-family
validation counts; ambang ialah sense >=10/12, AUC >=0.75, gain >=0.03.
Kira juga F2 train penuh, cross-F/R negative loss dan F3+0.25 F2 train penuh
daripada enumerasi bebas NumPy yang sedia ada.

Semua checkpoint dinilai oleh evaluator dan data corpus 227 yang sama; nombor
permutasi hanyalah metadata initialisation. Gunakan F/R region yang diperoleh
daripada train anchor masing-masing, seperti metrik asal. Dua kawalan bacaan:

1. Tetapkan coupling window kepada sifar pada checkpoint yang sama dan ukur
   perbezaan gain berbanding zero-coupling prior; gain mesti sifar sehingga
   numerical tolerance. Ia ujian bacaan, bukan inference fizikal pada chip.
2. Tukar context anchors antara F dan R secara deterministic berpasangan
   pada validation windows, sambil kekalkan labels asal; jangan guna corrupted
   windows untuk melaporkan ranking loss. Ukur accuracy dan gain pada context
   yang dipertukar sebagai sanity check label/konteks. Ini audit negatif pada
   validation yang telah diperhatikan, bukan held-out generalization.

Sebagai analisis posthoc tambahan, kira lower bound untuk posterior state
dominan bank bagi **sebarang** dua context leaves dengan bank unary dan
coupling yang dibekukan. Untuk state s* yang dominan, gunakan
`r(s)=q_bank(s)/q_bank(s*) * exp(sum_a g_a*abs(s_a-s*_a))` dan
`P(s*|context) >= 1/(1+sum_{s!=s*} r(s))`. Ketidaksamaan ini menggunakan
positive pairwise coupling dengan total normalized spoke scale satu. Sahkan
bound terhadap setiap validation context yang diperhatikan. Bound hanya untuk
parameter bank+coupling yang dibekukan, bukan had umum seni bina atau bukti
bahawa latihan tidak dapat mengubah parameter itu.

Setiap run ID asal, hash checkpoint/sumber, data hash, diagnosis, hasil
kawalan dan batasan disimpan dalam satu MLflow audit run. Tiada training,
retuning, penilaian test, atau pengubahsuaian run asal. Jika witness lulus
dan controls sepadan, kesimpulan ialah representasi ini mencukupi untuk gate
toy khusus ini dan outcome latihan bergantung pada initialization/objective.
Audit ini tidak membuktikan sebab tepat kegagalan atau mencukupi untuk bahasa
sebenar, THRML sampling, atau chip.
