# Critic: joint-context WikiText dan WiC berulang

19 November 2026. GPT-6.1 Sol Ultra mengkritik reka bentuk sebelum run dan mengaudit hasil secara berasingan selepas run.

## Experiment critique

- Semua 15 dataset arrays dibina semula daripada source: train vocabulary/counts, sampled occurrence IDs/positions, negatives, orders dan shuffle. 200,000 train windows mengandungi 113,065 unique centers dan 143,233 unique occurrence windows; validation 5,765 unique centers dan 7,162 windows merentas 60 artikel. Ini resampling daripada pasangan sedia ada, bukan center uniform baharu dari seluruh corpus.
- Semua 48 checkpoint dan 184,826 unique validation candidate triples **bagi setiap checkpoint** dikira dengan direct 64-state joint / 16-state additive enumeration atau vector arithmetic. Beza score maksimum 7.55e−15, beza metrik/objektif maksimum 4.44e−16, diagnostics ≤5.03e−16. 13 source snapshots identik, paired Ising initializations dan CPU/thread settings sepadan. Article bootstrap sepadan.
- Joint−additive macro AUC +0.000908 [−0.000884,+0.002641]; joint−vector −0.007674 [−0.011410,−0.004147]; joint−shuffle +0.045779 [+0.036743,+0.053715]. Ketidakpastian bootstrap bersyarat pada data/model yang sama. Joint MRR lebih tinggi sedikit daripada additive untuk semua tiga seed, purata +0.001660; jangan kata tiada peningkatan pada semua metrik.
- Center posterior berubah dari unary dengan TV sekitar 0.20, dan purata mutlak score joint−additive sekitar 0.14 apabila menggunakan parameter sama. Jadi implementasi bukan secara tidak sengaja additive atau mengabaikan konteks. Semua coupling Ising original >1.49 daripada had 1.5; saturation diperhatikan tetapi bukan bukti menaikkan had akan membaiki makna.

## WiC berulang

Subset 167/638 dan 151 target dibina semula. Enumeration 4–10 spin menghasilkan semua skor Ising dan entropy dalam 1.03e−15; cosine vector 70-digit menghasilkan AUC sama, dan bootstrap target berpasangan sepadan. Joint AUC 0.543615 [0.468665,0.624694]; initial 0.548924; additive 0.574701 [0.502471,0.653924]; CBOW 0.555189. Additive interval sedikit di atas 0.5, jadi jangan kata semua interval meliputi chance. Joint−additive −0.031086 [−0.054625,−0.010192], namun ini validation yang pernah dilihat, dengan beberapa perbandingan, liputan terhad dan shift context cardinality; bukan keputusan confirmatory atau bukti generalisasi sense. Joint−initial dan joint−CBOW tidak pasti.

## Hypothesis critique dan langkah dibatasi

Latihan joint mempelajari association center/konteks, tetapi kelebihan AUC dan manfaat sense untuk architecture shared-center minimal tidak ditunjukkan. Tidak ada ralat angka, sampel atau leakage yang ditemui sebagai penjelasan. Kapasiti empat state, coupling bounded, bagged contexts dan objective cooccurrence yang tidak mengajar sense secara langsung ialah hipotesis alternatif, bukan sebab yang sudah dipastikan.

Tindakan seterusnya: rekodkan cabang ini sebagai hasil campuran/negatif, kekalkan MRR improvement kecil, dan semak semula hipotesis sebelum menambah vocabulary/spin count atau memindahkan model ke hardware. Semua reserved tests kekal tertutup; tiada THRML parity untuk six-spin graph terlatih ini dalam run ini.
