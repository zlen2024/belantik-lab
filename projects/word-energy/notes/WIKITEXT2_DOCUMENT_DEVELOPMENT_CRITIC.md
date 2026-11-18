# Semakan bebas Sol: WikiText-2 dipisahkan mengikut artikel

18 November 2026. GPT-6.1 Sol Ultra mengkritik protokol dan kod sebelum latihan, kemudian melakukan audit read-only selepasnya.

- Penghurai dan sampler NumPy bebas membina semula source hashes, 600/60 artikel, vocabulary/counts/q daripada train, semua pasangan/posisi/ID artikel, tiga noise tables/orders, validation noise dan shuffle. Semua hash arrays sepadan. Section headings dikecualikan; tiada pasangan melintasi perenggan, artikel atau OOV di tengah.
- Kesemua 18 run dan 72 checkpoint diperiksa bersama source/config, hashes dan paired initializations. Direct 16-state joint-partition Ising dan vector dot mengira semula setiap endpoint: beza maksimum AUC micro, MRR dan macro artikel = 0; objective 2.22e−16. Kesemua interval 1,000 bootstrap × 60 artikel sepadan.
- Purata AUC micro original tied/role/vector = 0.6747083/0.6772467/0.67842; macro = 0.6740462/0.6792691/0.6804374. Frequency-only micro/macro = 0.6477175/0.64816275. Semua model dan seed mengatasi frequency-only dan shuffle pada micro AUC.
- Macro original−frequency: tied +0.0258834, role +0.0311063, vector +0.0322746. Hasil menyokong manfaat observed-pair associations pada 60 artikel validation dalam prosedur ini, bukan bukti human semantic understanding.
- Tied Ising di bawah vector dalam tiga seed; role di bawah vector dalam dua. Macro tied−vector −0.0063912, bootstrap bersyarat [−0.0121411,−0.0010986]; role−vector −0.0011683 [−0.0043472,+0.0020431]. Kapasiti tidak dipadankan dan dataset development sahaja; tiada kesimpulan umum tentang kelebihan architecture.
- Satu shuffle, tiga initialization berkongsi sampel, noise dan artikel. Kedua-dua dataset berasal daripada Wikipedia; near duplicates dan historical text8 overlap tidak dikecualikan. Test WikiText dan reserved text8 tidak dinilai; tiada THRML parity atau cip fizikal.

Audit tidak menemui blocker angka atau protokol. Langkah seterusnya yang dibatasi: kekalkan keputusan sebagai bukti association pada dataset ini dan rancang perbandingan capacity-matched atau dataset domain lain, sebelum membuka reserved test bagi protokol yang benar-benar dibekukan.

## Diagnostik critic posthoc (bukan kawalan pra-daftar)

Critic turut mengira baseline calon sahaja `log(bincount(train_context)+1) - log(q_train)` menggunakan satu pseudocount tetap. Pada validation, micro AUC = 0.648435, macro artikel = 0.6497081 dan MRR = 0.2855999. Macro purata tiga model asal melebihi nilai ini kira-kira 0.02434/0.02956/0.03073. Ini pemeriksaan tambahan terhadap penjelasan marginal frequency, bukan bukti semantik atau kawalan utama yang dipilih sebelum run. Kod diagnostik critic dilaksanakan read-only dalam sesi audit dan tidak disimpan sebagai skrip berasingan; formula dan sumber data yang diperlukan untuk menghasilkannya direkod di sini.
