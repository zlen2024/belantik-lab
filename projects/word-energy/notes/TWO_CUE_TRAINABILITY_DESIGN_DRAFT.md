# Draf trainability untuk dua cue dan anchor

Status draf untuk diaudit, belum dibekukan dan belum dijalankan. Ini cadangan menjawab jurang dalam `TWO_CUE_FIXTURE_CRITIC.md`, bukan demonstrasi model bahasa. Pertahankan objective cooccurrence/corruption asal pada pasangan token yang diberi sebagai positif dan negatif, dengan graf joint bersama bagi setiap window. Jangan menyuntik target posterior parity atau label sense sebagai loss baharu.

## Sumber maklumat yang ingin diuji

Gunakan target ambigu `T`, dua token anchor `S0` dan `S1`, empat keluarga cue `A0`, `A1`, `B0`, `B1` dengan beberapa ID dalam setiap keluarga, dan satu atau lebih distractor. Bagi setiap pasangan cue A_i/B_j, label generator `y = i xor j` digunakan hanya untuk **menjana cooccurrence**. Sediakan `(S_y, A_i, B_j)` sebagai window positif dan `(S_(1-y), A_i, B_j)` sebagai corruption berlabel oracle. Sediakan `(T, A_i, B_j)` bagi setiap sel tanpa perbezaan label. Sertakan contoh distractor dan corruption khusus untuk memberi signal pada T; ia mesti dinyatakan secara tepat sebelum benchmark dibekukan. Pada inference, model hanya menerima IDs, tanpa atribut A/B atau y.

Anchor-map untuk membaca keadaan T dibina daripada unary S0/S1 selepas latihan, menggunakan algoritma train-anchor sedia ada. Identiti dua anchor dan association mereka kepada label diketahui oleh protokol evaluator; ini eksperimen oracle-assisted dan bukan unsupervised sense discovery. Oleh kerana target T muncul dalam semua konteks, tentukan sama ada objective dan susunan positif/negatif benar-benar memberi T signal untuk mewakili kedua-dua anchor. Ini perlu diaudit dari gradients dan kawalan lexical, bukan diandaikan.

## Cadangan split kombinasi

Empat ID cue bagi setiap nilai A atau B. Dalam setiap kombinasi A_i/B_j terdapat 16 pasangan ID, gunakan `(index_A + index_B) mod 4` untuk memisahkan lapan pasangan train, empat validation dan empat reserved. Oleh sebab ID boleh muncul dalam lebih daripada satu split, ukuran ini ialah generalisasi gabungan token yang telah dilihat, bukan generalisasi token baharu. Sebelum run, verify pasangan tidak bertindih, setiap cue ID muncul dalam setiap split, setiap nilai cue dan label seimbang, kedua-dua urutan dirawat sama kerana sistem star simetri, dan seluruh input unik disimpan sebagai manifest dengan hash. Pertukaran nama token harus mengekalkan split dan metrik jika seed ditetapkan dengan betul.

## Perkara yang mesti diselesaikan sebelum protokol dan latihan

1. Tentukan satu objective positif/corruption lengkap untuk T, anchor dan distractor. Bina perbandingan oracle chance/frequency, single-cue dan target-only, dan audit bahawa tiada satu cue atau peranan sahaja dapat menghasilkan label. Jelaskan sama ada skor positif dan negatif membocorkan label kerana center ID.
2. Periksa representability dan gradient pada contoh kecil tanpa memilih checkpoint berdasarkan validation. Perbezaan label seimbang sahaja belum menunjukkan label boleh dipelajari dari objective atau identify region target.
3. Beri semua arm window, corruption dan budget yang sepadan. Bandingkan Ising joint dengan Ising additive dan coupling-zero, serta baseline vector non-linear yang boleh memodelkan hubungan dua cue. Gaussian joint versus frozen lexical memerlukan formulasi skor dan budget sendiri yang dinyatakan. Parameter count, komputasi dan supervision mesti dilaporkan.
4. Tentukan sense region tanpa label validation, per-sense accuracy/gain dan kriteria yang bermakna sebelum training. Simpan data, konfigurasi, tiga seed dan keputusan, termasuk kegagalan, dalam MLflow. Reserved split kekal tertutup sehingga satu protokol dan checkpoint dibekukan.
5. Uji seterusnya pada bahasa semula jadi dan audit observable THRML yang sama. Synthetic cue parity semata-mata tidak memenuhi sasaran pengganti embedding atau bukti chip.
