# Dua audit sumber maklumat latihan

21 November 2026. Kedua-dua audit mengira data simbolik secara deterministik; tiada model dilatih dan tiada pencapaian semantik atau chip ditunjukkan. Sumber label dan corruption adalah oracle sintetik.

## Draf dua petunjuk asal

Split mempunyai 32 pasangan cue train, 16 development dan 16 reserved, seimbang bagi empat sel, dengan 16 ID cue muncul pada setiap split dan tiada pasangan bertindih. Mengikut 32 window positif T, setiap cue tunggal mempunyai label seimbang. Label sense T boleh diterbalikkan tanpa mengubah input T itu sendiri, tetapi window anchor masih membawa supervisi parity. Ini menunjukkan input T sahaja tidak menentukan label sense, bukan bukti keseluruhan sistem dengan anchor mustahil mempelajarinya.

Kesemua 32 contoh negative lexical naif yang menggantikan anchor dengan anchor bertentangan bertembung dengan pasangan anchor–cue yang wujud positif di tempat lain. Itu hanya membuktikan pasangan tunggal tersebut tidak sah dijadikan negative lexical dalam reka bentuk seimbang ini; window penuh yang mengandungi kedua-dua cue masih boleh menjadi corruption yang sah. Jika contoh T-positif dibanding dengan distractor D yang tidak pernah menjadi pusat positif, model boleh mengenal pasti positive/negative hanya daripada identiti pusat. Ini audit shortcut yang mungkin, bukan ukuran prestasi model Ising.

Run MLflow `0c6cc727385b4479928b6225f3ca35c1`. Protokol dan skrip dibekukan dalam commit `a30f4ab`.

## Calon bridge empat token

Dalam cadangan sintetik kedua, dua target T0/T1 dan dua cue bersama menentukan anchor positif dengan peraturan `S_(a xor b xor t)`. Contoh salah menukar anchor sahaja. Audit mendapati 64 window train dan 32 development, dengan 16 cue ID dan kedua-dua target diliputi. Tiada pasangan cue bertindih antara split dan tiada collision positive/negative penuh. Semua lapan sel atribut sama banyak dalam split masing-masing. Untuk data seimbang yang ditakrifkan, entropy label bersyarat mana-mana satu atau dua daripada target, cue A dan cue B ialah 1 bit; apabila ketiga-tiganya diketahui ia 0 bit. Identiti pusat semata-mata mendapat 0.5 pada klasifikasi positive/negative seimbang.

Ini menunjukkan **maklumat tentang target dan kedua-dua cue wujud dalam label generator**, bukan bahawa objective akan mengajar representation target, bahawa sense region boleh dikenal pasti, atau bahawa model bertenaga boleh menyelesaikan tugasan. Ia juga menggunakan anchor label sintetik dan memerlukan empat token aktif, berbeza daripada kod tiga-token semasa. Perlukan scorer F4, kontrak graf, audit gradient, kawalan matched dan protokol latihan sendiri sebelum run model. Permintaan konteks empat token sahaja tidak membuktikan keserasian hardware. Run MLflow `646ddb48a81d464f9f097ee66a8b4643`, commit protokol `48d7b37`.

Kedua-dua data development dan reserved berkongsi ID cue train dan grammar yang direka. Ia menguji gabungan cue yang belum dilihat, bukan generalisasi ke perkataan baharu atau bahasa manusia. Audit reserved terhad pada kiraan imbangan dan liputan, tiada model atau prediksi reserved dinilai.
