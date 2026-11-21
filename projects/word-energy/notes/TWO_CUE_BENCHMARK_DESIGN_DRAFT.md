# Reka bentuk awal ujian dua petunjuk

Status: draf, belum dibekukan dan belum dinilai. Asalnya daripada `MATCHED_VECTOR_ALIGNMENT_CRITIC.md`. Tujuannya menguji sama ada dua token aktif bersama mengubah bacaan sense bagi perkataan yang sama. Ini ialah ujian mekanisme sintetik sebelum menyediakan bukti bahasa semula jadi, bukan ukuran prestasi pengganti embedding.

## Struktur contoh

Setiap kumpulan membentuk petak lengkap dengan dua atribut binari A dan B. Target ambigu sama dalam empat konteks. Setiap konteks mengandungi satu token cue daripada keluarga A dan satu daripada keluarga B. Label sense ditentukan oleh `A xor B`; jadual label ialah 0, 1, 1, 0. Mengubah A sambil mengekalkan B mengubah label, dan mengubah B sambil mengekalkan A juga mengubah label. Gabungan semua empat sel sama banyak, jadi satu cue, frequency dan target ID sahaja tidak meramal label melebihi chance pada set lengkap. Nama cue hanyalah simbol, bukan bukti makna bahasa.

Untuk menguji generalisasi kombinasi tanpa membuat token yang tidak mempunyai parameter terlatih, setiap nilai atribut mempunyai sekurang-kurangnya dua ID cue train-known. Pisahkan pasangan ID cue ke train dan validation supaya setiap ID berlaku pada kedua-dua split tetapi tidak ada pasangan ID yang muncul dalam kedua-duanya. Setiap sel A/B mesti seimbang dalam setiap split. Tempahan berasingan perlu disimpan tertutup, dengan sumber dan seed manifest jelas. Periksa bahawa target, cue tunggal, posisi, counts dan token IDs tidak memberi jalan pintas. Juga laporkan bahawa identiti cue dilihat semasa latihan dan keputusan tidak menguji token OOV.

Model Ising asal dan vector mean-context simetri terhadap pertukaran cue, jadi buat data yang tidak menganggap susunan cue sebagai maklumat label. Untuk mencegah falsafah ujian yang memihak kepada satu model, sertakan vector interaksi bukan linear dengan kiraan parameter dan data latihan dilaporkan serta sahkan ia boleh menyatakan jadual XOR. Sertakan prior seimbang, single-cue, Ising dengan interaksi dimatikan, Ising additive pada parameter yang sama, dan Gaussian joint versus lexical beku. Kiraan parameter Ising, Gaussian dan vector, tahap penyeliaan, loss dan masa perlu dipaparkan. Cara membaca sense mesti dinyatakan untuk setiap model sebelum fitting.

## Audit sebelum apa-apa training

1. Tentukan dengan tepat cara label melatih atau menilai model. Jika label diberikan semasa training, label itu perlu diberikan kepada semua kawalan dan hasil tidak boleh dipanggil pembelajaran sense tanpa penyeliaan.
2. Buktikan atau semak fixture bahawa label lengkap tidak boleh dicapai oleh skor additive satu-cue dengan threshold bersama, dan bahawa pembanding bukan linear boleh mencapai 4/4. Nilai ini hanya kapasiti matematik.
3. Semak sama ada joint Ising dan Gaussian yang dicadangkan mampu menyatakan kesan yang diuji di bawah skor dan graph sebenar. Jika tidak, jelaskan had itu dan nilai sama ada tugasan berkait dengan rancangan asal sebelum membelanjakan run.
4. Tetapkan metrik per-sense, kesan berpasangan minimum yang bermakna, bilangan kumpulan/seed dan split sebelum mengira sebarang outcome model. Audit leakage dan single-cue pada manifest yang dibekukan.
5. Selain fixture terkawal, rancang evaluasi pada konteks sebenar yang segar, dengan coverage dan license, kerana cue simbol tidak menunjukkan semantic transfer.

THRML perlu memeriksa observable posterior atau correlation yang sama dengan exact pada checkpoint terpilih, dengan rare-event budget dan syarat precision yang dibekukan terlebih dahulu. Menggunakan posterior bersama tidak dengan sendirinya menyelesaikan perbandingan partition function merentasi calon. Kos host dan batas graph/backend perlu direkod. Tiada dakwaan chip tanpa eksekusi fizikal yang direkod.
