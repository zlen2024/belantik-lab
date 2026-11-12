# Kajian ringkas: bagaimana Word2Vec diskalakan

Disediakan 10 November 2026 untuk projek landscape tenaga. Sumber utama: Mikolov et al., [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/html/1310.4546), terutama §2–3; dan [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).

## Apa yang penulis buat

1. **Skip-gram:** gunakan perkataan untuk meramal perkataan yang berhampiran. Wakilan ialah vektor, dan prestasi vektor dinilai dengan tugas analogi. Ini belajar daripada corak penggunaan, bukan label makna yang diberi manusia. [Mikolov et al. 2013 §2, §3](https://arxiv.org/html/1310.4546)
2. **Negative sampling:** bandingkan pasangan yang diperhatikan dengan beberapa contoh daripada taburan noise. Paper menyatakan 5–20 negatif berguna untuk dataset kecil dan 2–5 untuk yang besar. Taburan noise unigram berpangkat 0.75 berprestasi lebih baik dalam ujian penulis berbanding pilihan lain. Negatif sampel tidak dijamin salah dari segi semantik. [§2.2](https://arxiv.org/html/1310.4546#S2.SS2)
3. **Subsampling:** perkataan sangat kerap dibuang secara rawak dengan kebarangkalian yang bergantung pada kekerapannya; paper memberi threshold lazim sekitar 1e−5. Ini mengubah taburan contoh latihan dan mengurangkan dominasi `the`, `in`, `a`. [§2.3](https://arxiv.org/html/1310.4546#S2.SS3)
4. **Banyak data dan kosa kata:** untuk sebahagian ujian, penulis menggunakan korpus berita dalaman sekitar satu bilion perkataan dan membuang kata dengan kekerapan <5, menghasilkan sekitar 692 ribu kata. Rajah 2 menggambarkan vektor 1,000 dimensi selepas diproyeksikan ke 2D; ini tidak bermaksud semua eksperimen menggunakan dimensi itu. [§3 dan Rajah 2](https://arxiv.org/html/1310.4546)
5. **Penilaian analogi:** contoh `Berlin − Germany + France ≈ Paris`, menggunakan jarak cosine dalam ruang vektor. Ini ujian tertentu, bukan bukti semua makna dipelajari atau sense `bank` dipisahkan. [§3](https://arxiv.org/html/1310.4546#S3)

## Apa yang kita tiru dan apa yang berbeza

Pelan kita: 50 juta aksara train daripada text8, 10,000 kata paling kerap (min_count 5, kemudian cap), 100 dimensi, 1 juta pasangan tersampel tetap yang diulang 3 epoch, window maksimum 5, 5 negatif, dan q∝count^0.75. Kata kerap disubsample sebelum membina pasangan train. Ini lebih besar daripada run asal kita (10 juta aksara train, 2,000 kata, 64 dimensi, 200 ribu pasangan), tetapi **jauh lebih kecil** daripada korpus utama paper.

Kita gunakan PyTorch CPU dengan SparseAdam untuk embedding dan Adam untuk MLP, bukan implementasi C/SGD asal. Kita sampling satu context bagi satu window rawak, bukan memproses semua pasangan seperti konfigurasi C asal. Train disubsample tetapi validation/test tidak; ini boleh mengubah penilaian ranking. Oleh itu run kita ialah **Word2Vec-inspired**, bukan reproduksi yang setara.

Penilaian analogi diperoleh daripada [questions-words.txt di repo asal](https://github.com/tmikolov/word2vec/blob/master/questions-words.txt); kita laporan bilangan soalan yang semua kata ada dalam vocab dan accuracy bersyarat. Analogi menguji vektor input, bukan secara terus energy neural atau perbezaan makna berdasarkan konteks. Teks8 ada kekangan preprocessing dan tiada label deria. Semua model dan sumber data berada pada CPU tempatan; tiada dakwaan prestasi cip.
