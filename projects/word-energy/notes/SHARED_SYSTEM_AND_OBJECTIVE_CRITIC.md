# Sistem bersama dan kritik objective: 21 November 2026

## Apa yang dibina

`ising_shared_correlations` mengira korelasi tepat untuk satu sistem Ising berbentuk bintang: satu target di tengah, semua perkataan konteks dan semua candidate sebagai daun. Setiap token mempunyai empat spin. Semua candidate hadir serentak dan mempengaruhi posterior target yang sama. Ini pelaksanaan rujukan klasik menggunakan penjumlahan exact, bukan THRML dan bukan chip fizikal.

Jika ada `m` token konteks dan `K` candidate, sistem mempunyai `4(1+m+K)` spin dan `6(1+m+K)+4(m+K)` edge. Darjah maksimum spin pusat ialah `3+m+K`. Setiap spoke menggunakan coupling `g/(m+K)`; oleh itu menambah atau membuang candidate mengubah coupling dan boleh mengubah semua skor. Bandingkan ranking hanya apabila set candidate dan peraturan normalisasi yang digunakan dinyatakan. Saiz, degree dan kos sistem besar belum dibuktikan sesuai pada chip.

Ujian mengesahkan permutation equivariance candidate, repeated token, validation ID dan kes coupling sifar. Ujian exact 65,536 keadaan dan fixture 16-spin dengan perubahan ranking `±512/10045` menunjukkan sistem *mampu* menghasilkan corak bergantung pada target dan konteks apabila parameternya dipilih secara manual. Ini bukti kapasiti pada fixture, bukan bukti training telah mempelajarinya atau bahasa difahami.

## Kritik loss dan bukti

Critic bebas GPT-6.2 Sol ultra digunakan kerana GPT-6.1 Sol ultra mencapai usage limit. Ia menyemak code, data dan formulasi. Ia menolak cadangan D, C dan target-foil G sebagai objective yang sudah terbukti sejajar dengan ranking raw R: contoh balas tepat menunjukkan G boleh positif walaupun ranking raw R terbalik, termasuk ketika peranan target dan foil diseimbangkan. Skor loss yang baik sahaja tidak membuktikan target digunakan dengan cara semantik yang diinginkan. Tiada objective ini dipilih untuk training.

Semakan Lean sementara menggunakan Lean 4.36.0 dan Mathlib `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`, `lake env lean --stdin`: empat lemma kecil menyemak sifat checkerboard dan contoh balas foil; tiada `sorryAx`. Aksiom standard digunakan oleh lemma additive, manakala checkerboard Boolean axiom-free. Bukti tersebut memeriksa pernyataan yang dikodkan, bukan seluruh implementasi atau generalisasi model.

Critic juga mengesahkan kiraan SWORDS dan CoInCo di [audit SWORDS](SWORDS_CROSSED_SUPPORT_AUDIT_RESULTS.md) dan [audit CoInCo](COINCO_SOURCE_OVERLAP_AUDIT_RESULTS.md). Lean tambahan memeriksa implikasi teori set bagi 19 subset daripada 20 sumber dan contoh bahawa ID sumber berbeza boleh mempunyai kandungan sama; ia tidak membuktikan metadata sebenar atau kebebasan dokumen.

## Keputusan dan langkah seterusnya

Keputusan eksperimen sebelum ini: pada 132 target SWORDS development yang informatif, AUC Ising raw R `0.5813` berbanding baseline vector `0.6227`; perbezaan Ising context tolak uniform `+0.00765`, bootstrap CI `[-0.01756, 0.02445]`. Ini data development yang sudah diperiksa berulang kali, bukan held-out final test. Keputusan tidak membuktikan konteks membantu secara konsisten dan tidak boleh dibandingkan terus dengan metrik `0.7` daripada tugasan atau split lain.

Langkah seterusnya ialah membina atau mengenal pasti panel development sumber berasingan dengan label manusia yang menyokong perbandingan target dan context yang dipadankan, kemudian membandingkan shared-star Ising dengan vector yang mampu mewakili interaction pada contoh sama. Semak dahulu sama ada konteks yang model lihat benar-benar cukup untuk menyokong label. Jangan menganggap candidate yang tidak disebut oleh annotator sebagai negatif. Simpan official test untuk protokol akhir. Jika data padanan tidak mencukupi, laporkan limitation daripada mereka-reka label. Gaussian kekal baseline klasik yang memerlukan sistem joint sendiri; pengesahan THRML berbanding exact dan pelaksanaan chip ialah fasa berasingan.
