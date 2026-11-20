> Pembetulan 20 November: Lean kini ditemui di `.elan/bin` dan digunakan untuk membuktikan dua sifat graph. Lihat `LEAN_FOUR_SPIN_GRAPH_VERIFICATION.md`; identity covariance masih belum dibuktikan. Pernyataan tentang ketiadaan Lean di bawah merekod keadaan semakan sebelumnya, bukan keadaan terkini.

# Kritik matematik dan langkah seterusnya: 4-spin

20 November 2026. Critic: GPT-6.2-Sol, ultra. Semakan meliputi skill research-math-audit, research-lean-formalize dan research-lean-verify; protokol dan pelaksanaan 4-spin, laporan dan ujian yang sedia ada. Keputusan eksperimen 19 November tidak diubah.

## Dapatan utama

Perbandingan joint 4-spin dengan 2-spin mengubah parameter, sambungan dan skala coupling serentak; skor validation naik 0.00773 tetapi puncanya tidak dapat diasingkan. Joint4 minus additive4 ialah -0.00272 dengan selang bootstrap artikel [-0.00535, +0.00013]; vector10 lebih tinggi 0.02539. WiC hanya 167/638 contoh development yang telah digunakan sebelum ini; 0.52994 untuk joint4 belum menunjukkan kelebihan makna. Gate simulator THRML gagal seperti direkod, bukan bukti kesilapan formula dan bukan pengesahan chip.

Skill baru membezakan pernyataan, bukti formal, implementasi dan keputusan empirikal dengan jelas. Jurang utama ialah tiada executable lean/lake/elan dalam PATH atau direktori .elan/bin, dan tiada root Lake atau lean-toolchain yang dipin dalam projek Word Energy. Maka tiada Lean declaration yang dielaborasi, Lake build atau audit #print axioms untuk tugasan ini. Jangan panggil ini bukti Lean. Contoh numerik LEAN_ZERO_COUPLING_WORKED_EXAMPLE.md ialah fixture 6-spin model lama, bukan bukti formal atau pengesahan keseluruhan model 4-spin.

## Calon tuntutan untuk diformalkan

Sumber: notes/WIKITEXT2_FOUR_SPIN_PROTOCOL.md dan word_energy/wikitext_four_spin.py. Pada ruang 16 keadaan spin pusat, ambil q sebagai taburan unary pusat yang dinormalisasi. Ambil A(s) dan B(s) sebagai jumlah berwajaran Boltzmann bagi setiap konteks, termasuk normalisasi unary konteks dan coupling spoke seperti pada kod. Definisikan J=E_q[A B], P1=E_q[A], P2=E_q[B]. Maka identiti matematik yang dicadangkan ialah J-P1 P2=Cov_q(A,B). Dengan J, P1, P2 positif, beza skor log joint dengan skor jumlah pasangan mempunyai tanda yang sama dengan covariance itu.

Interpretasi: interaksi melalui keadaan pusat yang dikongsi tidak semestinya menaikkan skor joint berbanding penjumlahan pasangan. Context messages yang menggemari keadaan pusat yang bertentangan boleh menghasilkan covariance negatif; yang menggemari keadaan sama boleh menghasilkan covariance positif. Ini pernyataan mengenai skor model pada window tertentu, bukan bukti bahawa latihan joint lebih baik atau lebih buruk pada AUC. Perlu audit pemetaan definisi, faktor spoke, unary normalization dan frequency prior sebelum membekukan deklarasi. Lemma ini kini ialah sasaran matematik untuk diaudit, BUKAN theorem Lean yang telah dibina.

## Bukti sedia ada dan batas

Ujian projek membandingkan pengiraan 4-spin dengan penjumlahan tepat 4,096 keadaan, juga ujian gradient, token berulang, pertukaran konteks dan coupling sifar. Critic eksperimen sebelumnya melaporkan audit numerik bebas; itu menyokong fixture dan implementasi yang diuji, bukan pembuktian umum. Contoh model lama diperiksa semula menggunakan `PYTHONPATH=.; .venv\Scripts\python.exe scripts/audit_zero_coupling_example.py`: empat fixture lulus, perbezaan F3 tertinggi 8.88e-16.

Lean boleh menyemak lemma tepat dengan semua andaian yang dikodkan, tetapi tidak secara automatik menjamin floating point, percampuran sampler, prestasi semantik atau hardware. Gabungan bukti matematik, ujian kod dan eksperimen terkawal tetap diperlukan.

## Satu langkah seterusnya

Sediakan dan pin environment Lean/Mathlib yang boleh dibina, kemudian audit dan cuba buktikan identiti covariance yang dipetakan kepada formula 4-spin, termasuk semakan declaration dan axiom dependencies. Simpan status walaupun setup atau proof gagal. Selepas asas itu, rancang protokol baru untuk mengasingkan punca perbezaan dengan kawalan additive/vector dan penilaian makna yang belum disentuh; jangan ubah keputusan atau gate eksperimen lama.

Critic juga menjalankan enumerasi bebas 4,096 keadaan pada contoh beta=1, g_i=0.8, internal coupling sifar dan field spin konteks pertama ±0.6. Context bertentangan menghasilkan covariance -0.07768147 dan beza log -0.04252864; context sama menghasilkan +0.07768147 dan +0.04079350. Ini ialah dua fixture numerik, bukan bukti Lean atau ramalan AUC bagi dua model yang dilatih berasingan.
