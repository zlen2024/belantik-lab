# Cadangan latihan berasaskan perubahan konteks untuk R

Status cadangan, belum dibekukan atau dilatih. **Critic menemui dua contoh balas pada objective asal dan variasi target-removal di bawah. Jangan anggap kedua-duanya mengenal pasti makna target.** Tujuannya menjawab keputusan `SWORDS_ACTIVE_R_PROBE_CRITIC.md`: correlation R dalam sistem aktif adalah betul secara matematik tetapi checkpoint semasa belum menunjukkan manfaat konteks pada penilaian pembangunan. Ini bukan cadangan bahawa SWORDS, Wikipedia cooccurrence dan semantic substitution adalah perkara yang sama.

## Satu perubahan yang hendak diuji

Daripada menggunakan R hanya selepas melatih normalized free-energy/full-softmax, cuba latih model terhadap perbezaan R yang datang daripada kandungan konteks. Bagi window latihan `(target, context, candidate_positive)`, aktifkan target, context dan candidate dalam star Ising dan hitung `R_real`. Hitung `R_uniform` pada graph dan spoke normalization yang sama, tetapi gantikan context dengan unary uniform. Gunakan `D = R_real - R_uniform`. Candidate negatif diambil daripada jadual noise latihan sedia ada. Loss cadangan bagi setiap positif-negatif ialah `softplus(margin - D_positive + D_negative)`.

Latihan ini masih menggunakan cooccurrence dan noise sebagai supervision, bukan label sense atau contoh sinonim. Negative noise mungkin benar dalam konteks. `D` ialah kontras dua observable R daripada dua window, dan bukan raw Ising Hamiltonian atau korelasi daripada satu equilibrium. Raw R, uniform control dan D mesti dilaporkan berasingan. Auto-differentiation berlaku melalui pengiraan exact 16-state factors pada CPU; ia bukan gradient melalui THRML sampling. Jika D positif mengatasi noise, itu menunjukkan model membezakan contoh dalam objective tersebut, belum membuktikan relevan semantik atau multi-well.

## Kawalan yang dicadangkan

Gunakan input, corruption/noise, urutan batch, seed, bilangan update, optimizer dan jadual yang sama untuk empat-spin Ising dan tied vector10. Bagi vector, candidate score kontekstual boleh menggunakan `cos(v_target+v_context,v_candidate)` ditolak `cos(v_target,v_candidate)` dengan raw cosine controls turut direkod. Memandangkan fungsi R dan cosine berbeza, perbandingan tidak memadankan keluarga fungsi atau kos. Tambah frequency baseline dan checkpoint lama yang dibekukan, bukannya menyesuaikan readout setiap kali hasil berubah.

Perhatikan bahawa `D_uniform = 0` secara definisi. Kejayaan D mengatasi D_uniform sahaja akan menjadi perbandingan tautologi, bukan bukti bahawa konteks semantik berguna. Outcome mesti merangkumi metrik yang tidak ditakrifkan oleh loss itu sendiri: perbezaan raw active R dengan uniform-context raw R pada calon/label natural-context, prestasi calon semantic berbanding vector dan baseline context-only, dan outcome pada konteks/target yang berbeza. Laporkan juga konteks sebenar versus konteks shuffled dan kesan pada perkataan jarang. SWORDS development telah digunakan dan tidak boleh menjadi test bebas.

## Sebelum dijadikan protokol eksperimen

1. Sahkan secara algebra/finite differences bahawa D dan gradient benar untuk semua parameter, repeated token IDs, positif/negatif collisions dan coupling sifar. Bandingkan R dengan explicit enumeration serta conditional formula pada contoh kecil. Tetapkan margin, initialization, tanda, sample budget, model states dan stop rule sebelum training, bukan selepas melihat metrik.
2. Audit sama ada objective D memberi signal wujud bagi initialization sedia ada dan tidak membenarkan jalan pintas konteks shuffling, frequency, token identity atau cancellation. Bukti satu signal pada satu contoh tidak membuktikan trainability.
3. Tetapkan satu endpoint natural-context dan kumpulan sumber, unit ketidakpastian, calon serta liputan. Lapor sokongan bersama dan populasi tambahan jika kamus diperluas. Kekalkan data ujian rasmi tertutup.
4. Gunakan semula kamus 2,000 dahulu untuk memisahkan perubahan objective daripada vocabulary dan sampling; ini bukan keputusan bahawa 2,000 cukup sebagai produk. Reka laluan OOV/coverage selepas evidence konteks, dan catat kos supaya tidak menafikan matlamat pengganti praktikal.
5. Nyatakan peranan Gaussian plan. Belum ada natural-text Gaussian joint checkpoint yang sesuai untuk comparison yang adil; jangan menganggap ketiadaan baseline itu sebagai kekalahan atau sokongan kepada Ising.
6. Jika perubahan ini menghasilkan output yang relevan, pemetaan kepada THRML mesti mengukur R untuk **setiap graph yang sama**, dan latihan berbantukan sampel memerlukan estimator gradient serta bias audit. Retain failure rare-events lama. Tiada kuasa, kelajuan atau kejayaan chip tanpa ukuran fizikal.

Critic diminta memeriksa sama ada objective ini berpotensi menjadi shortcut, sama ada vector benar-benar menerima signal yang setara dan sama ada ia mendekati matlamat, sebelum pelaksanaan training.

## Penemuan critic sebelum latihan

`D` boleh membezakan candidate melalui hubungan context–candidate walaupun unary target uniform untuk semua target. Contoh satu spin oleh critic dengan mean context +0.8, mean candidate ±0.8, target uniform dan `tanh(spoke coupling)=0.5` memberi `D=6/29` dan `D=-2/7`. Ini bukan bukti target identity digunakan. Jadual negative lama juga tidak menolak collision dengan candidate positif, jadi collision menyumbang loss malar tanpa signal ranking untuk perbandingan tersebut.

Variasi `C = D(target) - D(uniform target)` menghapuskan kes khusus target uniform tersebut. Namun critic menemui contoh dua target ID dengan unary **sama dan bukan uniform** di mana C masih membezakan candidate, dan rank C terbalik daripada raw R. Ujian `tests/test_context_contrast.py` menghasilkan semula contoh itu pada graph empat spin yang disokong, dengan koordinat lain menyumbang secara tetap. Oleh itu C belum menjamin model menggunakan perbezaan target atau menambah baik raw R. Ujian gradient terhingga dan ketepatan exact hanya mengesahkan pelaksanaan formula, bukan hipotesis atau trainability. **Tiada objective dipilih untuk training daripada cadangan ini.**
