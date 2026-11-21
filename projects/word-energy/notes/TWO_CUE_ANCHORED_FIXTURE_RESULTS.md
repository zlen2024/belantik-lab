# Hasil fixture dua lembah dan anchor

21 November 2026. Semua parameter target, cue dan dua anchor ditetapkan manual sebelum run. Anchor class 1 ditetapkan dengan coupling negatif dan class 0 dengan coupling positif. Perbandingan taburan unary anchor menghasilkan rantau class 1 dengan lapan state dan tiada ties; label empat konteks tidak digunakan untuk memilih rantau itu. Anchor dan perkaitan class mereka tetap diberi oleh pereka, maka ini oracle, bukan sense yang dipelajari.

Target mempunyai dua strict one-spin-flip minima, `[-1,-1,+1,+1]` dan `[+1,+1,+1,+1]`. Posterior target untuk cue (0,0), (0,1), (1,0), (1,1) memberikan class-1 probability berturut-turut 0.38463, 0.51748, 0.51748, 0.38463. Threshold 0.5 memberi 4/4 parity; margin minimum hanya 0.01748. Tanpa coupling, dan juga apabila hanya salah satu cue aktif pada edge strength yang sama, nilainya 0.45017. Independent 4,096-state enumeration dan produksi sepadan dengan ralat maksimum 5.56e-17. Nonlinear cue-product vector juga 4/4 dan additive threshold mempunyai had 3/4 pada empat sel dengan klasifikasi strict.

**Kedua-dua minima target berada dalam rantau anchor class 0.** Oleh itu fixture ini belum menunjukkan dua lembah masing-masing mewakili dua sense. Semuanya dipilih manual, dan tiada latihan atau generalisasi bahasa diukur. Anchor diberi berasingan daripada konteks penilaian tetapi sifat anchor dan label family dibina oleh pereka.

Semakan posthoc menggunakan `score_windows` pada parameter yang sama memberi normalized log-partition joint 0.63691 bagi cue sama dan 0.61017 bagi cue berbeza. Skor additive ialah 0.62363 bagi keempat-empat konteks. Pada fixture target rata terdahulu, kedua-dua skor ialah 0.62363 bagi semua konteks. Bacaan ini mengukur keserasian window, sementara sense ialah kebarangkalian rantau posterior. Perbezaan ini belum menunjukkan objective latihan sebenar boleh mengenal pasti label atau belajar anchor.

MLflow `word-energy-two-cue-fixtures`, anchored run `1808b653756e4c1b9b3e7c23f51ac8a5`, partition score audit `e59553b0de3746d0af6f8faff08b79e1`. Protokol dan kod dibekukan dalam commit `347f231` sebelum run. CPU, enumerasi exact, tiada ujian chip.
