# Critic dua fixture dua-petunjuk

21 November 2026. Critic GPT-6.1 Sol ultra memeriksa protokol, kod production, artifact MLflow, snapshot dan kedua-dua pelan asal. Ia mengira semula formula dan hasil secara bebas. Semakan menyokong pengiraan fixture, dengan had berikut.

## Target rata

Semua internal coupling sifar dan unary target uniform. Dua koordinat target bersyarat adalah bebas untuk parameter ini. Definisikan `t=tanh(2)*tanh(0.4)=0.3662812786` dan tanda cue `A=2a-1`, `B=2b-1`. Posterior target ialah `(1+A*t*s0)*(1+B*t*s1)/16`, dan kebarangkalian rantau parity ganjil ialah `(1-A*B*t^2)/2`. Formula bebas sepadan dengan produksi dalam 1.39e-17. Maka 4/4 datang daripada bacaan parity bukan linear yang ditetapkan tangan pada dua marginal cue, tanpa menunjukkan interaksi kolektif target. Normalized logZ joint untuk keempat-empat kes adalah sama, `8 log cosh(0.4)=0.62362788`. Objective skor window asal tidak membezakan empat konteks ini pada parameter tersebut.

## Target dua minima dan anchor

Kedua-dua minima unary target berada dalam class 0 mengikut region anchor; mereka bukan satu minimum setiap sense. Anchor dipilih manual dan tidak dipelajari, walaupun region tidak dipilih menggunakan label empat window yang dinilai. Dengan `c=tanh(0.1)` dan `r=A*B`, exact class-1 mass ialah `(1-c)*(1-r*t^2)/(2*(1+r*c*t^2))`. Untuk cue bertentangan, mass >0.5 hanya jika `t^2>c`. Menguatkan wells melebihi kira-kira `J=0.13498` pada cue strength ini akan menggagalkan syarat tersebut. Ini menunjukkan sensitiviti margin kepada parameter, bukan kemampuan terlatih.

Normalized joint logZ ialah `8 log cosh(0.4)+log(1+r*tanh(0.1)*t^2)`, sementara additive logZ ialah `8 log cosh(0.4)`. Joint benar-benar berbeza mengikut gabungan cue pada fixture ini. Namun objective F3 asal ialah negatif logZ dan lebih menyukai cue yang sama dalam fixture. Empat label parity belum menyatakan contoh positif, corruption, atau mekanisme bagaimana model mempelajari anchor family daripada data. Hasil tidak membuktikan bahawa contrastive training sedia ada akan menghasilkan readout yang diinginkan.

## Lean 4

Dalam `projects/word-energy/theory`, critic menggunakan pinned Lean 4.36.0 dan Mathlib `fc4f59e` melalui `lake env lean --stdin`, import `WordEnergy.RealCovariance`. Kernel memeriksa `xor_no_strict_additive (c u v : ℝ) : ¬(c<0 ∧ 0<c+u ∧ 0<c+v ∧ c+u+v<0)` dengan `linarith`, dan `factorized_parity_mass (a b : ℝ) : ((1+a)/2)*((1-b)/2)+((1-a)/2)*((1+b)/2)=(1-a*b)/2` dengan `ring`. Audit kedua-dua theorem melaporkan `propext`, `Classical.choice`, `Quot.sound`, tanpa `sorryAx`. Kedua-duanya mengesahkan kenyataan formal tersebut, bukan training, kaitan semantik atau pelaksanaan hardware.

## Keputusan dan tindakan

Critic juga memeriksa `weak_well_threshold (c x : ℝ) (denom : 0 < 1-c*x) : (1/2:ℝ) < (1-c)*(1+x)/(2*(1-c*x)) ↔ c < x` dengan `nlinarith`. Audit axiom sama. Command tambahan membina `WordEnergy.FourSpinTheorem` dan menjalankan `FourSpinTheoremAudit.lean` dalam environment pinned. Claim ini ialah algebra bersyarat, bukan bukti implementasi atau eksperimen hardware.

Fixture menyokong kapasiti oracle bagi readout, bukan hipotesis bahawa model bahasa atau dua sense telah dipelajari. Tindakan seterusnya ialah menentukan protokol trainability kecil dengan positif/corruption yang khusus dan map anchor daripada training sahaja, serta kawalan factorized dan nonlinear vector dengan maklumat penyeliaan sepadan. Tanpa sumber supervisi yang mengenal pasti hubungan cue kepada anchor, task parity seimbang untuk satu token target sahaja tidak mengenal pasti label sense. Reka bentuk ini mesti ditetapkan dan diaudit sebelum sebarang run, dan hasil sintetik tidak menggantikan penilaian bahasa yang segar. Gaussian dan kesetaraan THRML bagi observable yang benar-benar digunakan kekal terbuka.
