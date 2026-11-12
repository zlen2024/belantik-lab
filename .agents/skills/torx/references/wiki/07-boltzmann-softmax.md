# 07. Boltzmann dan softmax (notebook 14)

Halaman paling berkaitan dengan attention.

## Campuran Gaussian sebagai Boltzmann machine

Visible `v` selanjar (pmode), hidden `h` kategori satu-panas (pdit, K keadaan). Tenaga:

`E(v,h) = Σ_k h_k [ ½ Σ_i (v_i − μ_ki)²/σ_ki² + ½ Σ_i log σ_ki² − log π_k ]`

## Softmax muncul

Tetapkan `v`. Hidden hanya boleh jadi satu `e_k`, jadi tenaga bagi satu skor setiap komponen:

`p(h = e_k | v) = softmax_k(θ_k(v))`

`θ_k(v) = log π_k − ½ Σ (v_i−μ_ki)²/σ_ki² − ½ Σ log σ_ki²`.

Dengan kovarians dikongsi dan gandingan linear: `θ_k(v) = c_k + Σ_i (v_i/σ_i) W_ik` ditambah terma bebas-k. Itu **dot product antara v dan lajur W_k**.

Nota notebook: satu **pbit** bias memberi **sigmoid**, manakala **pdit** satu-panas memberi **softmax**.

## Bahagian lain

- `p(v | h=k)` ialah Gaussian: `MixtureGaussianGate`.
- Block Gibbs ditulis tangan dalam JAX: ganti sampel `h | v` dan `v | h`.
- Penetapan ahli kelompok keras melebihi 99%.
- Kekangan satu-panas ditulis sebagai tenaga winner-take-all.

## Kenapa penting

Attention = `softmax_j(q·k_j / √d)`. Bentuk yang hampir sama, tetapi **bukan** sama: `θ_k(v) = v·μ_k/σ² − ‖μ_k‖²/(2σ²) + log π_k`. Dengan `v ↔ q`, `μ_k ↔ k_j`, perlu `σ² = √d` dan bias per kunci `log π_j = ‖k_j‖²/(2√d)` untuk membatalkan terma norma. Lihat `../estimator-math.md`. Maka **memilih kunci mengikut berat attention ialah pensampelan pdit**, satu primitif asli Torx. Lihat halaman 09 untuk hipotesis dan batasan.
