# Draf strategi: initialization daripada training co-occurrence

15 November 2026. Protokol ini dibekukan sebelum sembilan run development.
Ia terpisah daripada audit correction F3 dan bukan pengulangan gate asal.

## Idea

Hipotesis: bias awal yang disediakan daripada corak co-occurrence train boleh
memberi geometri awal yang lebih berguna berbanding bias raw random, tanpa
menyediakan label family finance/river atau menyalin checkpoint donor. Ini
tidak bermakna keseluruhan pipeline bebas label: negatives/corruptions synthetic
asal masih menggunakan generator grammar/family oracle; evaluation sense map
masih menggunakan train-anchor labels. Ujian mengasingkan *initializer sahaja*.

## Draf implementation

1. Daripada **train windows sahaja**, bina matriks word×word symmetric counts
   bagi ordered pairs `(i,j)` dengan `i != j`, setiap pasangan posisi sekali;
   repeated IDs menyumbang mengikut posisi. Jangan guna validation/test windows,
   `POOLS`, family labels, token spelling, atau checkpoint seed berjaya.
2. Bentuk `C=log1p(counts)` dan center secara simetri `H C H`,
   `H=I−11ᵀ/V`. Susun eigenvalues menurun dan ambil dua eigenvector terbesar;
   jika dua eigenvalue teratas bukan positif, atau minimum gap antara eigenvalue
   1↔2 dan 2↔3 <= `1e-8*max(1, ||H C H||₂)`, berhenti dan
   rekod sebab, bukannya memilih paksi berdasarkan sense validation.
3. Tetapkan tanda setiap eigenvector secara deterministik: koordinat dengan
   nilai mutlak terbesar positif (tie: indeks token terkecil). Center setiap
   koordinat atas vocabulary, bahagikan population standard deviation
   (`ddof=0`, berhenti jika <= `1e-12`) dan darab 0.1
   untuk raw bias dua spin. Simpan matriks transform/hash dan eigenvalues.
   Skala 0.1 menyamai *skala nominal* normal initializer asal, bukan jaminan
   taburan atau jumlah informasi yang sama.
4. Kekalkan raw internal dan interaction initialization, data, schedule, F2,
   corrected F3 (`normalize_spokes=True`), optimizer, negative dan budgets asal.

Tiga arms dengan seed berpasangan dan endpoint lexical400 + joint100. **Joint
bermula daripada fixed lexical400** bagi semua arms, bukan checkpoint lexical
yang dipilih dengan validation. Trainer sekarang rewind kepada lexical-selected,
jadi implementation perlu menyediakan handoff eksplisit tanpa mengubah default
legacy. Uji bahawa handoff bit-identical dengan snapshot lexical400.

- **Random:** initialization asal sebagai baseline.
- **Spectral:** bias daripada train co-occurrence di atas.
- **Row-permuted spectral:** permutasikan seluruh baris bias spectral kepada
  word IDs menggunakan `np.random.default_rng(6061).permutation(V)` yang sama
  untuk semua seed; simpan permutation tepat sebelum training. Ia mengekalkan nilai/pasangan
  bias tetapi memecah padanan token–corpus. Ini control, bukan model label-free
  yang setara informasi dalam semua aspek.

Gunakan tiga seed baru berpasangan `101,103,107` sebagai *development seeds*
(belum ditemui dalam tag MLflow pada masa draf). Internal/interaction RNG sama
antara arms bagi seed yang sama. Kekalkan random-bias draw dan PRNG consumption
yang sama walaupun bias diganti, supaya beza tidak mengubah RNG lain. Tiada
adaptasi skala/seed/negatives berdasarkan hasil. Jumlah 9×(400+100)=4500
optimizer updates; target <=30 minit dan process peak working set <=2 GiB,
environment OMP/OPENBLAS empat thread. Ini bukan hard OS limit atau ukuran peak
seluruh child processes. Hentikan jika limit diukur terlampaui, nonfinite
atau audit gagal; kekalkan artifacts yang sempat selesai.

## Verification sebelum training

- API initializer menerima hanya matriks token training dan frozen training
  vocabulary. Ujian output invariant apabila validation/test contents atau
  family labels berubah tanpa menukar train input; test row permutation serta
  hashes train input, token map, count matrix, eigenvectors dan final biases.
  Tolak eigenvalue ties/degenerated std; tidak menyembunyikan
  kegagalan dengan label-aware fallback.
- Reproduce baseline legacy secara berasingan; pastikan corrected F2/F3 telah
  lulus known-answer enumeration/gradient tests. Lakukan snapshot audit bahawa
  hanya bias initializer berbeza; semak shared schedule/data hashes.
- Lapor fixed lexical400 dan joint100 bagi **semua** seeds/arms; family counts,
  sense/gain, anchor/context TV, retrieval AUC/loss, numerical health, cost dan
  failures. Kira jumlah seeds yang memenuhi thresholds sense/AUC/gain asal bagi
  setiap arm, tanpa memilih seed yang berjaya. Jika membandingkan selected checkpoints juga, tetapkan selector asal
  sebelum run dan jangan menggantikan fixed endpoint.
- Jangan buka test yang pernah dilihat untuk tuning. Ini development experiment;
  generalisasi memerlukan dataset/split baharu yang belum disentuh dan lebih
  daripada tiga seed, sebelum sebarang claim superiority atau hardware.

Laluan ke THRML tetap memerlukan perbandingan simulator terhadap exact untuk
model learned; draf ini hanya melatih/menilai classical reference.
