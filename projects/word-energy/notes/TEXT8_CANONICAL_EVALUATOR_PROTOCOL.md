# Evaluator canonical untuk analisis text8 baharu

17 November 2026. Evaluator historic dikekalkan untuk kebolehulangan run lama.
Sebelum eksperimen baharu, bina evaluator berasingan yang menyatukan kandidat
pendua menurut pair IDs, mengira skor setiap pasangan unik sekali, dan
menggunakan skor yang sama untuk lookup positive/noise. Untuk tied Ising F2,
`(u,v)` dan `(v,u)` juga dikanonkan kerana score fizikalnya simetri; untuk
role-specific/vector, turutan mesti dikekalkan. Tiada global epsilon yang
menyamakan skor token berlainan.

Untuk tied Ising, rujukan stabil ialah
`log1p(sum_{s,t} p_u(s)p_v(t) expm1(sum_a g_a s_a t_a))`, yang setara dengan
negative normalized F2 free energy dan memberi sifar tepat apabila g=0.
Uji nilai dan gradient berbanding enumerasi 16-state, same-ID candidates,
reversed pairs, distinct IDs dengan unary landscape identik, invariance
terhadap chunk, dan λ=0 AUC=0.5/MRR=H21/21 untuk semua kandidat.
Kawalan zero-coupling ialah keadaan model yang ditetapkan, bukan inferens
daripada score hampir sifar.

Validasi retroaktif hanya pada 12 checkpoint epoch3 asal dan validation yang
sama; bandingkan corrected exact-ID metrics terdahulu, jangan menukar model,
training atau membuka reserved test. Record hashes dan perbezaan dalam MLflow
audit baharu. Jika keputusan canonical tidak sepadan, simpan kedua-dua versi,
audit sebelum digunakan dan jangan memilih versi berdasarkan score yang lebih
baik. Ini repair pengukuran, bukan tambahan latihan atau bukti semantics.
