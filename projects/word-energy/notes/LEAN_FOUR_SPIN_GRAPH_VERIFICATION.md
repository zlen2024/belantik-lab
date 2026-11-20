# Lean 4: semakan graph 4-spin

20 November 2026. Pembetulan: Lean memang telah dipasang pada `C:/Users/MuhamadDanielHakimiB/.elan/bin`, tetapi direktori itu tiada dalam PATH sesi. Laporan terdahulu bahawa Lean tidak ditemui merujuk kepada semakan PATH ketika itu; ia bukan bukti bahawa Lean tidak dipasang.

## Apa yang dibuktikan

Model simulator dalam `scripts/audit_thrml_four_spin_joint.py` menggunakan 12 spin dan 26 edge; setiap satu daripada tiga perkataan menyumbang K4, dan lapan edge menghubungkan pusat dengan konteks. `WordEnergy.schedule_valid` menyatakan bahawa senarai 26 edge yang dimodelkan mempunyai vertex 0–11, semua 12 spin diberi warna 0–3, dan kedua-dua hujung setiap edge mempunyai warna berlainan. Ini bermaksud pembahagian kepada empat block bebas yang dinyatakan selamat dari sudut edge graph.

`WordEnergy.center_needs_four_colors` menyatakan empat vertex yang semuanya saling bersambung tidak boleh diberi tiga warna berbeza pada setiap edge. Digabungkan dengan jadual yang sah, graph yang dimodelkan memerlukan tepat empat warna. Ia bukan tuntutan tentang mixing atau kelajuan.

Padanan sumber: 26 edge dan warna untuk 12 vertex Lean dibandingkan dengan senarai `EDGES` dan `BLOCKS` dalam Python, melalui `scripts/audit_lean_four_spin_graph.py`. Ini ujian padanan konkrit, bukan pembuktian formal bahawa Lean dan Python sentiasa sepadan selepas perubahan.

## Pengesahan

- Lake root: `projects/word-energy/theory`; pin `leanprover/lean4:v4.36.0`, Lean commit `611c117bfaf7b7ae1cfcf246ce6fba31e8f509e9`; Lake `5.0.0-src+611c117`.
- Tiada Mathlib digunakan atau dipasang; kedua-dua theorem menggunakan Lean core. Projek ini tidak mengesahkan kesiapsediaan Mathlib. Tiada dependency luar dalam manifest.
- Dari root Lake, tambah `.elan/bin` ke PATH sesi, jalankan `lake build WordEnergy.FourSpinGraph`: lulus.
- Jalankan `lake env lean AxiomAudit.lean`: kedua-dua declaration dipaparkan dan `#print axioms` melaporkan kedua-duanya tidak bergantung pada sebarang axiom. Tiada `sorry` pada bukti.
- SHA-256 sumber `WordEnergy/FourSpinGraph.lean`: `D428EC0D0B2B3D4895B8A4C032FDD9709F39727AD99C78A8D599F8A56BA52495`.
- Dari projek Word Energy, `.venv/Scripts/python.exe scripts/audit_lean_four_spin_graph.py` menyemak padanan terhadap kod simulator.

Ini bukti bagi graph dan pernyataan yang dikodkan, ditambah ujian padanan Python, bukan pengesahan seluruh implementasi THRML, kebetulan taburan, ketepatan sampling, pembelajaran makna, atau chip fizikal. Gate sampling terdahulu kekal gagal. Identiti covariance untuk membandingkan skor joint dan additive belum diformalkan atau dibuktikan; ia memerlukan kerja seterusnya dan mungkin Mathlib.
