# Semakan pelan Ising/Z1

Critic: GPT-6.1 Sol, reasoning ultra. Skor untuk mutu pelan, bukan bukti pelaksanaan atau prestasi hardware. Sasaran review ≥9.5/10 tanpa blocker; kategori berwajaran tetap: idea20%, matematik25%, latihan/evaluation25%, hardware20%, provenance10%.

## Pusingan 1 — 7.4/10

Snapshot [round_01_plan.md](round_01_plan.md), SHA256 `AB000351A13086D5149D1FE6C79518B6EF928839AA72D241CB8A15FF5346D5EB`.

| Kategori | Skor /10 |
|---|---:|
| Idea |9.5|
| Matematik/readout |8.0|
| Training/evaluation |4.5|
| Hardware |7.0|
| Provenance |9.5|

Blockers: objective dan gradient untuk data word ID belum ditakrifkan; konfigurasi Gaussian lama tidak terus terpakai untuk Ising; toy sanity disamakan dengan superiority terhadap vector. Isu lain: fixture asal tidak menilai relevance, degree bergantung pada graf token, dan adjacency mapping perlu injection atau minor embedding yang sah, bukan sekadar path.

Penulis menambah objective ranking Ising normalized unary, exact gradient, kontrak D/L/H/G/K/parameter/training, sense map dan baseline, degree budget dan syarat graph mapping. Semakan tambahan menemukan fixture konteks sama kuat memberikan skor relevance sama; dicatat bersama fixture tiga konteks dalam `distractor_fixture.py/json`. Kedua-duanya ditetapkan manual dan bukan eksperimen learned.

## Pusingan 2 — 9.55/10

Snapshot [round_02_plan.md](round_02_plan.md), SHA256 `28393C1C3B565C4AFBB5A4D119884F4207216155C1EF120D97673C14666AF328`. Fail pelan utama sepadan dengan snapshot ini semasa semakan. Pautan relatif dalam snapshot archival merujuk lokasi pelan utama.

| Kategori | Berat | Skor /10 |
|---|---:|---:|
| Idea |20%|9.5|
| Matematik/readout |25%|9.8|
| Training/evaluation |25%|9.4|
| Hardware |20%|9.5|
| Provenance |10%|9.5|

Critic mendapati tiada blocker substantif untuk pelan pilot. Ia mengira semula fixture distractor dan menguji semua11 raw parameters termasuk perkataan berulang: perbezaan maksimum finite difference `3.97e-10`; F3 dengan λ=0 ialah `−2.22e−16`. Tiada training atau THRML simulation dijalankan dalam review.

Perincian untuk ditetapkan sebelum implementasi: gain posterior gate ialah mean atas12 bank windows per seed; W=0 ialah degenerate numerical control kerana F2/F3 sifar dan ranking objective tidak melatih unary. Relevance learned dalam mixed-domain masih perlu diuji, fixture manual bukan bukti learning. Akses adjacency/backend sebenar, julat parameter dan calibration Z1 belum disahkan. Markah ini menilai pelan, bukan keserasian chip atau hasil empirikal.
