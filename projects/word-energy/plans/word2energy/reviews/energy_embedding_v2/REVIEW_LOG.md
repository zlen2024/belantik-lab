# Critique pelan Word Energy Embedding v2

11 November 2026. Critic: GPT-6.1 Sol, reasoning ultra. Penulis/reviser: agent utama. Skor menilai kualiti pelan, bukan prestasi atau kebolehlaksanaan empirikal yang sudah dibuktikan. Sasaran pengguna: sekurang-kurangnya9.5/10; rubrik tetap dan tiada isu P1 terbuka.

## Pusingan 1 — 8.2/10

Snapshot: [round_01_plan.md](round_01_plan.md). SHA256 `8225000B8A45006C1AE55C8BFD0DEE47A539E0BEFE9B770D9ECCFDC5399C3047`.

| Kategori | Maksimum | Markah critic |
|---|---:|---:|
| Kesetiaan idea/skop |15|14.5|
| Matematik/kontrak model |25|22|
| Objective/gradien/inference |15|12|
| Pilot reproducible |20|12|
| Penilaian/baseline/data |15|12.5|
| Skalabiliti/Torx |10|9|
| Jumlah |100|82|

Verdict critic: seni bina asas setia dan matematik utama betul, tetapi pilot belum boleh dilaksanakan tanpa keputusan saintifik tambahan. Empat P1: batas λ/occupancy/F belum diterjemah kepada sasaran; config tidak lengkap; kos grid DL dan W tetap dalam D1; metrik sense/candidates belum operasional.

Enam P2: negatives bukan jaminan anti-collapse; gauge/identifiability; acceptance gates tidak bernombor; synthetic split dan contextual baseline; gate real-text holdout; kos dan kontrak Torx.

Revisi untuk pusingan2: menambah batas matematik, parameterization W, config pilot, data generator/split/corruption, quadrature dan schedule sampler, metric sense train-only dan candidate sets, baseline/controls, gates lulus/gagal/tidak konklusif, manifest, compute complexity dan audit diskret Torx. Hasil eksperimen masih belum ada.

## Pusingan 2 — 9.35/10

Snapshot: [round_02_plan.md](round_02_plan.md). SHA256 `AA6858BCC30322F74F606D833D23380947572FD71D9C88378E1E476326FB5ACA`.

| Kategori (rubrik sama) | Maksimum | Markah critic |
|---|---:|---:|
| Kesetiaan idea/skop |15|14.8|
| Matematik/kontrak model |25|24.0|
| Objective/gradien/inference |15|14.0|
| Pilot reproducible |20|17.5|
| Penilaian/baseline/data |15|13.7|
| Skalabiliti/Torx |10|9.5|
| Jumlah |100|93.5|

Critic mengesahkan isu matematik utama diselesaikan. Baki P1 ialah shuffled-positive control masih mengekalkan family supervision melalui negatives dan joint corruptions. Baki P2: fixture dua means disebut noncollinear secara mustahil, dan score/loss/input-table serta batch schedule baseline SGNS belum lengkap. Clarification tambahan diperlukan untuk rollback apabila quadrature refinement gagal.

Revisi pusingan3: control dinamakan khusus sebagai ujian sumbangan tambahan positive co-occurrence, bukan control tanpa supervision; baseline rawak ditambah. Fixture W dan covariance kini angka tepat. SGNS loss/input table, seeds schedule dan rollback dijelaskan.

Semakan numerik tambahan penulis: [math_spot_check.json](math_spot_check.json), fungsi ditetapkan manual sahaja. Ia bukan training atau benchmark v2; refinement F/R fixture konsisten dengan batas teori.

## Pusingan 3 — 9.6/10, ambang tercapai

Snapshot: [round_03_plan.md](round_03_plan.md). SHA256 `A69A3CFED608518790193DE0A5A189EF09723BDDA4D8A89F9CA21E92B9009372`. Fail pelan utama sepadan tepat dengan snapshot ini sewaktu semakan akhir.

| Kategori (rubrik sama) | Maksimum | Markah critic |
|---|---:|---:|
| Kesetiaan idea/skop |15|14.8|
| Matematik/kontrak model |25|24.4|
| Objective/gradien/inference |15|14.4|
| Pilot reproducible |20|18.7|
| Penilaian/baseline/data |15|14.2|
| Skalabiliti/Torx |10|9.5|
| Jumlah |100|96.0|

Verdict critic: tiada isu P1 atau P2 yang menghalang pelaksanaan pilot ditemui dalam snapshot ini. Pembetulan control, fixture W, SGNS dan rollback diterima. Loop review ditamatkan kerana ambang pengguna9.5/10 sudah dicapai dengan rubrik asal.

Dua detail kecil untuk direkod dalam config sebelum implementation run: formula frequency-only baseline dan butiran numerik root-finding/adaptive integration. Critic menyatakan kedua-duanya tidak memerlukan perubahan reka bentuk untuk kelulusan pelan.

Kelulusan ialah penilaian seorang critic terhadap mutu pelan. Tiada latihan v2 atau keputusan semantik dihasilkan dalam loop ini. Cross-check joint integral bebas, sampler, pembelajaran sense dan manfaat berbanding baseline masih perlu diuji. Versi snapshot mengekalkan bytes asal untuk audit; pautan relatif dalam snapshot mengikuti lokasi asal pelan utama.

## Checkpoint implementation — 13 November 2026

Critic: GPT-6.1 Sol, reasoning ultra. Skop: discrepancy antara protokol checkpoint Gaussian joint dan implementation, bukan semakan semula skor pelan.

**Fakta diukur:** Pelan menetapkan pemilihan validation: macro sense accuracy tertinggi, kemudian total joint loss terendah, kemudian checkpoint paling awal. Implementation lama memilih total joint loss sahaja. Audit seed 17 bagi checkpoint langkah 100 memberi accuracy `1.0`, gain `0.0870306`, context-node delta `6.43e-8`, domain delta `1.05e-7`, dan adaptive integration error `2.54e-10`; semuanya melepasi gate region `1e-4`. Rekod MLflow audit: `fb7fc512dd1d466b8c3e2bda882294cf`.

**Inferens critic:** Ini ialah jurang prospective protocol, tetapi bukan bukti checkpoint seed 17 yang salah: accuracy langkah 100 sudah maksimum dan lossnya paling rendah. Namun audit terdahulu tidak merekod score semua checkpoint, maka inferens itu belum provenance yang mencukupi untuk pilot tiga-seed.

**Tindakan terikat:** repair measurement. Trainer ditukar untuk menilai setiap checkpoint validation dengan adaptive train-anchor region evaluator, memerlukan gate numerik, dan memilih accuracy → loss → step. Seed 17 diulang tanpa membuka test: kelima-lima checkpoint (20 hingga 100) lulus gate dan memperoleh accuracy `1.0`; loss memilih langkah 100. Artifact latihan baharu direkodkan dalam MLflow run `99df11c7083944fabb947ee637628e00`.

**Kesimpulan hipotesis:** Tiada prediction saintifik gagal. Satu seed validation menyokong peralihan region mengikut konteks pada toy ini sahaja. Ia belum menunjukkan manfaat training berbanding geometri rawak atau contextual vector baseline, belum tiga seed, belum sampler, dan bukan bukti hardware.

## Checkpoint control geometry beku — 13 November 2026

Critic: GPT-6.1 Sol, reasoning ultra. Tindakan daripada critic sebelumnya: nilai checkpoint lexical sedia ada pada validation dengan map train-anchor dan audit adaptif yang sama, tanpa retrain atau test.

| Seed | Accuracy lexical λ0 → λ1 | Accuracy joint λ1 | Gain lexical → joint |
|---|---:|---:|---:|
| 17 | 0.5 → 1.0 | 1.0 | 0.08319 → 0.08703 |
| 29 | 0.5 → 1.0 | 1.0 | 0.09120 → 0.09544 |
| 43 | 0.5 → 0.5 | 0.5 | 0.04347 → 0.04528 |

**Fakta diukur:** Semua control lulus audit node/domain/adaptive. Joint adaptation menambah zero accuracy untuk setiap seed. Pertambahan gain posterior `0.003844`, `0.004234`, dan `0.001818` lebih besar daripada sensitiviti integrasi yang direkodkan, tetapi semuanya ialah ukuran validation di bawah map checkpoint masing-masing.

**Verdict critic:** Validation screen tiga-seed gagal kerana seed 43; jangan purata untuk menyembunyikan kegagalan. Claim bahawa joint adaptation meningkatkan sense classification dalam pilot ini tidak disokong. Yang disokong lebih sempit: contextual coupling boleh mengubah occupancy/prediction pada frozen lexical geometry bagi seed 17 dan 29. Test contextual Gaussian kekal belum dibuka; ia bukan alat untuk menyelamatkan claim selepas validation gagal.

**Had provenance:** Contract MLflow bagi tiga control direkod oleh environment logging yang melaporkan `jax: not-installed`; ini tidak menerangkan result berpasangan, tetapi contract control sahaja bukan capture lengkap environment evaluator.

## Supersession selepas repair boundary — 13 November 2026

Critic: GPT-6.1 Sol, reasoning ultra. Semakan mendapati tolerance mutlak dalam pembinaan region boleh menambah boundary palsu di Gaussian tail dan memberi half-credit kepada interval besar. Ia ialah ralat measurement, bukan kegagalan seed 43 atau M4.

Repair menggunakan perbandingan strict `qF > qR`, genuine sign crossings sahaja, serta interior probe jika midpoint underflow. Regression test ditambah. Semua checkpoint M2 lexical dan joint yang telah disimpan diaudit semula pada validation tanpa retrain: accuracy kini `1.0, 1.0, 1.0`, dengan max region delta di bawah `4.4e-7`. M4 juga `1.0, 1.0, 1.0`; M1 `0.5, 0.5, 1.0` selepas audit stabil.

Semua verdict terdahulu yang menyatakan seed 43/M4 gagal, atau menyimpulkan bahawa geometry lexical seed 43 bias kepada river, adalah **superseded measurement errors** dan tidak boleh dirujuk sebagai bukti saintifik. Critic membenarkan holdout contextual M2 dibuka hanya selepas manifest checkpoint/map/evaluator dibekukan. Manifest: [M2_CONTEXTUAL_HOLDOUT_MANIFEST.md](../../M2_CONTEXTUAL_HOLDOUT_MANIFEST.md).

## Holdout M2 dan contextual vector control — 13 November 2026

Holdout M2 dibuka hanya selepas manifest `0598f64` dibekukan. Semua tiga seed joint dan frozen lexical mencapai macro sense accuracy `1.0` pada 12 context test (F=6/6, R=6/6, tiada tie), lexical retrieval AUC `1.0`, dan region mass delta maksimum di bawah `1.6e-7`. Ini melengkapkan toy contextual accuracy/retrieval checks untuk checkpoint yang dipilih.

Joint gain posterior test berbanding frozen lexical ialah `0.00364`, `0.00419`, `0.00350` bagi seed 17,29,43; accuracy tidak berubah. Baseline contextual SGNS beku kemudian diuji melalui manifest `f72ea73`: tied D6 dan untied D3 (kedua-duanya 156 parameter embedding) juga mencapai `1.0`, F=6/6 dan R=6/6 pada semua seed. Jadi data ini menyokong contextual readout pada toy, tetapi **tidak menyokong claim bahawa joint energy training lebih baik daripada vector contextual biasa**.

Had kekal: 12 context setiap seed bukan 36 dokumen bebas; vocabulary diketahui dan corpus synthetic kecil. Hasil tidak membuktikan natural-text semantics, sampler/MCMC fidelity, THRML/Torx parity, thermodynamic hardware execution atau advantage representation. Langkah research seterusnya untuk matlamat thermodynamic ialah sampler audit M2 melawan reference exact, atau benchmark yang lebih sukar dan pradaftar; jangan scale atau claim superiority hanya kerana toy perfect.
