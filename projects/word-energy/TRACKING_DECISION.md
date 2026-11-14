# Keputusan experiment tracking

12 November 2026: pengguna memilih **MLflow**, menggantikan cadangan W&B dalam snapshot pelan migrasi terdahulu. Mula dengan MLflow lokal, SQLite metadata dan local artifacts; tiada remote, login atau cloud sync diperlukan. CLI dan Python SDK ialah laluan utama; MCP tidak dikonfigurasi kecuali keperluan sebenar muncul kemudian.

Pemasangan W&B sudah dimulakan sebelum pilihan ini berubah; percubaan W&B offline dihentikan tanpa sync dan package W&B dibuang daripada environment projek. Hasil percubaan tracker menggunakan nilai sintetik ialah smoke test infrastruktur, bukan hasil model.

MLflow 3.18.0 telah dipasang dan diuji pada Python 3.13. Smoke test mencipta satu run sintetik lokal, mengesahkan tiga nilai metric dan satu artifact melalui SDK, dan menyenaraikan run menggunakan CLI. Experiment `word-energy-infrastructure-smoke`, run `e6d7a2dc9e594a98b6fd2436a9851ded`. Ini bukan eksperimen Word Energy. Metadata berada di `tracking/mlflow.db`. Semak `artifact_uri` dalam setiap run sebelum backup: run semasa menggunakan `mlruns/<experiment>/<run>/artifacts/`; `tracking/artifacts/` kekal sebagai lokasi dirancang tetapi bukan root artifact aktif bagi semua run ini. Semua lokasi artifact yang dirujuk oleh database diabaikan Git dan perlu disertakan dalam backup.

Ujian salinan SQLite dan artifact ke direktori lain mendapati database boleh dibaca, tetapi URI artifact dalam run masih menunjuk ke path asal. Jadi salinan itu **belum membuktikan restore MLflow ke lokasi lain berfungsi secara berdikari**. Sebelum memindah tracker, rancang remapping yang sesuai untuk versi MLflow dan uji restore tanpa bergantung kepada path lama; simpan kedua-dua database dan artifacts. Rekod JSON lokal berada di `experiments/runs/mlflow-restore-check/`.

Dari direktori `projects/word-energy`, ulang smoke test dengan `.venv/Scripts/python.exe experiments/mlflow_offline_smoke.py`. Untuk lihat runs melalui PowerShell:

```powershell
$dbPath = (Resolve-Path tracking/mlflow.db).Path.Replace('\','/')
$env:MLFLOW_TRACKING_URI = "sqlite:///$dbPath"
.venv/Scripts/mlflow.exe runs list --experiment-id 1
```

Gunakan URI eksplisit untuk mengelakkan tracker lain dipilih mengikut direktori semasa. Tiada server MLflow atau MCP dihidupkan secara automatik.

Setiap run research sebenar perlu merekod hypothesis ID, code version, environment, dataset dan split, seed, config, metrics, artifact paths, limitations serta kegagalan. Backup metadata SQLite dan direktori artifacts bersama; Git sahaja bukan backup data run.
