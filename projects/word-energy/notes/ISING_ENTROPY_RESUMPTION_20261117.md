# Sambungan panel entropy selepas pengguna berhenti

17 November 2026. Panel [asal](ISING_ENTROPY_INTERVENTION_PROTOCOL.md)
dimulakan pada 16 November dan dihentikan atas arahan pengguna. Dua belas
run lengkap; run `b9af12890f824290b86ec81b7ac09d56` untuk `(7043,269,baseline)`
masih tercatat RUNNING tetapi hanya mempunyai source snapshot dan protocol,
tanpa checkpoint, metrik atau config lengkap. Proses batch asal telah keluar;
pemeriksaan proses setempat pada hari berikutnya tidak menemui proses Python.
Ia perlu ditandakan KILLED dengan sebab gangguan pengguna, dikekalkan dalam
ledger, dan satu run baharu bermula dari keadaan awal yang sama dengan tag
`retry_of_run_id`. Tiada checkpoint atau Adam state untuk disambung.

Pengguna meminta kerja disambung. Baki kombinasi dikekalkan tepat seperti
protokol asal; run lengkap tidak diulang. Tiada perubahan pada data, seed,
optimizer, hyperparameter, loss, threshold, fixed endpoint atau turutan
kombinasi. Source trainer dan data disahkan hash-nya terhadap source snapshot
run asal; tambahan pada arm runner hanya menambah metadata retry.

**Penyimpangan sumber:** protokol asal menetapkan satu batch sehingga 45
minit. Ia dihentikan pengguna sebelum lengkap; sambungan menggunakan batch
baharu sehingga 45 minit lagi, jadi jumlah masa merentas batch boleh melebihi
45 minit. Ini perubahan operasi selepas sebahagian hasil dilihat, bukan
prapendaftaran baharu atau pembukaan test. Hasil kekal pembangunan posthoc
pada satu corpus, dan semua run, termasuk attempt yang terputus, dilaporkan.
Jangan menilai gate keseluruhan sehingga semua 48 kombinasi mempunyai tepat
satu attempt FINISHED yang sah. Kawalan baseline dipadankan semula dengan
rekod historical sebelum mana-mana treatment bagi seed yang sama.
