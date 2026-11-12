# Archive research lama

Archive ini ialah salinan setempat. Eksperimen asal tidak dipadam. Backup penuh pada drive berasingan masih belum disahkan.

Kandungan di `legacy-projects/` hanya dianggap lengkap apabila manifest SHA-256 dan laporan batch menyatakan `verified`. Direktori dengan status lain perlu disemak sebelum digunakan. Metadata dan indeks tidak menentukan hasil eksperimen yang tidak diketahui. `.git`, fail tersembunyi, video dan data besar termasuk skop salinan; junction direkod secara berasingan.

`archive_project.py` menggunakan inventori SHA-256 yang telah dibekukan, memeriksa semula source serta destination dan tidak mengubah original. Ia berhenti jika terdapat perubahan source, fail tidak boleh dibaca atau reparse point yang belum diselesaikan. Ketiadaan backup off-device tetap limitation walaupun archive lokal lulus hash.

[Indeks archive](index.yaml) menunjukkan status setiap projek. Jalankan `python archives/build_index.py` selepas laporan copy berubah; `copy_in_progress` bukan archive yang telah selesai. Outcome `null` bermaksud belum disahkan, bukan gagal atau berjaya.

Ujian restore setempat `analog-lab-learn` telah membandingkan 1,559 fail dan memeriksa Git HEAD, branch, refs, worktree dan status termasuk perubahan belum commit. Salinan ujian ada di `restore-checks/` dan tidak direkod sebagai projek legacy baharu. Ini belum menguji pemulihan daripada drive lain. Sesetengah `.venv` dan editable installs mungkin menggunakan absolute path lama; hash salinan yang sepadan tidak menjamin environment berfungsi selepas dipindahkan.
