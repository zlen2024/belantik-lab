# Pelan migrasi: research workspace yang kemas dan boleh diaudit

12 November 2026 · **Status: pelan sahaja.** Tiada folder eksperimen dipindahkan, repo baharu diwujudkan, dependency dipasang atau eksperimen dilatih semasa penyediaan pelan ini. Dua pelan Word Energy Embedding ialah sumber reka bentuk utama; versi Ising/Z1 dahulu ialah arah utama, versi Gaussian ialah rujukan/baseline. Markah critic menerangkan mutu pelan pada versi yang disemak, bukan keputusan eksperimen atau hardware.

## 1. Perlindungan yang telah dibuat

Salinan keselamatan setempat di [`_safety_backup_20261112_word_energy/`](_safety_backup_20261112_word_energy/) merangkumi:

- Kedua-dua pelan, teks perbualan asal dan rumusan, semua fail review/snapshot critic, fixture dan laporan hasil skala yang berkaitan.
- `AGENT.md`, `AGENTS.md`, `CLAUDE.md`, dua fail skill lock, seluruh `.agents/skills` serta `.claude/skills`.
- Salinan asal cadangan tools daripada lampiran pengguna.

Sebanyak **2,581 fail disalin dan kandungannya disahkan menggunakan SHA-256** terhadap sumber semasa salinan dibuat. Manifest lengkap: [BACKUP_MANIFEST.json](_safety_backup_20261112_word_energy/BACKUP_MANIFEST.json). Sebanyak 32 junction `.claude/skills` direkod dengan sasaran asal; kandungan junction disalin sebagai direktori biasa dalam safety copy, bukan dipulihkan secara automatik sebagai junction. Ini membolehkan kandungan dibaca tetapi **bukan salinan lengkap atribut filesystem**. Semasa restore, semak mapping terlebih dahulu dan bina junction hanya jika diperlukan.

**Batas penting:** ini salinan pada drive C: yang sama dan hanya meliputi item di atas; ia bukan backup seluruh workspace dan bukan backup off-device. Ujian restore sementara telah memadankan 166 fail daripada backup: dua pelan, teks asal, reviews dan empat skill trees. [Laporan restore subset](research_workspace_planning/restore_subset_check.json) menyimpan hash dan skop; itu bukan restore lengkap atau ujian junction dan tidak mengesahkan backup semua eksperimen. Eksperimen lama dan semua kandungannya masih berada pada lokasi asal. Jangan padam safety copy selepas Git diwujudkan tanpa backup berasingan yang telah diuji.

## 2. Keadaan sebenar yang telah diperiksa

- Akar `master` bukan Git repo. Tinjauan projek bukan tersembunyi mengenal pasti empat repo di dalamnya yang mempunyai sejarah berasingan; direktori tersembunyi belum diaudit menyeluruh untuk nested repo tambahan:
  - `thrml`: branch `main`, HEAD `5911729fba023c15dd5d38755736be2005d85117`, status bersih pada masa pemeriksaan.
  - `torx`: branch `main`, HEAD `f4c9fe890c5e1132567d355549c25aa6c2b35a91`, status bersih.
  - `DTM_deepdive/paper_code`: branch `master`, HEAD `7c22d19c218ab353a770f7c2b1504570b7cbe3ba`, status bersih.
  - `analog-lab-learn/Claude-cloud`: branch `ccr-723d1382-lrs1tx`, HEAD `859b748a65189f0093e41e935c524f880a418f77`; **`sky130-web/package-lock.json` ada perubahan belum commit**. Jangan reset, checkout atau hanya mengarkibkan commitnya: working tree itu juga perlu dipelihara.
- `fundemental diff` ialah **junction** ke `fundamental_diff`. `.claude/skills` mempunyai 32 junction yang diperiksa ke `.agents/skills`. `.agents/skills` ialah direktori yang mengandungi aset skills; `.claude/skills` juga mempunyai fail/direktori sebenar tersendiri. Jangan menganggap semua kandungan kedua-dua root sama atau membuang salah satu.
- Inventori anggaran kandungan biasa, tidak termasuk metadata `.git` dan junction yang tidak diikuti: `mnist_diff` 10.324 GiB, `DTM_deepdive` 3.185 GiB, `normal_computing` 1.855 GiB, `rbm-anomaly-detection` 1.260 GiB, `fundamental_diff` 0.698 GiB, `z1_voters` 0.531 GiB, `videos` 0.472 GiB, `p_family_xor` 0.326 GiB, `word2energy` 0.217 GiB, `analog-lab-learn` 0.146 GiB, `vbuild_assets` 0.123 GiB, `attn_thrml` 0.083 GiB. Angka ialah tinjauan, bukan ukuran backup, mungkin berubah dan termasuk environment/cache; jangan guna untuk memilih fail untuk dibuang. Kapasiti C: ketika semakan sekitar 118.5 GiB bebas.
- `git`, `python` dan `uv` ditemui pada PATH. `dvc`, `mlflow`, `marimo`, `quarto`, `restic`, `borg` dan `zotero` tidak ditemui pada PATH; ini **bukan bukti muktamad** aplikasi atau Python package tersebut tidak terpasang di tempat lain. Tiada pemasangan dibuat.

## 3. Keputusan reka bentuk yang dicadangkan

Sediakan satu workspace Git untuk **research aktif, dokumentasi, indeks archive dan konfigurasi yang selamat**. Simpan kandungan archive besar dalam filesystem yang sama secara tersusun tetapi jangan auto-commit seluruh archive ke Git. Repo upstream `thrml`/`torx` dan repo legacy lain dikekalkan sebagai repo bebas dengan `.git` masing-masing; root Git tidak akan secara automatik menyimpan sejarah repo bersarang. Jika suatu hari hendak menggunakan submodule, ia keputusan berasingan dengan URL, commit, akses dan kerja belum commit yang disemak terlebih dahulu. [Git: submodules](https://git-scm.com/docs/gitsubmodules.html).

Cadangan struktur akhir, **bukan keadaan sedia ada**:

```text
master/
├── AGENT.md, AGENTS.md, CLAUDE.md   # panduan agent & cara membaca
├── .agents/skills/ .claude/skills/ # skill aktif; junction dipulihkan secara sah
├── README.md                        # peta research workspace
├── projects/
│   └── word-energy/                 # ruang kerja baharu
│       ├── plans/                   # salinan tepat dua pelan + review/teks asal
│       ├── src/ tests/ experiments/ # dibina pada fasa implementasi kelak
│       ├── theory/ notes/ reports/
│       └── references.bib
├── archives/
│   ├── README.md
│   ├── index.yaml                  # Git: indeks dan metadata sahaja
│   └── legacy-projects/            # kandungan archive: diabaikan Git utama
├── shared/                         # template, panduan, reference metadata
└── _safety_backup_20261112_word_energy/ # dikecualikan Git; salinan setempat
```

Fail besar, datasets, model, media, log, environments dan archive asal kekal wujud dalam salinan archive. `.gitignore` menghalang ia terlanjur masuk ke Git utama, **bukan** bermaksud fail tersebut dikecualikan daripada archive atau backup. Fail `.env`, credential dan kunci tidak diterbitkan atau dimasukkan Git; inventori private mengenal pasti keperluan pemeliharaan dan kaedah penyimpanan selamat tanpa menulis nilainya ke log.

Cadangan archive menyimpan **keseluruhan projek lama**, bukan merombak fail individu atau meneka mana eksperimen berjaya/gagal. Letakkan di bawah `archives/legacy-projects/<nama-asal>/` mengekalkan struktur relatif asal. Buat index dan manifest **di sebelah** projek; jangan edit isi projek lama untuk memasukkan metadata. Pengesahan fail tersembunyi dan `.git` termasuk objek, refs, branch serta perubahan working tree adalah wajib.

Folder sasaran archive ialah `analog-lab-learn`, `attn_thrml`, `channel`, `DTM_deepdive`, `fundamental_diff`, `mnist_diff`, `ncm_torx`, `normal_computing`, `p_family_xor`, `rbm-anomaly-detection`, `research`, `thrml_linalg`, `vbuild_assets`, `videos`, `word2energy` dan `z1_voters`. Klasifikasi ini cadangan berdasarkan arahan pengguna, **bukan kesimpulan bahawa semuanya tidak lagi berguna**. `thrml` dan `torx` ialah dependency upstream repo; cadangan arkib juga menyimpan snapshot lengkap mereka tetapi persekitaran aktif baharu perlu dependency yang disediakan semula dan diuji; jangan menukar atau memindahkan repo sedia ada sebelum audit. `fundemental diff` direkod sebagai junction, bukan satu salinan folder kedua. Fail root seperti PDF, dokumen, script dan log diinventori dan dipetakan secara eksplisit; tiada yang diabaikan secara senyap. `.agents`, `.claude` kekal infrastruktur aktif; `.pi`, `.tmp` dan `__pycache__` diperiksa dan diinventori dahulu, jangan padam.

Pengguna menjelaskan workspace juga mengandungi video. **Video, footage, projek editing, audio, aset dan render semuanya termasuk dalam skop archive dan backup**, mengikut inventori sebenar, termasuk apabila ia tersimpan di dalam folder lain. Jangan mengecilkan, menukar format atau membuangnya semasa migrasi. Angka folder `videos` sahaja tidak mewakili semua media di workspace. Git utama boleh mengabaikan fail media besar, tetapi manifest perlu menyimpan path, saiz dan hash supaya ia boleh dikesan dan dipulihkan.

## 4. Fasa kerja yang dicadangkan

### Fasa 0 — Tetapkan inventori dan keadaan awal

1. Buat inventory machine-readable dengan absolute source path, relative destination, jenis fail, saiz, timestamp, SHA-256, hidden/system attribute, reparse/junction target, git boundary dan error. Jangan membaca nilai credentials ke laporan atau output terminal. Jangan ikut junction secara rekursif membuta tuli.
2. Rekod dependency luaran, nested repo, branch/HEAD, refs serta staged/unstaged/untracked status, git worktrees/submodules dan lokasi `.git` directory/file. Rekod `analog-lab-learn/Claude-cloud` sebagai dirty; salin perubahan bersama working tree. Jangan jalankan `git reset`, `clean`, checkout atau operasi yang menukar sejarah.
3. Semak proses sedang menulis, ruang kosong sebelum dan selepas snapshot, saiz `.git` serta files yang terkunci/tidak boleh dibaca. Output mesti membezakan hash siap, skipped, error, changed-during-read dan junction; **0 errors dan 0 perubahan tidak ditangani** diperlukan sebelum migrasi dilabel lengkap. Jika fail berubah, ambil snapshot konsisten selepas write berhenti dan hash semula; jangan claim lengkap dengan count separa.
4. Audit lokasi `.venv`, editable installs, absolute paths, config, links Markdown, notebooks dan scripts. Python venv yang dipindah boleh kekal sebagai artifak lama tetapi tidak dijamin berfungsi; rekod versi/dependencies dan bina environment baharu daripada lockfile apabila sesuai. Tidak menganggap semua cache atau data boleh dibuang.

Bekukan satu manifest sumber sebelum membuat archive/staging di dalam root; inventori source original sebagai set tertutup dengan masa/hash. Senaraikan dan tandakan `_safety_backup_20261112_word_energy`, ZIP keselamatan, `archives/`, staging, restore temporary dan destination lain sebagai output yang berasingan, bukannya source legacy yang diimbas semula ke dalam dirinya. Backup juga mesti melindungi safety copy/ZIP jika diputuskan perlu, tetapi sebagai objek sekali sahaja, tanpa recursion atau salinan pendua tanpa had. Inventori selepas itu tidak menggantikan manifest asal; rekod delta fail baharu/berubah.

Preflight destination: kira penggunaan berasaskan saiz sumber sebenar termasuk hidden, metadata `.git` dan untracked files; tambah jumlah kandungan target junction yang perlu dipelihara, ruang staging, index/hash, overhead filesystem dan ruang untuk sekurang-kurangnya satu restore test. Semak ruang bebas pada **setiap** volume sumber/destinasi serta kes pertindihan copy/backup; jangan double-count target junction yang sama jika ia disimpan sekali, tetapi sediakan ruang tambahan jika tool dereference membuat salinan berganda. Semak filesystem destination menyokong nama/Unicode, panjang path Windows, jenis reparse/junction, hardlink dan metadata yang diperlukan. Jika tidak disokong, guna format dan manifest yang mengekalkan maklumat pemulihan; tandakan sebarang kekurangan sehingga diuji. Semak kemampuan tool dan path sebenar dengan fail ujian kecil, bukan anggapan bahawa semua long paths akan berjaya.

Untuk repo yang dikesan, termasuk `.git` file, worktree, submodule atau `gitdir`/`commondir` di luar folder projek, catat lokasi sebenar dan semua objek/ref yang diperlukan untuk restore. Semak `git worktree list --porcelain` dan lokasi gitdir/commondir melalui arahan baca sahaja; audit alternates/submodules dan perubahan tidak commit. Jika bergantung pada path di luar source, masukkan dependency yang dibenarkan ke pelan backup atau tandakan backup tidak lengkap; jangan edit pointer/history dalam salinan arkib asal.

### Fasa 1 — Backup lengkap di luar drive asal

Salin seluruh workspace yang diperlukan **termasuk dataset, checkpoints, `.git`, hidden files dan pelan** ke storan berasingan yang dipilih pengguna; pastikan destinasi tidak berada dalam source yang disalin. Sertakan sistem metadata junction/reparse dan rekod permissions/atribut yang disokong. Encrypt backup jika mengandungi data sensitif, simpan kunci berasingan dan jangan tulis credential ke repo.

Banding hash source–backup dan jumlah fail mengikut jenis. Buat **restore test ke direktori sementara lain**: kedua-dua plan dan review terbaca, sekurang-kurangnya satu projek lengkap dicompare hashes, satu nested Git repo diperiksa history + status termasuk dirty worktree, dan skill sebenar + target junction disahkan. Hanya selepas itu backup dilabel verified. Jika tiada storan berasingan, pelan dan inventori boleh disiapkan, tetapi backup off-device masih tertangguh; jangan menyatakan archive atau Git ialah backup bebas.

**Polisi junction/reparse:** arkib merekod jenis link, path relatif, original target dan hash kandungan target; target data yang sama disimpan sekali dan dibuktikan lengkap. Jangan dereference tanpa sempadan, mereplikasi target di luar akar atau secara senyap mengabaikan link. Ketika restore ke akar terasing yang berlainan, bina semula link hanya selepas memetakan target ke lokasi dalam restore itu; reject link yang terkeluar daripada akar restore yang dibenarkan atau kembali ke `master` hidup. Uji semua 32 mappings Claude-ke-agents serta junction `fundemental diff`; sahkan link bukan dangling, boleh dibaca, kandungan target sepadan hash dan tiada cycle. Jika target luar workspace memang diperlukan, simpan dan rancang restore sebagai dependency berasingan dengan skop jelas; jangan cipta link luaran secara automatik. Salinan safety yang ada didereference dan mapping tersimpan, jadi bukan bukti operasi recreate sudah berjaya.

### Fasa 2 — Rangka workspace aktif dan dua pelan

Wujudkan `projects/word-energy/plans/` dan salin kedua-dua dokumen sumber **tanpa mengedit byte**; sertakan `RUMUSAN_IDEA_BIKAR_ENERGY_EMBEDDING.md`, `TEKS_ASAL_BIKAR_ENERGY_EMBEDDING.txt`, direktori `reviews/energy_embedding_v2/`, `reviews/z1_first/`, contoh kod/JSON, dan laporan terdahulu yang dirujuk. Kekalkan relative path asal melalui susunan yang setara atau bina dokumen navigasi/redirect berasingan tanpa mengubah kandungan snapshot yang disemak. Semak setiap pautan tempatan dan hash selepas salin.

Dalam README aktif, jelaskan hubungan dua pelan: Gaussian sebagai baseline; Ising/p-bit sebagai laluan utama menuju Z1; skor 9.6 dan 9.55 merujuk review pada **snapshot tertentu**, bukan jaminan eksperimen. `src` baharu belum mewarisi hasil atau model lama secara automatik. Jika salin kod terpilih kemudian, nyatakan asal, lesen, versi, perbezaan dan sebab penggunaannya.

### Fasa 3 — Salin archive penuh secara berperingkat

Satu projek setiap batch: snapshot read-only, catat semua fail termasuk `.git` dan working tree, salin ke destination sementara, verify manifest/hash/path dan status Git, kemudian namakan destination archive secara atomik jika boleh. Simpan original dalam keadaan asal sepanjang verifikasi; archive bukan proses rewrite source. Sertakan raw filesystem contents, jangan default-exclude `.venv`, checkpoints, cache atau media. Jika fail tidak dapat disalin, projek berstatus `incomplete` dan original kekal. Jangan memanggil operasi `move`/delete sehingga path source dan destination disahkan tidak bertindih, semua salinan sah dan pelan pembersihan berasingan jelas.

Setiap `archive.yaml`/indeks simpan `original_path`, `archive_path`, `archive_id`, UTC date, content manifest digest, repo status/commit jika ada, symlink/junction manifest, ukuran dan limitations; `hypothesis`, `outcome`, dataset dan tracking ID hanya isi jika ada bukti, jika tidak `null/unknown`. Indeks menyimpan rujukan kepada report/asalan, bukan menulis semula sejarah atau menghasilkan run MLflow palsu. Simpan metadata private berasingan jika ia mendedahkan path/perkara sensitif.

### Fasa 4 — Git untuk kerja baharu

Sebelum `git init`/commit: sahkan lokasi repo, semak secrets dan licences, tentukan `.gitignore` dan `.gitattributes`, semak **senarai penuh fail yang akan ditrack**, saiz serta nested-repo boundaries. Cadangan: Git untuk code baru, konfigurasi selamat, markdown, source citation, small test fixtures dan archive index; exclude backups, raw archive, `.git` bersarang, `.venv`, data besar, checkpoints, PDFs berlesen dan secrets secara default; pengecualian khusus direview. Jangan guna `git add .` tanpa melihat staged contents.

Git utama merekod **index** untuk legacy dan salinan pelan yang telah diperiksa, bukan mencipta sejarah palsu bagi fail lama. Allowlist awal yang perlu diaudit: `AGENT.md`, `AGENTS.md`, `CLAUDE.md`, workspace README, skill lockfiles, source sebenar `.agents/skills/`, entri `.claude/skills/` yang bebas, manifest path/target junction, projek aktif `plans/`, source/config/tests kecil dan indeks/metadata archive. Setiap fail dinilai untuk saiz, lesen dan secrets sebelum ditrack; jangan menganggap semua upstream skill assets boleh terus diterbitkan. Junction tidak direkod sebagai salinan biasa yang dijangka boleh restore di Windows: simpan manifest portable dan sediakan arahan rebuild/test, sambil memastikan data sebenar dibackup. Periksa `git status --short`, dry-run staged list dan seterusnya `git diff --cached` termasuk ukuran; tiada operasi commit jika ada item di luar allowlist yang belum direview.

Catat commit/branch upstream sebagai metadata dan preserve repo Git asal dalam archive. Mula dengan local Git; remote/private hosting, lesen penerbitan dan sebarang push ialah keputusan berasingan dan **remote bukan prasyarat untuk Git lokal**. Git init/commit belum dilakukan dalam fasa plan ini.

### Fasa 5 — Panduan agent dan skills

Kekalkan satu panduan sumber bersama `AGENT.md`; `AGENTS.md` untuk Codex dan `CLAUDE.md` untuk Claude menunjukkan laluan ke panduan tersebut. Kemas kini hanya selepas laluan migrasi muktamad dan uji ia terbuka dari root serta subproject. Ia patut meliputi: bahasa Melayu untuk nota, cara cari research dan archive, pembezaan idea/hasil, split train/validation/test, cara log failure, pilihan skill, batas hardware, reproducible command, dan peraturan menjaga secrets.

Inventori semua skill, lock, local edits serta fail rujukan yang skill panggil; banding hash safety backup dengan versi aktif. Pelihara **kedua-dua direct copies dan junction mappings**—jangan menimpa skill `thrml` atau mana-mana salinan Claude yang bebas dengan nama sama. Uji discovery Codex dan Claude menggunakan `SKILL.md` sebenar dan contoh path di lokasi baharu. Kemaskan junction hanya selepas absolute target disahkan. Jika skill `archive-migration` baharu membantu selepas workflow jelas, reka dan validate secara berasingan dengan `skill-creator`; sekarang cadangan, belum dicipta. Elakkan menggantikan arahan skil yang berfungsi dengan satu prompt besar.

### Fasa 6 — Aktifkan tools secara beransur

Mulakan hanya Git + Markdown + Python/uv + SHA-256 + `rg` apabila tersedia. Setiap run baharu mempunyai `experiment_id`, config, seed, data version, code revision/snapshot, environment, device, runtime, metrics, limitations dan paths dalam JSON/YAML bersama report BM. Ini berfungsi walaupun tracker belum tersedia. Data preprocessing fit pada train sahaja; test dibuka selepas protokol/checkpoint dibekukan.

Mengikut minat pengguna, **W&B ialah calon utama untuk pilot tracker**, bukan lagi MLflow sebagai pilihan yang telah diputuskan. Mula dengan satu run percubaan kecil dalam mode offline; JSON/Markdown dan artifacts lokal kekal rekod yang boleh dibaca. Semak bagaimana export/restore dan query agent berfungsi sebelum memutuskan workflow kekal. Mode offline menyimpan data run di lokal; untuk melihatnya dalam hosted W&B, dokumentasi menerangkan langkah sync kemudian, jadi jangan menjanjikan dashboard hosted tanpa sync. Sync/cloud hanya dibuat selepas pengguna memilih akaun/projek dan apa yang boleh dimuat naik. Jangan mengimport hasil lama seolah-olah run baru.

W&B ada CLI untuk operasi seperti konfigurasi dan sync, serta Python SDK/Public API untuk merekod dan meng-query runs; jangan anggap CLI sahaja mempunyai semua operasi API. [W&B SDK](https://github.com/wandb/wandb), [W&B Public API](https://docs.coreweave.com/products/wandb/track/public-api-guide). Bila CLI tidak menyediakan query yang dikehendaki, gunakan skrip Python SDK tempatan yang jelas dan boleh diulang sebelum menambah MCP.

MLflow kekal alternatif local-first yang munasabah, bukan tracker kedua serentak secara default. CLI menyediakan operasi seperti senarai eksperimen/runs, maklumat run, artifacts dan server; SDK melengkapkan fungsi programatik. [MLflow CLI](https://mlflow.org/docs/latest/api_reference/cli.html). Metadata backend SQLite dan artifact store adalah berasingan; keduanya mesti termasuk backup/restore. [MLflow backend](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/), [artifact store](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/).

**Kedua-dua produk mempunyai MCP rasmi**, tetapi MCP bukan syarat permulaan. [W&B MCP](https://github.com/wandb/wandb-mcp-server) boleh digunakan kemudian untuk pertanyaan agent terhadap data W&B. [MLflow MCP](https://mlflow.org/docs/latest/genai/mcp/) berstatus experimental, memerlukan versi ≥3.5.1 dan dokumentasi menerangkan kategori `ml` untuk experiments/runs/models serta `genai` untuk traces; semak versi dan keupayaan sebenar sebelum bergantung padanya. MCP biasanya perlu dipasang/dikonfigurasi atau disambungkan; ketersediaan upstream bukan bukti integrasi tersedia di mesin ini. Pilih CLI dahulu jika mencukupi, SDK bila perlu, MCP hanya jika akses agent lebih mudah dan memberi fungsi yang diperlukan. Kunci API kekal di secret store, bukan dalam repo atau mesej.

| Keperluan | W&B | MLflow |
|---|---|---|
| Run training dan log metrics | Python SDK `wandb.init`/logging; CLI melengkapkan operasi, bukan pengganti automatik semua SDK | Python SDK untuk instrument training; CLI juga menyediakan pengurusan projek/runs/artifacts yang didokumenkan |
| Baca/banding runs untuk agent | Public API Python; bina command wrapper yang mengeluarkan JSON jika perlu | `mlflow runs list/describe` dan Python SDK, bergantung pada versi |
| Offline/local | Offline run files pada mesin, sync kemudian untuk hosted UI; semak output dan restore | Local SQLite metadata + local artifacts; kedua-duanya perlu backup |
| MCP rasmi | Ada untuk Models/Weave, pilihan jika diperlukan | Ada, experimental; kategori ML perlu dipilih/diuji mengikut versi |

Semua perbandingan ini ialah dapatan dokumentasi, belum ujian tempatan. Pilot W&B perlu mencatat versi SDK, offline directory, run ID, config, metrics dan artifacts, serta menguji baca/eksport yang benar-benar disokong versi itu. Public API untuk run tersimpan pada server tidak semestinya membaca offline files tanpa sync; jangan claim ia berbuat demikian. Jika data benar-benar perlu kekal setempat dan UI diperlukan, nilai MLflow lokal sebagai alternatif. Harga, had penggunaan dan self-hosted license W&B disemak pada masa keputusan, tanpa menganggap academic eligibility atau self-hosting percuma. [W&B pricing](https://wandb.ai/site/pricing/).

## 5. Cadangan tools dan waktu penggunaan

| Fungsi | Cadangan | Bila / sebab |
|---|---|---|
| Code dan sejarah | Git lokal | Mulakan selepas audit; remote hanya kemudian. Tidak menggantikan backup. |
| Notes dan indeks | Markdown + YAML/JSON + `rg` | Local-first, agent boleh baca; Obsidian pilihan UI, bukan database wajib. |
| Python environment | `uv` + environment/lock per project | `uv` ditemui pada PATH; audit keserasian Python/JAX/THRML dahulu. |
| Experiment tracker | W&B offline dahulu, kemudian pilihan sync | Calon utama berdasarkan minat pengguna; CLI + SDK dahulu, MCP jika berguna. MLflow SQLite lokal ialah alternatif; jangan jalankan dua tracker serentak secara default. Semak pelan harga, privasi, deployment dan akses sebelum sync. |
| Dataset/checkpoints | Fail + hash dahulu; DVC kemudian | DVC simpan metadata Git dan memerlukan storage yang sesuai; remote bukan backup automatik tanpa salinan/pengujian restore. [DVC remote storage](https://doc.dvc.org/user-guide/data-management/remote-storage). |
| Backup | Restic ke drive atau storan lain | Calon Windows/local; rancang encryption, kunci dan restore test. Borg tidak dipilih sebagai default Windows tanpa semak keperluan platform. [Restic](https://restic.readthedocs.io/en/stable/040_backup.html). |
| Paper | Zotero + BibTeX | Bila library sumber hendak disusun; PDF berlesen bukan default untuk Git/public. Periksa backup library sendiri. |
| Notebook | Python scripts; marimo pilihan | Pilot dahulu sahaja; ia menyimpan notebook sebagai Python. [marimo docs](https://docs.marimo.io/). |
| Laporan formal | Markdown dahulu; Quarto bila perlu publish | Menghasilkan HTML/PDF/DOCX dari sumber teks; tidak wajib untuk setiap run. [Quarto](https://quarto.org/). |
| Interface agent | Filesystem + CLI dahulu; MCP bila perlu | Kurangkan integrasi; tiada connector atau hak akses diasumsikan. |

Jangan install semua sekali. Untuk mana-mana tool baharu, rekod versi, license/deployment constraints, lokasi data dan cara export/restore. Jangan menggunakan kelebihan/benchmark vendor sebagai hasil projek kita.

## 6. Syarat selesai migrasi dan pemulihan

Migrasi hanya dikatakan selesai jika:

1. Kedua-dua pelan dan snapshot review dipadankan hash, boleh dibaca, dan navigation links sah. Tiada markah critic direka semula.
2. Semua item sumber mempunyai destination, status dan manifest; tiada missing/skipped/unreadable yang tidak diselesaikan atau diterima secara jelas. Hasil, negative runs, large artifacts, untracked work dan `.git` termasuk.
3. Semua repo Git yang ditemui dalam Fasa 0 dapat diperiksa selepas restore dengan commit/branch/status setara, termasuk gitdir/commondir/worktree/submodule yang terpakai; empat ialah minimum diketahui, bukan upper bound. Dirty change projek Claude-cloud masih ada. Junctions/discovery agent direkod dan diuji. Tiada arahan editable install/venv dianggap berjaya sebelum diuji.
4. Safety backup di drive berasingan lulus sekurang-kurangnya satu restore test dan integrity compare. Salinan di C: yang sudah dibuat ialah perlindungan tambahan sahaja.
5. Git status/staged list, ignore rules dan secret check diperiksa; tidak ada data sensitif atau binaries besar diterbitkan secara tidak sengaja. Archive index dan README menerangkan cara cari serta pulihkan tanpa mengubah original.
6. Tiada original dibersihkan secara automatik. Pembersihan/relokasi akhir ialah kerja berasingan selepas backup, salinan dan recovery diverifikasi; senarai terperinci perlu ditunjukkan dahulu. Jika satu langkah gagal, hentikan batch tersebut dan kekalkan source.

**Keputusan yang diperlukan mengikut peringkat:** destinasi off-device diperlukan sebelum backup penuh luaran; pilihan remote/private hosting tidak menghalang Git lokal; kaedah menyediakan upstream `thrml`/`torx` hanya diperlukan sebelum menyediakan dependency aktif. Tiada pilihan ini perlu dibuat untuk membaca dan mengkritik pelan sekarang.
