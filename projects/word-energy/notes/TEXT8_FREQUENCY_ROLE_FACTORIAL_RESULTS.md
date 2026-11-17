# Text8: frequency × role factorial development

17 November 2026. Dua belas run, empat variasi × tiga seed. Tujuan: mengkaji kesan tambahan kekerapan calon dan pemisahan jadual target/context. Lihat [protokol](TEXT8_FREQUENCY_ROLE_FACTORIAL_PROTOCOL.md). Reka bentuk dipilih selepas melihat pilot: semua dapatan ialah **posthoc development**, belum disahkan pada dataset bebas.

| Variasi | Jadual | Frequency offset | Parameter | AUC purata | MRR@21 purata |
|---|---|---|---:|---:|---:|
| tied | dikongsi | tiada | 3,002 | 0.63989 | 0.27446 |
| tied-frequency | dikongsi | ya | 3,002 | 0.69803 | 0.33121 |
| role | berasingan | tiada | 6,002 | 0.67869 | 0.31466 |
| role-frequency | berasingan | ya | 6,002 | 0.70038 | 0.34232 |

AUC mengukur berapa kerap konteks yang diperhatikan diberi skor lebih tinggi daripada calon noise; seri dikira separuh. MRR@21 mengukur kedudukan konteks sebenar antara 20 calon noise. Kedua-duanya bukan ujian terus kefahaman makna.

## Kesan berpasangan

| Perubahan | AUC seed 17 / 29 / 43 | Purata AUC | Lulus kriteria? | Purata MRR |
|---|---|---:|---|---:|
| Tambah frequency, table dikongsi | +0.05368 / +0.05755 / +0.06319 | +0.05814 | ya | +0.05676 |
| Tambah frequency, table berasingan | +0.01935 / +0.02163 / +0.02409 | +0.02169 | ya | +0.02766 |
| Pisah table, tanpa frequency | +0.03489 / +0.04248 / +0.03903 | +0.03880 | ya | +0.04021 |
| Pisah table, dengan frequency | +0.00056 / +0.00656 / −0.00007 | +0.00235 | tidak | +0.01111 |
| Interaksi | −0.03433 / −0.03592 / −0.03910 | −0.03645 | tidak | −0.02910 |

Kriteria deskriptif dibekukan sebelum run: ketiga-tiga perbezaan AUC mesti positif dan purata sekurang-kurangnya 0.005. Ia bukan p-value dan tidak dikenakan kepada MRR. Kesan pemisahan table dengan frequency kecil dan tidak konsisten pada AUC, tetapi MRR meningkat dalam tiga seed. Frequency membantu kedua-dua bentuk table. Interaksi negatif bermaksud kesan frequency lebih kecil apabila table dipisahkan; ini menerangkan corak skor dan tidak membuktikan mekanisme sebab.

## Rekod setiap run

| Arm | Seed | Run ID | AUC epoch3 | MRR epoch3 | Masa (s) |
|---|---:|---|---:|---:|---:|
| tied | 17 | `c6031e2c34cd4dd3a4ab3bbefaee91dd` | 0.64842 | 0.28082 | 10.6 |
| tied | 29 | `497b4ab87e6242ea855a57ae9b1b8093` | 0.63126 | 0.26493 | 8.1 |
| tied | 43 | `fd23d07a29b04e019e0dbc03798510e9` | 0.64000 | 0.27762 | 8.8 |
| tied-frequency | 17 | `8eab021c85ea4545bef2b804dec3b716` | 0.70210 | 0.33470 | 8.9 |
| tied-frequency | 29 | `52250706518949f0bd4d8886fdf5fde5` | 0.68881 | 0.32341 | 8.9 |
| tied-frequency | 43 | `d56b97169f46494baac3d86ef902217a` | 0.70319 | 0.33553 | 8.9 |
| role | 17 | `a81eb47f76b64f6a8965439f06dc9b40` | 0.68331 | 0.32210 | 8.7 |
| role | 29 | `af152459036641caae5ffd925e9f011b` | 0.67374 | 0.30433 | 8.9 |
| role | 43 | `330e7b92c7c744d886c4e51be7bb267e` | 0.67903 | 0.31755 | 8.4 |
| role-frequency | 17 | `5a141602f79445108dc872fb0d1c3880` | 0.70266 | 0.34245 | 8.6 |
| role-frequency | 29 | `1eb62d21661049979e488a452a76777b` | 0.69537 | 0.33815 | 10.2 |
| role-frequency | 43 | `c25f7f57ae3d4b7096de8e3468e30c01` | 0.70312 | 0.34638 | 10.3 |

## Pemeriksaan dan batas

- Evaluator menggunakan skor yang sama untuk pasangan yang sama supaya roundoff batch tidak mencipta seri palsu. Untuk graph simetri tanpa offset, pasangan songsang berkongsi nilai; dengan offset atau jadual berasingan, arah dikekalkan. Tiada tolerance menyamaratakan skor yang hampir.
- Audit evaluator run `6ce5112a6c3643139db6027f9febc13f` menghasilkan perbezaan maksimum 0 bagi AUC/MRR terhadap 12 checkpoint epoch3 pilot yang telah diperbetulkan. Interaksi sifar tanpa offset memberi AUC 0.5 dan MRR H21/21; dengan offset ia menghasilkan frequency-only AUC 0.660195 dan MRR 0.265787.
- Data: 50,000 pasangan train, 5,000 validation pairs, vocabulary train 1,000, validation token coverage 66.19%, 20 calon noise setiap pasangan. Dataset run `6cf76591c3fa49c686a3d81d7f225607`. Semua run berkongsi sample, noise dan corpus; tiga seed bukan tiga dataset bebas.
- Role-specific menggandakan parameter daripada 3,002 kepada 6,002 dan mengubah dynamics optimizer; hasilnya tidak boleh dikaitkan dengan simetri sahaja. Frequency offset ialah `0.25 log(count_train(candidate))` yang ditambah pada skor host semasa train dan evaluation; belum dibuktikan sebagai tenaga fizikal pada cip.
- Tiada shuffled control dengan frequency offset. Kelebihan berbanding frequency-only menunjukkan discrimination tambahan terhadap noise, bukan bukti semantik. Perbandingan vector/SGNS lama tidak mengawal explicit frequency prior ini.
- Noise boleh merangkumi perkataan relevan, data tiada sempadan artikel, provenance lama tidak lengkap dan tiada label sense. Reserved test tidak dibuka. Ini run klasik exact enumeration, bukan THRML parity atau ujian cip thermodynamic.
- Loss training ialah purata online ketika parameter berubah sepanjang epoch; jangan membandingkannya terus dengan endpoint validation loss sebagai generalization gap.

MLflow experiment `word-energy-text8-frequency-role-factorial`; summary run `89d6495c9b124e649693884d44b76d00`. Audit summary memeriksa status, hash data, source snapshot, semua checkpoint, paired initialization dan metrics. Summary awal `406cc3ef66cc4dffae81238f3bde3079` digantikan kerana ia memaparkan kriteria AUC untuk MRR; angka sama dan run asal kekal. Checkpoint, parameter, metric mengikut epoch dan kawalan disimpan di MLflow tempatan.

Semakan bebas dan batas tafsiran: [laporan critic](TEXT8_FREQUENCY_ROLE_FACTORIAL_CRITIC.md).

Langkah seterusnya: reka perbandingan yang memisahkan frequency prior daripada struktur yang dipelajari pada validation berbeza dan rancang parity THRML; jangan anggap pilihan terbaik di sini sudah terbukti untuk sense atau perkakasan.
