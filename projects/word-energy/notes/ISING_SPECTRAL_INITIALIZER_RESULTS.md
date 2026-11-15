# Keputusan development: spectral bias initialization

15 November 2026. Protokol [initializer](ISING_LABEL_FREE_INITIALIZER_DRAFT.md)
dibekukan sebelum run. Tiga seed `101/103/107`, tiga arms, fixed handoff
lexical400→joint100 dan corrected F3. Setiap run menggunakan exact enumeration
klasik; tiada test evaluation, THRML training atau physical chip execution.

## Hasil setiap run

| Arm / seed | Lexical400 sense | Joint100 sense | F / R betul | Gain joint100 | AUC | Gate toy validation | MLflow run |
|---|---:|---:|---|---:|---:|---|---|
| Random 101 | 0.5 | 0.5 | 0/6 / 6/6 | 0.033314 | 1.0 | gagal | `c9bf31a9746a44b3b65c699e81b6f0ee` |
| Random 103 | 0.5 | 0.5 | 0/6 / 6/6 | 0.000002 | 1.0 | gagal | `ef96b4fe6043436b8be8032be4e1469e` |
| Random 107 | 1.0 | 1.0 | 6/6 / 6/6 | 0.444342 | 1.0 | lulus | `0e8448a6b2464b05824c0da0c14e4ff4` |
| Spectral 101 | 1.0 | 1.0 | 6/6 / 6/6 | 0.445069 | 1.0 | lulus | `52f0c53ff42043c9befdf6cf907748ac` |
| Spectral 103 | 1.0 | 1.0 | 6/6 / 6/6 | 0.444849 | 1.0 | lulus | `e4ce7df27d6243269c9fb73cb90badb1` |
| Spectral 107 | 1.0 | 1.0 | 6/6 / 6/6 | 0.445153 | 1.0 | lulus | `52643ad42d944b73961e900ae0eea112` |
| Permuted 101 | 0.5 | 0.5 | 0/6 / 6/6 | 0.000038 | 1.0 | gagal | `be4a881aa08d46b6a808593187d597c2` |
| Permuted 103 | 0.5 | 0.5 | 0/6 / 6/6 | 0.061485 | 1.0 | gagal | `2ddccd344d814a47a31d7a691085d4c1` |
| Permuted 107 | 0.5 | 0.5 | 0/6 / 6/6 | 0.000007 | 1.0 | gagal | `72e3c0a9e7c54526a658917c54137e12` |

Gate toy di sini menggunakan sense >=10/12, lexical AUC >=0.75 dan context
gain >=0.03. Pass counts: spectral **3/3**, random **1/3**, permuted **0/3**.
Ini keputusan development baru, bukan perubahan status preregistered pilot
lama. Seed107 spectral dan random sama-sama lulus; spectral tidak meningkatkan
accuracy untuk semua pasangan seed secara individu.

Critic juga mengira AUC lexical400 secara bebas: kesemua sembilan mendapat
1.0, dan kiraan lulus ketiga-tiga threshold pada lexical400 sama (random 1/3,
spectral 3/3, permuted 0/3). Ketiga-tiga seed spectral sudah mencapai sense
1.0 pada lexical400, sebelum joint training; ujian ini tidak membuktikan joint
training diperlukan untuk kejayaan toy tersebut.
Audit lexical400 yang dipautkan kepada sembilan artifact ialah MLflow run
`6343919160a4482594a73156edce1072`; ia menyimpan AUC, gain, dan threshold
result setiap arm/seed serta hash source artifact.

## Apa yang boleh disimpulkan

Dengan model/dataset toy ini, bias awal daripada struktur co-occurrence train
menunjukkan signal yang menggalakkan. Satu permutasi tetap yang memecah padanan
word→bias gagal ketiga-tiga seed; ini menyokong bahawa padanan itu mungkin
penting, tetapi satu permutation tidak mencukupi untuk menggeneralisasi.

Bias spectral yang digunakan adalah **identik untuk ketiga-tiga seed**; hanya
raw internal initializations lain yang berubah. Jadi 3/3 bukan tiga geometri
awal spectral bebas. Semua perbandingan menggunakan satu train/validation split
synthetic yang sudah digunakan untuk development. Initializer hanya menerima
training token IDs, tetapi negatives dan corruptions asal masih menggunakan
grammar/family oracle dan sense evaluation menggunakan label training anchors.
Keputusan ini tidak membuktikan pipeline tanpa label, superiority pada natural
text, robustness population, atau hardware feasibility.

## Audit dan resources

Kesemua sembilan run berjaya dan handoff lexical400→joint start disemak bitwise.
Train protocol/schedule hash sepadan. Co-occurrence/eigen/bias/permutation hashes,
exact permutation RNG6061, sumber kod dan snapshots terdapat pada MLflow
`config.json` setiap run. Spectral coordinates mempunyai population SD 0.1;
dua eigenvalue gaps 3.939608 dan 1.930946 melebihi tolerance ~6.92e-8.

Jumlah masa yang direkod oleh sembilan proses sekitar 266.3 saat, tidak termasuk
overhead launch penuh; maksimum process peak working set ~541.4 MiB. Setting
OMP/OPENBLAS empat thread direkod, bukan bukti hard thread limit. Pengawasan
memory berlaku pada snapshot, bukan hard OS limit. Tiada kegagalan dalam batch;
versi runner batch ini hanya menyimpan snapshot pada penghujung, jadi jika ia
terganggu, snapshots dalam memori mungkin hilang. Selepas batch selesai, runner
ditambah pengendalian exception untuk menyimpan partial snapshots dalam MLflow;
ini diuji secara unit dan **tidak mengubah source snapshot sembilan run asal**.
Ia tidak menjamin recovery jika process dibunuh secara paksa. `pytest` tidak dipasang; unit tests spectral dan F3
menggunakan `unittest`.

Critic GPT-6.1 Sol Ultra memeriksa kesemua sembilan rekod, paired initial
parameters, source hashes, fixed handoffs dan 18 fixed endpoints menggunakan
enumeration NumPy bebas. Sense/gain/TV/joint loss sepadan sehingga absolute
error maksimum `5.55e-16`; tiada blocking flaw ditemui.
