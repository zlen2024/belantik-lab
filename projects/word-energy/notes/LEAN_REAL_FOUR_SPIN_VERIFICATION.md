# Bukti Lean: skor joint 4-spin dan covariance pada nombor nyata

20 November 2026. Sasaran: menghubungkan identiti covariance dengan formula skor pada `word_energy/wikitext_four_spin.py`, termasuk fungsi Boltzmann exp dan bentuk matematik log1p/expm1. Ini bukti matematik pada nombor nyata ideal, bukan proof JAX atau perkakasan.

## Keputusan theorem

`WordEnergy.RealModel.four_spin_code_gap_sign` dan `four_spin_code_gap_negative` menyatakan, untuk sebarang logits sebenar bagi 16 keadaan setiap occurrence, sebarang dua jadual interaksi sebenar dan offset yang sama:

- skor joint tolak skor additive > 0 jika dan hanya jika covariance bagi dua mesej konteks > 0;
- skor joint tolak skor additive < 0 jika dan hanya jika covariance < 0.

Juga, kedua-duanya sifar serentak mengikut trichotomy nombor nyata. Dua occurrence dengan identiti token sama dibenarkan kerana ketiga-tiga vektor logits boleh sama, tetapi keadaan occurrence masih dijumlah secara berasingan. Keputusan membandingkan kedua-dua rumus dengan parameter, konteks dan offset yang sama. Ia tidak mendakwa perbezaan yang sama bagi dua model yang dilatih berasingan, ranking, AUC atau pembelajaran makna.

## Pemetaan dan andaian

- `SpinState = Fin 16`: konfigurasi untuk empat spin. Vektor 16 logits setiap occurrence dianggap sebagai input real sebarang; logits Ising dan coupling bertanda spesifik kod ialah satu pengkhususan. Interaksi dalam theorem ialah jadual real sebarang, jadi formula empat spoke kod termasuk dalam skop matematik dengan mapping indeks yang betul.
- `probability` ialah exp(logit) dibahagi jumlah 16 exp. Lean membuktikan semua kebarangkalian dan pembilang/penormal adalah positif serta jumlah kebarangkalian satu.
- `message` ialah jumlah context probability dikali exp(interaction). Lean membuktikan mesej dan expectation relevan positif sebelum menggunakan log. Beta dan separuh coupling spoke terletak dalam argumen interaksi; theorem sah bagi sebarang nilai real tersebut.
- `delta` ialah jumlah context probability dikali (exp(interaction)-1), yang sepadan secara matematik dengan penggunaan expm1. Lean membuktikan `message = 1 + delta` dan kedua-dua bentuk joint/pair `1 + ...` bersamaan expectation mesej. Kesamaan ini bergantung pada normalisasi yang dibuktikan.
- Dalam real ideal `log1p(x)=log(1+x)` dan `expm1(x)=exp(x)-1`. Lean membuktikan identiti untuk ekspresi ini, tetapi tidak memformalkan pelaksanaan atau pembundaran JAX, enumerasi susunan state table atau peralihan dari array ke definisi Lean. Padanan indeks, signs dan real-to-float masih diuji dan diaudit secara berasingan.

## Persekitaran dan audit

- Lean `leanprover/lean4:v4.36.0` commit `611c117bfaf7b7ae1cfcf246ce6fba31e8f509e9`; Mathlib pin commit `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`. `theory/lean-toolchain`, `lakefile.lean` dan `lake-manifest.json` menyimpan pins. Cache berada dalam `.lake/` dan tidak dimasukkan ke Git.
- `lake build WordEnergy.FourSpinTheorem`: lulus untuk sumber akhir selepas theorem negatif ditambah. Lake mengeluarkan amaran cadangan module-system pada import Mathlib; ia bukan ralat proof.
- `lake env lean FourSpinTheoremAudit.lean`: lulus; declaration akhir dipaparkan dan transitive axiom audit dibuat untuk kedua-dua theorem akhir serta lemma yang diminta. Semua menunjukkan `[propext, Classical.choice, Quot.sound]`; tiada `sorryAx`, custom axiom atau `native_decide` dijumpai. Ini dependency asas Lean yang lazim, bukan bukti tanpa sebarang axiom.
- SHA-256: `RealCovariance.lean` `99E9FAFB775D21A3D4E06BE3C67FD7AF3C880FE7DE10465E19B8C0F20A96DEFE`; `Boltzmann.lean` `0D335512CFA00A6E961961126AFD55A8AE66E796084FBBA685D52F00C28F5B15`; `CodeMapping.lean` `627BC83BC3D92C49EB72F36C19AF64475C47A0487B7022E5253D5BA742F1D3A7`; `FourSpinTheorem.lean` `FF81857F792579C453DAE61376E792BC8E809F1D4734ACF20CAF220384145114`.
- Kelayakan: kernel-checked theorem untuk deklarasi di atas, tertakluk pada axiom standard yang dilapor dan ketepatan pemetaan konsep ke sumber. `.olean` untuk module akhir telah dibina sebelum probe import.

## Semakan bebas terhadap kod

`python scripts/audit_four_spin_covariance_fixture.py` membandingkan penjumlahan 4,096 keadaan, formula factorized dan skor JAX pada dua konfigurasi. Context searah menghasilkan covariance +0.0776814664 dan gap log +0.0407934991; context berlawanan covariance -0.0776814664 dan gap -0.0425286420. Kedua-duanya lulus toleransi 1e-12. `python -m unittest discover -s tests -p 'test_wikitext_four_spin*.py' -v`: 5 tests lulus, termasuk direct enumeration, gradients, repeated IDs dan zero coupling. Ini bukti numerik terbatas pada kes diuji, bukan pengesahan formal seluruh program.

Matematik ini tidak membuktikan bahawa shared context meningkatkan AUC, bahawa model memahami semantic sense, bahawa Gibbs sampler bercampur dengan baik atau bahawa model berjalan pada chip sebenar. Gate THRML yang terdahulu kekal gagal.
