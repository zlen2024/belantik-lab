# WiC: ujian konteks dan contoh jiran bank

18 November 2026. [WiC](https://pilehvar.github.io/wic/) menguji sama ada satu perkataan mempunyai makna sama dalam dua ayat. Ujian ini menggunakan development validation sahaja, lesen CC-BY-NC 4.0; tiada latihan pada label WiC dan test tidak dibuka. [Protokol](WIC_CONTEXT_PROBE_PROTOCOL.md) menerangkan formula dan pemilihan contoh.

Model Ising asal dilatih pada pasangan perkataan, jadi ujian ini menambah **bacaan konteks baharu yang tidak dilatih secara joint**: posterior empat state untuk target dengan sehingga empat perkataan berdekatan. Skor `−JS` membandingkan posterior dua ayat. Ia menguji readout ini, bukan membuktikan model menghasilkan attention kontekstual atau sense embedding yang sudah terlatih. Vector menerima peraturan bacaan tersendiri; Jaccard dan bilangan konteks ialah kawalan mudah.

## Liputan dan hasil

Daripada 638 contoh validation, hanya **167 (26.2%)** layak: 468 target tiada dalam vocabulary dan 3 tiada konteks yang layak. Subset mengandungi 82 same-sense, 85 different-sense, 151 target. Daripada 1,024 slot konteks, 868 berada dalam vocabulary. Eligibility dan semua skor ditentukan sebelum label dibaca; hasil tidak mewakili seluruh WiC.

| Readout | AUC purata | Seed atau nilai | Bootstrap target 95% |
|---|---:|---|---|
| Ising tied + konteks | 0.5472 | 0.5013 / 0.5603 / 0.5801 | [0.4696, 0.6301] |
| Ising role + konteks | 0.5612 | 0.5504 / 0.5825 / 0.5506 | [0.4917, 0.6422] |
| Vector shared3, target + konteks | 0.5774 | 0.5742 / 0.5681 / 0.5900 | [0.4970, 0.6591] |
| Vector directed3, target + konteks | 0.5652 | 0.5432 / 0.5587 / 0.5937 | [0.4857, 0.6522] |
| Ising initial/random | 0.5489 | 0.5451 / 0.5532 / 0.5485 | [0.4885, 0.6142] |
| Jaccard overlap konteks | 0.5638 | satu kawalan | [0.4845, 0.6442] |
| Vector directed3, konteks sahaja | 0.5596 | 0.5303 / 0.5532 / 0.5953 | [0.4782, 0.6503] |
| Perbezaan bilangan konteks | 0.5367 | satu kawalan | [0.4493, 0.6225] |

Bootstrap meresample target words secara berkelompok dan mengekalkan semua contoh target tersebut; 1,000 resamples bersyarat pada subset/model ini. Semua interval termasuk 0.5. Bacaan Ising role hampir dengan overlap ringkas dan initial; **tiada manfaat makna atau training yang dapat dibuktikan daripada probe ini**. Ia tidak menolak semua kemungkinan energy embedding: latihan konteks sebenar dan vocabulary lebih luas belum diuji. Lemma target benchmark digunakan untuk parameter walaupun perkataan permukaan mungkin inflected; konteks OOV digugurkan dan boleh menghasilkan jurang. Konteks/topik serupa boleh membantu walaupun tiada disambiguation sebenar.

## Contoh `bank` sebagai target

Daripada 2,000 perkataan, pangkat candidate menggunakan score termasuk prior frequency; lebih kecil lebih tinggi. Ini **pemerhatian ilustratif**, bukan gold label WiC dan bukan bukti bahawa dua lembah makna telah dipelajari.

| Model | Seed | `river` | `money` | Lima teratas |
|---|---:|---:|---:|---|
| Ising tied | 17 | 19 | 1,200 | the, of, th, in, south |
| Ising tied | 29 | 26 | 1,212 | the, of, in, th, south |
| Ising tied | 43 | 35 | 1,145 | the, of, in, on, th |
| Ising role | 17 | 876 | 929 | the, a, his, its, an |
| Ising role | 29 | 1,139 | 680 | the, a, his, their, its |
| Ising role | 43 | 325 | 1,253 | of, on, in, at, for |

Contoh ini menunjukkan keputusan sangat dipengaruhi perkataan lazim dan berubah mengikut seed. Dalam tied, `river` jauh lebih tinggi daripada `money`, jadi representasi ini belum menunjukkan kedua-dua makna `bank` dengan baik. Output tanpa prior dan semua state probabilities direkodkan dalam MLflow; state nombor yang sama antara seed tidak semestinya mempunyai makna sama.

## Pembetulan dan audit

Run awal `13bb3b828d694c1f885d8eb0d22a2ec3` dikekalkan dan ditanda superseded: critic menemui `cos(x,x)` dibundarkan secara berbeza di sekitar 1 untuk representasi yang identik. Run `fa6a014eefc549338b7735a9e52206ed` mengikat identiti matematik tepat serta menyusun multiset konteks; tiada tolerance dan tiada perubahan contoh/model. Semua skor Ising kekal sama; perubahan AUC vector sangat kecil. Audit Sol membina semula eligibility dan label, memeriksa 30 checkpoint/readout combinations dan mengesahkan AUC corrected vector dengan 70-digit cosine kepada 1.11e−16; bootstrap turut sepadan.

MLflow experiment `word-energy-wic-context-probe` menyimpan source run IDs, skor setiap contoh, label, liputan dan interval secara tempatan. Data WiC berlesen noncommercial kekal lokal, bukan di Git. Kedua-dua WiC test dan reserved corpus tests tidak dibuka.
