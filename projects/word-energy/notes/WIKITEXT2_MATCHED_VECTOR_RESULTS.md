# WikiText-2: perbandingan bilangan parameter yang hampir sama

18 November 2026. [Protokol](WIKITEXT2_MATCHED_VECTOR_PROTOCOL.md): Ising F2 yang telah dilatih dibandingkan dengan vector 3D baharu menggunakan dataset, prior frequency, loss, noise, optimizer dan epoch yang sama. Enam vector run baharu; enam Ising epoch3 diguna semula. Bilangan parameter hampir sama **tidak menjamin kapasiti representasi atau kesukaran optimization sama**.

| Model | Parameter | AUC micro purata | AUC macro artikel purata | MRR purata |
|---|---:|---:|---:|---:|
| tied-frequency | 6002 | 0.67471 | 0.67405 | 0.30518 |
| shared3 | 6000 | 0.68117 | 0.68135 | 0.31696 |
| role-frequency | 12002 | 0.67725 | 0.67927 | 0.31505 |
| directed3 | 12000 | 0.68470 | 0.68745 | 0.32736 |

| Perbezaan | Micro seed 17 / 29 / 43 | Macro purata | Bootstrap artikel 95% |
|---|---|---:|---|
| role-frequency_minus_directed3 | -0.00878 / -0.00548 / -0.00809 | -0.00818 | [-0.01158, -0.00485] |
| tied-frequency_minus_shared3 | -0.00616 / -0.00709 / -0.00613 | -0.00730 | [-0.01062, -0.00387] |

Kedua-dua model Ising mendapat AUC lebih rendah daripada vector sepadan dalam ketiga-tiga seed untuk panel ini. Interval bootstrap resample 60 artikel, bersyarat pada model, contoh dan seed yang sama; ini development dan bukan ujian architecture umum. Tiada parity spin–vector dari segi kapasiti, graph atau kos fizikal dibuktikan.

## Semua run

| Model | Seed | Run ID | AUC | Macro AUC | MRR |
|---|---:|---|---:|---:|---:|
| tied-frequency | 17 | `d848a6d5a711462795a4476589346b33` | 0.67563 | 0.67343 | 0.30588 |
| tied-frequency | 29 | `00139333a47d4afb959bbc2c849e9a47` | 0.67385 | 0.67524 | 0.30519 |
| tied-frequency | 43 | `5bd444057a9c49b9a2df5f5ad933b035` | 0.67464 | 0.67347 | 0.30448 |
| shared3 | 17 | `4108db8ef76b4294a0f2bed1fe05d878` | 0.68179 | 0.68212 | 0.31778 |
| shared3 | 29 | `b54970eb179047aa9b9f751bb29f4224` | 0.68094 | 0.68176 | 0.31651 |
| shared3 | 43 | `6f37e349c8234687aabee076e4a089a0` | 0.68078 | 0.68017 | 0.31659 |
| role-frequency | 17 | `e6f37d07e0fa42d98185d5882ee50bb2` | 0.67664 | 0.67825 | 0.31375 |
| role-frequency | 29 | `5fcbf994b7ef4e0892e7c97366c964c8` | 0.67944 | 0.68105 | 0.31831 |
| role-frequency | 43 | `435407d0b8554c80b9eb46fecfff4e2c` | 0.67566 | 0.67851 | 0.31308 |
| directed3 | 17 | `8099f617e9394ef2a98730c496fce48f` | 0.68542 | 0.68822 | 0.32779 |
| directed3 | 29 | `4a55f1441381488c968a6c3de035c8c9` | 0.68492 | 0.68713 | 0.32665 |
| directed3 | 43 | `6084df22a44f41b988b956b022d4e6d5` | 0.68375 | 0.68700 | 0.32763 |

Audit Sol 6.1 Ultra mengira semua nilai dengan implementasi NumPy berasingan dan memeriksa 48 checkpoint serta hash source; beza maksimum AUC/macro 0, MRR 5.55e−17, objective 4.44e−16. Tiada WiC atau reserved test digunakan untuk melatih vector ini.

MLflow experiment `word-energy-wikitext2-matched-vectors`, summary `82f5d9a714474aa1a2eb87572e41510a`. Data, checkpoint dan metrik lengkap disimpan dalam MLflow tempatan.
