# Fresh synthetic Ising benchmark: results

Follow-up: the [eight-permutation exploratory panel](ISING_CORPUS227_PERMUTATION_PANEL_RESULTS.md)
has now been completed on selected corpus 227. It does not change this gate.

16 November 2026. All 72 scheduled runs completed in 2,828.31 seconds; the frozen six-corpus promotion gate **failed**. Source and protocol were committed before this panel as `1178334`. See [protocol](ISING_FRESH_SYNTHETIC_BENCHMARK_PROTOCOL.md) and [independent corpus 227 critic](ISING_FRESH_CORPUS_227_CRITIC.md).

Each cell below is the number of four paired model seeds passing all three joint100 thresholds: sense >= 10/12, lexical AUC >= 0.75, and mean context gain >= 0.03. The condition required spectral to exceed both controls on every corpus, so its tie with the fixed permutation on corpus 227 fails the gate.

| Corpus | Random | Spectral | Permuted |
|---|---:|---:|---:|
| 211 | 2/4 | 4/4 | 1/4 |
| 223 | 1/4 | 4/4 | 0/4 |
| 227 | 2/4 | 4/4 | 4/4 |
| 229 | 2/4 | 4/4 | 0/4 |
| 233 | 3/4 | 4/4 | 3/4 |
| 239 | 3/4 | 4/4 | 0/4 |

Spectral passed 24/24 runs; random passed 13/24; permuted passed 8/24. These seeds are nested within six corpora; do not interpret the 24 runs as 24 independent datasets. The one fixed permutation is shared numerically across corpora. The pass counts at lexical400 were also 13/24, 24/24, and 8/24 respectively, so these results do not establish that the joint stage was necessary. All three arms had lexical AUC 1.0 on every run, so AUC alone did not distinguish the controls on this toy problem.

The source audit checks paired internal and interaction hashes, data and schedule hashes, initializer/control metadata, 400-to-100 handoff, budgets, run identity, and endpoint gates. GPT-6.1 Sol Ultra independently audited all 72 runs and recomputed 144 lexical400/joint100 endpoints using NumPy enumeration, with maximum absolute error 6.66e-16 and zero gate disagreements. Its focused corpus 227 analysis is in the separate critic note. Peak recorded process working set was 547.4 MiB. MLflow experiment `word-energy-fresh-synthetic-v2`, summary run `3255f7ed95a04073b964fdfb55586eca`, contains all detailed configuration, metrics and links to the individual runs.

This is a small synthetic development benchmark with the same oracle-assisted grammar; word types occur in training. It does not establish performance on unseen words or natural text, nor execution on thermodynamic hardware. No reserved test split was evaluated. The per-process working-set measurements are snapshot-based, not an OS hard limit; thread settings were requested via environment variables.

## Run ledger

Each run ID is searchable in MLflow. Sense, AUC and gain below are joint100 endpoints; the full record also contains lexical400 and per-family metrics, provenance and checkpoint details.

| Corpus | Seed | Arm | MLflow run ID | Sense | AUC | Gain | Pass |
|---|---:|---|---|---:|---:|---:|---|
| 211 | 251 | permuted | `57bbc4643e784f019be564457ccecd98` | 0.500 | 1.000 | 0.000000 | no |
| 211 | 251 | random | `78712296e4544df481426c6ffb3f3fb4` | 0.500 | 1.000 | 0.107949 | no |
| 211 | 251 | spectral | `0f638ba6d9064195b5d7f746ac82fbeb` | 1.000 | 1.000 | 0.444583 | yes |
| 211 | 257 | permuted | `01738fb6d0fa4e21a9c6388fcb7238fa` | 0.500 | 1.000 | 0.000000 | no |
| 211 | 257 | random | `cf491305f2794a86a2b5e1efb8e6ca69` | 1.000 | 1.000 | 0.446098 | yes |
| 211 | 257 | spectral | `70dad07cecfa4689b1d35441ac787430` | 1.000 | 1.000 | 0.445107 | yes |
| 211 | 263 | permuted | `f3a7efeddb07460d8b34e77ad6085cd7` | 1.000 | 1.000 | 0.444631 | yes |
| 211 | 263 | random | `fcf33261a61d4b2195e6df4b38b58989` | 1.000 | 1.000 | 0.445859 | yes |
| 211 | 263 | spectral | `94704468b83c4c3cb6ae54d4745fa259` | 1.000 | 1.000 | 0.444903 | yes |
| 211 | 269 | permuted | `354f5593175e4c8da20e31a6f5a3aca6` | 0.500 | 1.000 | 0.000009 | no |
| 211 | 269 | random | `d5ba64c577c94a0f8a4a13cf9c0c4f6c` | 0.500 | 1.000 | 0.000000 | no |
| 211 | 269 | spectral | `b9b36a01a8e14139abdd3b6953dab0fb` | 1.000 | 1.000 | 0.445280 | yes |
| 223 | 251 | permuted | `77bd7d0f86eb4523a26becae11509b2a` | 0.500 | 1.000 | 0.000000 | no |
| 223 | 251 | random | `24124fedd9f44adba0a7e0bd0d5d6c26` | 0.500 | 1.000 | 0.000001 | no |
| 223 | 251 | spectral | `62806e5ee6d847fcb715ca2b12c78c1c` | 1.000 | 1.000 | 0.444580 | yes |
| 223 | 257 | permuted | `28173e534768404484ce0e938ee73e74` | 0.500 | 1.000 | 0.000001 | no |
| 223 | 257 | random | `62908b3b761647929e0bc02a79c1647e` | 0.500 | 1.000 | 0.016962 | no |
| 223 | 257 | spectral | `a0420208e1fb48f6a8f3a0db0e497e9f` | 1.000 | 1.000 | 0.444929 | yes |
| 223 | 263 | permuted | `8bf25d6fea004886bd8bd33cbcf0ff83` | 0.500 | 1.000 | -0.000000 | no |
| 223 | 263 | random | `04c4de441d444303a9acfdd7c1743e54` | 1.000 | 1.000 | 0.446625 | yes |
| 223 | 263 | spectral | `d16ee9f000844bee8ed11f2073466965` | 1.000 | 1.000 | 0.444811 | yes |
| 223 | 269 | permuted | `3f37aa921c404e30a6f30fae31753a07` | 0.500 | 1.000 | 0.000001 | no |
| 223 | 269 | random | `01eb104680d4469080adf07f7854d643` | 0.500 | 1.000 | 0.000000 | no |
| 223 | 269 | spectral | `49ab8920a26448798bc3442726ca1fec` | 1.000 | 1.000 | 0.446009 | yes |
| 227 | 251 | permuted | `24d33345120a4166892e864b530a3f72` | 1.000 | 1.000 | 0.490223 | yes |
| 227 | 251 | random | `24053e7bca264701871ff69cbe8e62ad` | 1.000 | 1.000 | 0.445103 | yes |
| 227 | 251 | spectral | `d4b3e7caa3ee44fb8807f1fc88a23b44` | 1.000 | 1.000 | 0.445676 | yes |
| 227 | 257 | permuted | `186b1886014c4a61a71fc350ebff1eed` | 1.000 | 1.000 | 0.492165 | yes |
| 227 | 257 | random | `ee39823b23994938b186585dfe5a343f` | 0.500 | 1.000 | 0.000001 | no |
| 227 | 257 | spectral | `d9604f3221c6403a8bb8fcaf3a743372` | 1.000 | 1.000 | 0.445564 | yes |
| 227 | 263 | permuted | `feac3cbe3e8048ff90b18cbbb27c1d65` | 1.000 | 1.000 | 0.444279 | yes |
| 227 | 263 | random | `64f1c4ae770f4fd7be47e66603d13047` | 1.000 | 1.000 | 0.445274 | yes |
| 227 | 263 | spectral | `01eb12adaec34d71ac8d8cc4a89fd2b2` | 1.000 | 1.000 | 0.444827 | yes |
| 227 | 269 | permuted | `1197c00282e74b889d7720a76e1aacdd` | 1.000 | 1.000 | 0.443092 | yes |
| 227 | 269 | random | `e375dc93f98a4bef8e97eb3526f339e6` | 0.500 | 1.000 | 0.000000 | no |
| 227 | 269 | spectral | `9c9d68e98a024e7e9be8bb1461404120` | 1.000 | 1.000 | 0.445078 | yes |
| 229 | 251 | permuted | `ae9b28f1211c4b4ea13a7c314b8f45bf` | 0.500 | 1.000 | 0.207932 | no |
| 229 | 251 | random | `0034fdae765b4503970b8f7f31277889` | 0.500 | 1.000 | 0.000002 | no |
| 229 | 251 | spectral | `854878536f62447d95877d9a9fd3d463` | 1.000 | 1.000 | 0.445311 | yes |
| 229 | 257 | permuted | `6557df51baac472f807b26711e684ddb` | 0.500 | 1.000 | 0.000160 | no |
| 229 | 257 | random | `9e73d05ab30a42a9bfe2d3780d18b412` | 1.000 | 1.000 | 0.444852 | yes |
| 229 | 257 | spectral | `865b749511b64e81826cf8a2aaeaac96` | 1.000 | 1.000 | 0.444571 | yes |
| 229 | 263 | permuted | `ce3b85431bf64c25a30c45c6ae73a5fe` | 0.500 | 1.000 | 0.000000 | no |
| 229 | 263 | random | `9119f2478fcd4d4cbdacfa77364206b8` | 0.500 | 1.000 | 0.000350 | no |
| 229 | 263 | spectral | `57b9c9ad81ca4fa9b9b55e347f23aefa` | 1.000 | 1.000 | 0.444812 | yes |
| 229 | 269 | permuted | `4a956ec9313c4bd0a02800775df195c3` | 0.500 | 1.000 | 0.057742 | no |
| 229 | 269 | random | `367f773a27554d8db592d8a0f12be0b3` | 1.000 | 1.000 | 0.444665 | yes |
| 229 | 269 | spectral | `c03e0414f6f04997a9c65c5303cbff06` | 1.000 | 1.000 | 0.445028 | yes |
| 233 | 251 | permuted | `df37f5395aab4a2f828d771d0c7ed8ce` | 1.000 | 1.000 | 0.445810 | yes |
| 233 | 251 | random | `150f4e7112ed411787b224946ea37b3b` | 0.500 | 1.000 | 0.000003 | no |
| 233 | 251 | spectral | `9601450f3ce4401ab5bdffdd0fb9f706` | 1.000 | 1.000 | 0.445190 | yes |
| 233 | 257 | permuted | `5e99b52db46642bb9beb4c1c0c219936` | 1.000 | 1.000 | 0.445738 | yes |
| 233 | 257 | random | `3bb9f2a5bdb24d6bac1f69120d5d65c5` | 1.000 | 1.000 | 0.492468 | yes |
| 233 | 257 | spectral | `07ed82cc315740dfba13a3d4f49192ee` | 1.000 | 1.000 | 0.444884 | yes |
| 233 | 263 | permuted | `5b41891c478a44b38056ae3d716520e1` | 0.500 | 1.000 | 0.018563 | no |
| 233 | 263 | random | `20e7cae430764372a37a0f055a38a81a` | 1.000 | 1.000 | 0.440164 | yes |
| 233 | 263 | spectral | `80919b7c7f1f4652b7e2b00b906482d5` | 1.000 | 1.000 | 0.444968 | yes |
| 233 | 269 | permuted | `dcc93597d57c4dada2f974eb5548e4ae` | 1.000 | 1.000 | 0.445992 | yes |
| 233 | 269 | random | `dea850c348f34f80aa26a9d2c4d61828` | 1.000 | 1.000 | 0.446490 | yes |
| 233 | 269 | spectral | `38c5cf58ab7c4faeb816ccff98dc225b` | 1.000 | 1.000 | 0.445314 | yes |
| 239 | 251 | permuted | `cda73bb22dc046e8854783025e0dd32d` | 0.500 | 1.000 | 0.000003 | no |
| 239 | 251 | random | `63df462e521d4a09b1496273d2a1e7bf` | 1.000 | 1.000 | 0.443987 | yes |
| 239 | 251 | spectral | `d345036e96ef49399e1438ea41662b00` | 1.000 | 1.000 | 0.444976 | yes |
| 239 | 257 | permuted | `7839822a49b04cc48721e904c8943b7f` | 0.500 | 1.000 | 0.000112 | no |
| 239 | 257 | random | `9f2203c7ef1844d2b37e8f10eb286774` | 1.000 | 1.000 | 0.446228 | yes |
| 239 | 257 | spectral | `58f9c0bf276c497a8b7712911f7c3e5b` | 1.000 | 1.000 | 0.445062 | yes |
| 239 | 263 | permuted | `53776e6db9a14834ae2efd6194ed1729` | 0.500 | 1.000 | 0.000002 | no |
| 239 | 263 | random | `ab4155a71c44466f9caae524ff8e32b3` | 1.000 | 1.000 | 0.492596 | yes |
| 239 | 263 | spectral | `c5958d4049094a11a0f3eb47324f1595` | 1.000 | 1.000 | 0.445057 | yes |
| 239 | 269 | permuted | `70493e0901054c29a9535828a7b2bae8` | 0.500 | 1.000 | 0.000017 | no |
| 239 | 269 | random | `64b164f1bbf148288b460f2efa3b60c2` | 0.500 | 1.000 | 0.000006 | no |
| 239 | 269 | spectral | `b88b56f5da814a67a68095d0b0a0d960` | 1.000 | 1.000 | 0.445311 | yes |
