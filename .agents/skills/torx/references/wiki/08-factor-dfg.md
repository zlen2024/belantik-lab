# 08. Factor dan DFG

Sumber: notebook 15 dan 16, `torx/factor.py`, `torx/dfg.py`, `torx/composite_factors.py`.

## Factor

Satu taburan bersyarat dibungkus sebagai objek. Ia baca input dan parameter bernama setiap panggilan, pulangkan satu output stokastik. Parameter kekal di luar objek, jadi factor boleh digunakan semula dengan nilai berbeza.

Kelas asas: `AbstractFactor`, `AbstractReferenceFactor`, `AbstractHasLogProbability`, `AbstractEnumerableOutputFactor`, `DeterministicFactor`, dll. Port diisytihar dengan `jax.ShapeDtypeStruct` (bentuk dan dtype tanpa data). Factor tersuai melaksanakan `sample` dan `init_params`; boleh tambah `log_probability` untuk rujukan tepat.

## DFG

Output satu factor disambung ke input yang lain, tanpa kitar, jadi disampel sekali mengikut urutan ibu-anak. `Site`, `DFG`, `DFGInfo`, `DFGParams`.

## Komposit

- `TiledFactor`: ulang satu factor atas banyak tapak (dengan weight tying).
- `ChainFactor`: rantai dalam siri.

## Satu antara muka di semua peringkat

`sample(key, inputs, params)` sama untuk satu pbit, satu graf, atau satu komposit. Gate Torx juga factor; beza ialah berapa banyak penempatan sudah ditetapkan (PSC memasang factor dalam wayar litar).

## Contoh 16

Sweep Gibbs 4×4 torus: factor satu spin + papan dam + graf berulang. Medan sebagai input membenarkan pattern completion tanpa mengubah wayar.

## Tips

- Beri factor dengan parameter di luar; jangan simpan nombor dalam objek.
- Jika perlu rujukan tepat, laksanakan `log_probability` dan bandingkan.
