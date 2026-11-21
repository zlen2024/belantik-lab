# Pilot empat slot konteks TWSI

21 November 2026. Code/protokol `1995175`; MLflow `8f2b8e9cfa8344d6bed4c2155e003cda`. Tiga checkpoint Ising dan vector sedia ada dilatih semula pada baris TWSI dan split eksploratori yang sama dengan pilot satu cue. Context dipilih mengikut offset −1,+1,−2,+2; empat slot tetap termasuk dummy uniform apabila token tidak tersedia. Daripada 10,227 contoh, 587 mempunyai satu token terpakai, 2,601 dua, 3,954 tiga, dan 3,085 empat. Lebih banyak slot tidak menjamin petunjuk semantik yang diperlukan berada di situ.

| Model | Pair accuracy validation: 1 cue → 4 slot (seed 17/29/43) | Reversal group semua betul: 1 cue → 4 slot |
|---|---|---|
| Ising | .581→.574 / .591→.602 / .548→.578 | 7→5 / 4→5 / 5→3 daripada 121 |
| Vector | .571→.566 / .565→.558 / .568→.569 | 14→6 / 9→6 / 10→10 daripada 121 |

Ising empat slot mencapai train pair accuracy .855/.856/.845; vector .807/.805/.807. Accuracy validation dengan context dibuang ialah .577/.596/.579 Ising dan .566/.568/.583 vector. Apabila context bundle diacak sekali, strict reversal Ising ialah 3/7/10 dan vector 8/4/13 daripada 121; untuk context sebenar ia 5/5/3 dan 6/6/10. Context removal menghasilkan 0 strict reversal untuk setiap model secara pembinaan. Satu acakan bukan distribution null dan group bukan iid.

Keputusan tidak menunjukkan penambahbaikan stabil pada tugasan reversal atau bukti bahawa konteks difahami. Perubahan daripada satu cue turut mengubah normalisasi spoke Ising (g/3 menjadi g/6), padding dan cara vector menggabungkan konteks, jadi kesan itu tidak boleh ditetapkan pada bilangan perkataan semata-mata. Ini set development yang sudah digunakan berulang dan label adalah weak preference berdasarkan ayat penuh; pembinaan conflict-filter pra-split dan ketiadaan source IDs mengehadkan generalisasi. Tiada Gaussian, THRML, uji chip atau final test.

CPU elapsed sekitar 20 saat dan peak working set yang diperiksa semasa epoch 464.4 MiB; nombor bukan pengukuran kos chip. Dua ujian scorer exact/gradient/padding lulus sebelum run. Semakan critic 6.2 Sol medium direkod berasingan.
