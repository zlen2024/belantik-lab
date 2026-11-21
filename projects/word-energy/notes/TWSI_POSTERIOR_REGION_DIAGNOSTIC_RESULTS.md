# Diagnostic posterior target dan candidate-derived regions

22 November 2026. Run `1a5d7f26dc1e458bba6baaf3cc5ac3e1`, code/protokol `7ab3802`. Tiga checkpoint empat-cue sebelumnya dinilai pada 2,184 contoh development TWSI yang sama; tiada model dilatih. Region A/B ditakrifkan daripada perbandingan unary distribution kedua-dua candidate pada setiap state, tanpa label atau context. Candidate tidak aktif sebagai perkataan dalam star kali ini: dua slotnya ialah dummy uniform dan coupling kekal g/6. Region ini hanyalah probe yang bergantung pada candidate, bukan sense region yang dibuktikan secara bebas.

| Seed | Purata perubahan taburan target dari unary (TV) | Pair accuracy context sebenar | Pair accuracy unary | Reversal group betul sepenuhnya |
|---|---:|---:|---:|---:|
| 17 | 0.126 | 0.566 | 0.562 | 4/121 |
| 29 | 0.109 | 0.565 | 0.580 | 3/121 |
| 43 | 0.130 | 0.565 | 0.562 | 0/121 |

TV ialah ukuran sejauh mana dua taburan berbeza; 0 bermaksud sama dan 1 paling berbeza. Maka context memang mengubah taburan target secara matematik pada checkpoint ini. Ia **tidak bermaksud** makna betul dipilih: accuracy hampir sama dengan prior unary dan berbeza arah mengikut seed, sementara strict reversal rendah. Context yang diacak sekali mendapat pair accuracy .566/.582/.564 dan reversal 1/2/2 daripada 121. Tiada interval atau ujian bebas; group berkongsi target dan sumber.

Ini bacaan lain daripada skor R yang digunakan semasa latihan dan ujian sebelum ini. Candidate-derived region juga boleh mencerminkan frekuensi atau bentuk unary, dan label TWSI berasal daripada ayat penuh sedangkan model melihat sehingga empat token. Kesimpulan hanya tentang checkpoint, readout dan data ini. Tiada Gaussian, THRML, training baharu atau chip. Ujian exact enumeration kecil dan dummy uniform lulus sebelum run; semakan critic direkod berasingan.
