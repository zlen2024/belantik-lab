# Semakan AI secara buta: cue berbanding ayat penuh

22 November 2026. Panel eksploratori memilih 12 target berbeza dan 24 baris (12 pasangan reversal) daripada TWSI train yang telah digunakan sebelum ini, dengan protocol `81f4641`. Ia bukan sampel rawak daripada semua bahasa atau dokumen. Dua critic GPT-6.2 Sol medium ditugaskan berasingan: satu hanya melihat target, dua candidate dan bag cue empat slot; satu melihat ayat penuh dan candidate yang sama. Label weak TWSI dan output model terdahulu tidak dimasukkan ke dalam panel mereka.

| Penilai AI | Tidak pasti | Pilihan sepadan label weak TWSI | Kedua-dua arah reversal sepadan |
|---|---:|---:|---:|
| Cue sahaja | 19/24 | 5/24, atau 5/5 daripada pilihan tegas | 0/12 group |
| Ayat penuh | 6/24 | 14/24, atau 14/18 daripada pilihan tegas | 5/12 group |

Kedua-dua penilai memilih dengan tegas pada lima baris yang sama dan bersetuju pada kelima-limanya. Penilai cue menandakan tidak pasti untuk 19 baris, yang **konsisten dengan** konteks yang dipaparkan sering tidak cukup baginya untuk membezakan candidate. Ia tidak membuktikan semua model atau manusia tidak boleh menggunakan cue itu; sampel kecil dipilih khusus kerana label berubah.

Penilai cue menamatkan tugas secara normal. Agent ayat penuh menghasilkan JSON dengan 24 ID unik dan pilihan yang sah tetapi kemudian berakhir dengan usage-limit error sebelum confirmation akhir; status lengkapnya tidak dapat disahkan daripada final agent. Oleh itu hasil ayat penuh ialah artifact yang kelihatan lengkap, dengan caveat provenance tambahan; jangan membuat claim kukuh berasaskannya. Kedua-dua review dilakukan AI, bukan human annotation atau gold truth. Sumber dokumen, near duplicates dan kebebasan contoh belum disahkan. Ayat penuh tidak semestinya menyokong label lemah TWSI; 4/18 pilihan tegas AI berbeza daripada label itu.

MLflow `bacf64ec383e4d2eb92ea50ed966eb2a` menyimpan input blind, key pilihan, output AI, hashes dan ringkasan lokal. Raw text tidak disimpan di Git; data sumber diliputi CC BY-NC-SA 4.0. Tiada training, Gaussian comparison, THRML atau hardware run. Keputusan tidak membuktikan model Word Energy gagal atau menggantikan embedding biasa.
