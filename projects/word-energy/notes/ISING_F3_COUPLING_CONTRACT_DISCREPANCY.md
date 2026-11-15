# Audit perbezaan coupling F3: pelan vs implementation

15 November 2026. Status: disahkan; pilihan plan-conformant telah dilaksanakan
dan diuji pada validation, sambil mengekalkan default legacy. Ditemui semasa
critique 6.1 Sol Ultra untuk strategi initialization.

Pelan `WORD_ENERGY_Z1_FIRST_PLAN.md` mentakrifkan F3 dengan dua spoke dan
pekali `lambda/2` setiap spoke. Kod `word_energy/ising/exact_training.py`
(`window_free_energy` dan `joint_center_probability`) serta reference
`word_energy/ising/model.py` (`physical_energy`) menggunakan pekali `lambda`
setiap spoke tanpa pembahagian dua. Pelan F2 pula menetapkan pekali `lambda`
bagi satu spoke dan ini sepadan dengan implementation F2.

Known-answer check: pada `beta=lambda=1`, semua bias/internal sifar,
W=(1,1), dan semua spin +1: F2 physical energy = -2 dalam kedua-dua kontrak;
F3 implemented = -4 sedangkan F3 mengikut pelan = -2. Ini menyemak **energy
satu state**, bukan perbezaan numerik free energy atau performance.

## Apa yang boleh disimpulkan

- Pilot Ising dan run attribution/transplant yang telah dibuat menggambarkan
  sistem yang **dilaksanakan dengan coupling F3 lebih kuat**, bukannya
  implementasi tepat formula F3 yang ditulis dalam pelan.
- Pengukuran F2 lexical dan penemuan bahawa collapse seed29 hadir selepas
  lexical training tidak secara automatik terpadam: lexical F2 menggunakan
  pekali yang sepadan. Namun respons joint, sense/context gate dan sebarang
  hubungan dengan formula pelan mesti dianggap memerlukan audit/pengulangan.
- Parity fixture THRML 6-spin menggunakan model tangan berasingan dan tidak
  membuktikan correctness implementasi trained F3. Ia tidak secara automatik
  menjadi tidak sah bagi fixture yang disemak.
- Tiada keputusan baharu tentang sama ada faktor dua menyebabkan collapse
  atau sama ada kontrak yang dibetulkan akan lulus gate.

## Langkah repair yang perlu diaudit

Bina implementation/versi baharu dengan faktor `1/(L-1)` apabila L>1;
pastikan F2 kekal sama dan F3 betul dalam kedua-dua JAX dan NumPy, serta
center marginal, free energy, finite-difference gradients dan lambda=0.
Jangan overwrite run, checkpoints atau notes lama. Training lama boleh
direproduksi sebagai legacy implementation. Bekukan semula eksperimen
validation dan controls sebelum run; test holdout terdahulu telah dibuka dan
tidak boleh dianggap holdout belum disentuh. Semak kemungkinan version lain
atau amendment sebelum menyifatkan mana-mana run sebagai plan-conformant.
