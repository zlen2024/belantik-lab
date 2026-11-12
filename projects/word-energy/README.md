# Word Energy research

Ruang kerja baharu untuk Word Energy Embedding. Ini scaffold; model baharu belum diimplementasi atau dilatih.

- [Pelan utama Ising/Z1](plans/word2energy/WORD_ENERGY_Z1_FIRST_PLAN.md)
- [Pelan Gaussian baseline](plans/word2energy/WORD_ENERGY_EMBEDDING_V2_PLAN.md)
- [Idea dan teks asal](plans/word2energy/RUMUSAN_IDEA_BIKAR_ENERGY_EMBEDDING.md)
- [Rekod critic Ising](plans/word2energy/reviews/z1_first/REVIEW_LOG.md) dan [critic Gaussian](plans/word2energy/reviews/energy_embedding_v2/REVIEW_LOG.md)
- [Keputusan experiment tracker](TRACKING_DECISION.md): MLflow lokal dipilih pengguna.

Skor critic menilai pelan, bukan hasil latihan atau bukti pemetaan chip. Protokol asal dan test-set lama dikekalkan; baca plan sebelum eksperimen. Legacy masih berada di `../../word2energy` sehingga archive disahkan. Keperluan tracker langsung dicatat dalam `requirements-tracking.txt`; freeze environment yang diuji dan database lokal berada di `tracking/` dan perlu dibackup, bukan di-commit.
