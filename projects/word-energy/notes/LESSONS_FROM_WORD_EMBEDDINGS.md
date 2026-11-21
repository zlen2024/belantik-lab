# Apa yang boleh dipelajari daripada word embedding untuk Word Energy

22 November 2026. Kajian sumber dan tafsiran reka bentuk; bukan eksperimen atau bukti Ising lebih baik. Dua pelan sedia ada masih dikekalkan: Ising sebagai laluan utama menuju hardware dan Gaussian joint sebagai baseline klasik.

## Word2Vec: tugasan latihan membentuk geometri

[Skip-gram dan negative sampling](https://arxiv.org/abs/1310.4546) belajar daripada pasangan perkataan dan konteks yang dijumpai dalam corpus. Ia menaikkan skor pasangan observed dan menurunkan skor pasangan noise yang disampel. Noise bukan label pasti bahawa perkataan itu salah dalam setiap konteks: pasangan yang sah juga boleh tersampel. Kekerapan, subsampling perkataan lazim dan taburan noise mempengaruhi signal. Skip-gram lazimnya mempunyai parameter input dan output berasingan ketika latihan; itu tidak sama dengan kontrak kami yang mengikat satu jadual token untuk Q/K.

**Kaitan cadangan, bukan derivasi setara:** Ising boleh mendefinisikan satu shared joint energy dan menggunakan skor seperti free energy ternormal sebagai compatibility untuk melatih contoh observed berbanding noise. Jika lower F bermakna lebih serasi, objective contohnya `softplus(margin + F_positive - F_negative)` menghukum susunan yang salah. Kena tetapkan graph, suhu, normalisasi unary dan negative distribution; perbezaan free-energy antara window tidak boleh dibaca terus daripada raw chip energy tanpa mengurus offset. Ini tidak menjadikan Ising identik dengan Skip-gram, dan objective free energy tidak automatik mengoptimumkan korelasi R atau sense occupancy. Versi full-softmax dan negative/noise sudah diuji dalam projek; perlu audit mengapa kegunaannya terhad, bukan menganggap ia belum pernah dibuat.

## GloVe: statistik seluruh corpus

[GloVe](https://aclanthology.org/D14-1162/) menggunakan kiraan co-occurrence global dan weighted regression, bukan rawatan satu window semata-mata. Ini memberi idea yang boleh diuji: pastikan perbandingan energy juga mempertimbangkan kekerapan pasangan dan perkataan, dengan kawalan frequency dan pemisahan data yang betul. GloVe bukan bukti bahawa objective Ising tertentu akan berjaya.

## Multi-sense dan contextual embeddings

[Multi-sense Skip-gram](https://aclanthology.org/D14-1113/) pernah mempelajari beberapa vektor untuk satu word type dan membezakan penggunaan mengikut konteks. Jadi idea bahawa `bank` mempunyai beberapa keadaan bukan keunikan energy landscape. Satu Ising unary boleh mempunyai beberapa kawasan state berprobabiliti tinggi, dan joint context boleh mengubah occupancy; tetapi beberapa state tinggi tidak secara automatik membuktikan beberapa makna atau bahawa sampler boleh bergerak antara basin.

[ELMo](https://aclanthology.org/N18-1202/) membina representation setiap occurrence sebagai fungsi konteks melalui bidirectional language model; [BERT](https://aclanthology.org/N19-1423/) juga mempelajari representation daripada context kiri dan kanan melalui pretraining. Pelajaran praktikal ialah konteks mesti mengandungi maklumat yang diperlukan dan objective mesti melatih model menggunakannya. Kedua-duanya menggunakan neural language models, bukan p-bit graph; meniru objective atau menambah attention tidak secara automatik memetakan sistem itu kepada thermodynamic hardware.

[FastText](https://aclanthology.org/Q17-1010/) membina word vector daripada character n-grams dan boleh menghasilkan representasi untuk perkataan yang tiada semasa latihan. Jadual token tetap 2,000 perkataan dalam prototaip kami tidak mempunyai sifat itu. Subword-to-Ising-parameters ialah hipotesis reka bentuk, bukan kemampuan sedia ada; preprocessing dan kos host/chip perlu dilapor.

## Apa yang projek ini sudah tahu

Dalam [perbandingan full-softmax](FULL_SOFTMAX_MATCHED_VECTOR_RESULTS.md), Ising dan vector bertied menggunakan input, objective, jadual dan kiraan parameter yang hampir sama. Pada WikiText development, macro MRR .19800 berbanding .19810; pada EWT development, .12611 berbanding .13483. Ini sudah menjadi baseline berguna tetapi bukan perbandingan dengan word2vec atau BERT produksi; data dan panel development sudah digunakan berulang, dan kos pengiraan tidak dipadankan.

Dalam [pilot empat cue](TWSI_FOUR_CUE_PILOT_RESULTS.md), label berasal dari ayat penuh tetapi model menerima maksimum empat token sekitar target. Kedua-dua model fit training lebih baik daripada validation; membuang konteks sering meninggalkan accuracy yang hampir sama. [Diagnostic posterior](TWSI_POSTERIOR_REGION_DIAGNOSTIC_RESULTS.md) menunjukkan konteks mengubah taburan state, tetapi candidate-derived regions tidak membuktikan semantic sense. Ini konsisten dengan masalah signal, input, generalization atau capacity; belum mengasingkan puncanya.

## Arah yang wajar diuji apabila research disambung

1. Tentukan tugas dan input dahulu: perkataan mana model perlu ramal, konteks apa ia benar-benar lihat, dan apa maksud low energy. Lihat sama ada context yang kelihatan cukup menyokong label; elak mengoptimumkan label ayat penuh menggunakan cue yang tidak mencukupi.
2. Kekalkan Ising pairwise shared system dan baseline Gaussian mengikut kedua-dua pelan. Cadangkan objective self-supervised berasaskan contoh teks serta negatif yang dinyatakan, atau gunakan label sense untuk evaluation/anchor map jika skopnya memerlukan. Pastikan train/test source terpisah dan kawalan frequency, cue-only, target-only dan shuffle dilapor.
3. Bandingkan energy landscape dengan baseline yang benar-benar kuat untuk soalan itu: Skip-gram/CBOW dengan butiran jelas untuk lexical learning; vector contextual atau multi-sense yang sesuai untuk sense dalam ayat. Padankan data/split dan lapor perbezaan parameter serta kos, bukannya memanggil semua baseline setara.
4. Ukur lexical retrieval dan contextual meaning secara berasingan; untuk `bank`, gunakan sense regions yang ditentukan daripada train anchors sahaja dan bandingkan posterior dengan/ tanpa context pada dokumen belum dilihat. Multi-well, response terhadap context dan skor rendah bagi contoh betul ialah tiga pemeriksaan berbeza.
5. Sebelum skala atau claim hardware, semak scorer/gradient Ising secara exact, replika observable pada THRML dengan uncertainty dan mixing, audit graf/degree/mapping backend dan rekod host overhead. GPU/CPU simulator bukan ukuran chip. Gaussian memerlukan joint natural-text comparator sendiri.

Ini susunan cadangan, bukan protokol yang telah dibekukan. Tiada eksperimen baharu atau claim bahawa Word Energy sudah reliable dijalankan dalam kajian ini.
