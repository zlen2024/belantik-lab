# Paper berkaitan masalah multi-sense, energy dan konteks

22 November 2026. Bacaan terarah bagi soalan pengguna; ini bukan eksperimen atau reproduction. Kaitan kepada Ising di bawah ialah cadangan kami, bukan klaim paper mengenai chip.

## 1. Athiwaratkun & Wilson (2017), Multimodal Word Distributions

[Paper ACL](https://aclanthology.org/P17-1151/) · [kod penulis word2gm](https://github.com/benathi/word2gm). **Paper ini sudah dipetik dalam pelan Gaussian v2 projek**, jadi bukan penemuan baru kepada pelan. Namun ia sangat dekat dengan contoh asal pengguna: §1 dan §4.3 membincangkan `bank` sebagai sungai dan kewangan. Table 1 menunjukkan jiran bagi dua komponen Gaussian yang masing-masing berkaitan dengan sungai dan kewangan; ini pemerhatian kualitatif penulis, bukan jaminan setiap komponen bagi setiap perkataan sepadan satu sense.

Mereka mewakili word type sebagai campuran Gaussian dan mengukur pertindihan dua density melalui expected likelihood kernel `K(w,c)=∫p_w(z)p_c(z)dz`. Training §3.3 menggunakan perkataan dalam window sebagai positive dan random negative, subsampling perkataan kerap, negative distribution unigram pangkat 3/4, serta max-margin ranking. Mereka melaporkan perbandingan pada tugas similarity dan entailment; hasil asal tidak membuktikan prestasi atau dinamik konteks Ising kita. D=50 dan window=10 ialah tetapan evaluasi lazim dalam paper, bukan parameter optimum sejagat.

**Perbezaan tanda:** paper ini memanggil overlap `E` dan memaksimumkan similarity tersebut untuk pasangan positif. Kontrak energy projek lazimnya menggunakan skor rendah sebagai lebih sesuai; `−log K` ialah satu transformasi yang menukar arah apabila `K>0`. Jangan menyalin simbol/tanda tanpa menyatakan perbezaannya. Gaussian mixture ini tidak sendirinya menghasilkan satu sistem Ising bersama yang disampel pada hardware, dan overlap pasangan sahaja tidak membuktikan occupancy bergantung pada ayat.

## 2. Neelakantan et al. (2014), Multi-Sense Skip-gram

[Paper ACL](https://aclanthology.org/D14-1113/). Masalah: satu vektor statik sukar membezakan polysemy. §4 mengelompokkan representasi konteks untuk memilih sense-vector semasa melatih, dan mengemas kini sense-vector yang dipilih. Paper menguji skip-gram multi-sense dan cara menganggarkan jumlah sense, menggunakan konteks local.

Kaitan kepada Ising ialah **hipotesis**: latih atau uji supaya context mempengaruhi occupancy beberapa kawasan state yang dapat dipisahkan secara bebas, berbanding hanya meningkatkan skor pasangan. Model mereka menggunakan beberapa vektor dan context clusters; ia bukan model Ising atau bukti bahawa bilangan basin sama dengan bilangan makna.

## 3. Garí Soler et al. (2019), Context-sensitive Models for Lexical Substitution

[Paper ACL](https://aclanthology.org/W19-0423/). Masalahnya sepadan dengan ujian projek: memilih kata pengganti yang sesuai dengan ayat. Mereka membandingkan model konteks dan mendapati dalam tetapan LexSub mereka, model yang secara eksplisit memodelkan hubungan word–context dapat lebih baik daripada beberapa representasi kontekstual umum. Mereka membandingkan ranking candidate pada subset SemEval-2007 dan metrik GAP. Ini hasil dalam konfigurasi dan dataset paper, bukan hukum bahawa contextual embeddings umumnya lemah.

Kaitan kepada projek: beri model konteks yang mencukupi, tetapi juga **ajar dan ukur** sama ada hubungan word–context digunakan. Keputusan empat cue TWSI kami tidak membuktikan cukup konteks atau objektif yang sepadan. Kawalan cue-only, target-only dan cue-shuffle, serta baseline kontekstual yang kuat adalah penting; hasil bergerak daripada unary sahaja tidak membuktikan sense betul.

## 4. Vilnis & McCallum (2015), Word Representations via Gaussian Embedding

[Paper arXiv / ICLR](https://arxiv.org/abs/1412.6623). Mewakili perkataan sebagai Gaussian distribution dan mengkaji uncertainty serta hubungan asimetri; menyediakan asas density embedding sebelum campuran Gaussian. Satu Gaussian mudah adalah unimodal; menambah koordinat semata-mata tidak menjaminkan banyak lembah. Ini relevan kepada baseline Gaussian dan motivasi untuk mixture, tetapi bukan kaedah Ising yang disampel pada chip.

## Rujukan hasil projek dan tindakan yang dicadangkan

Semakan projek [empat cue](TWSI_FOUR_CUE_PILOT_RESULTS.md) dan [posterior](TWSI_POSTERIOR_REGION_DIAGNOSTIC_RESULTS.md) menunjukkan context boleh mengubah taburan tetapi belum membuktikan sense yang betul. Paling berguna untuk dibaca mendalam dahulu: paper 1 §3–4 untuk representation, objective, sampling dan evaluasi, kemudian paper 3 untuk konteks dan baseline lexical substitution. Paper 2 memberikan kawalan multi-sense vector yang lebih relevan daripada single vector sahaja.

Laluan yang munasabah apabila kerja disambung: bina Gaussian mixture lexical baseline yang benar-benar boleh dibanding dengan Word2GM; kekalkan Ising sebagai laluan utama, dengan joint graph dan baca occupancy/correlation yang telah ditetapkan; ukur pada contoh natural-text dan sense anchor yang betul, bersama multi-sense/contextual vector baseline dan controls pada input/data sama. Semak perubahan taburan semantik secara berasingan daripada skor retrieval, dan semak parity THRML serta kekangan hardware sebagai langkah lain. Tidak menganggap hasil dalam paper sebagai hasil kita dan tidak menamakan reproduksi sebelum dijalankan.
