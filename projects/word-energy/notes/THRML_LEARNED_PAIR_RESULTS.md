# THRML: parity simulator untuk pasangan Ising yang telah dilatih

18 November 2026. [Protokol asal](THRML_LEARNED_PAIR_PROTOCOL.md) menggunakan checkpoint WikiText role Ising seed17 dan pasangan `bank→river` serta `bank→money`. Empat spin, empat edge, dua block Gibbs; bandingkan taburan 16 state dan moments dengan enumeration exact. Ini menggunakan parameter yang telah dipelajari, berbeza daripada fixture enam-spin lama yang dipilih secara manual.

| Audit | Samples setiap chain | Gate | Julat total variation | Max R-hat | Min ESS | Max MCSE |
|---|---:|---|---|---:|---:|---:|
| Asal `219a97eea88f4a3487b0aaa87f29fb79` | 2,000 | **gagal**: 9/30 observable melebihi MCSE 0.01 | 0.00205–0.01178 | 1.00210 | 2,238 | 0.01520 |
| Susulan `a5c8caebda284e4f9fd3b1271cc449db` | 8,000 | **lulus**: 30/30 pada ambang sama | 0.00162–0.00507 | 1.00061 | 8,485 | 0.00729 |

Setiap pasangan diuji dengan empat chain dan tiga seed sampler. Susulan dipilih selepas kegagalan asal, direkod dan dibekukan sebagai [audit ketepatan adaptif](THRML_LEARNED_PAIR_PRECISION_FOLLOWUP.md); ambang tidak dilonggarkan. Dua spin memerlukan lebih daripada 8,000 effective samples untuk capai MCSE 0.01 walaupun IID, jadi budget asal tidak mencukupi. FFT diagnostics diuji terhadap implementasi lama; audit bebas Sol memeriksa semua 30 observable dan samples, dengan perbezaan ESS maksimum 1.1e−11.

**Kesimpulan terhad:** taburan simulator THRML bagi dua graph terlatih sepadan dengan enumeration dalam kriteria susulan. Sampel ini tidak secara langsung mengira perbezaan partition function/free energy F2. Prior frekuensi masih pada host. Tiada cip fizikal, latency/energy advantage, compiler mapping atau bukti menyeluruh untuk semua graph diuji.

MLflow experiment `word-energy-thrml-learned-pair-parity`; kedua-dua run, parameter, state probability, diagnostics dan raw samples dikekalkan secara lokal.
