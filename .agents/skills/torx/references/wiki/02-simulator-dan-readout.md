# 02. Simulator dan readout

Sumber: notebook 04, `basic_usage`, kod `torx/psc/simulation/`.

## Readout = purata fungsi

`<f, ρ> = Σ_a f(a) ρ_a`. Dua penganggar:

- Tepat: `StateVectorSimulator` enumerasi `ρ_out = K ρ_in`.
- Sampel: `BranchingSimulator.sample(circuit, x, key)` beri `(num_samples, num_pbits)` integer, penganggar `(1/N) Σ f(s_n)`.

Histogram = `np.bincount`. `f(s) = s_i` beri `expval_all`.

## Kadar 1/√N

Jarak total-variation `TV(ρ̂_N, ρ) = O(N^{-1/2})`. Notebook 04 menyemak dua perkara berasingan: (a) ketumpatan dan min dalam `atol = 0.04` pada N = 5000; (b) cerun log-log ralat antara −0.85 dan −0.2 (diukur −0.480). Kita nampak perkara sama untuk dot product: 100× lebih sampel untuk 10× lebih tepat (`torx_learn.py`: 124 → 38 → 11 → 3.3).

## Pembezaan

`BranchingSimulator(diff_method=..., num_samples=...)`:

| Kaedah | Kos | Nota |
|---|---|---|
| `param_shift_inf` | `(1 + G·K_max)·S` sampel | guna gate θ → ±∞ |
| `param_shift_single` | satu penilaian tambahan setiap gate | K=2 sahaja; `reps` mesti 1 |
| `param_shift_filter` | satu laluan | penganggar nisbah (min bersyarat daripada sampel yang sama); bias nyata pada N kecil dengan cabang jarang dan, untuk kerugian tak linear, bias bertanda bergantung gate (formula delta dalam `../gradients.md`); sahaja yang menyokong `reps > 1` |

`build_circuit` boleh dipanggil **dalam** `jax.jit(jax.grad(...))` (corak `tests/test_gradients.py`); mengkompil di luar hanya pilihan kelajuan. Hanya `expval`/`expval_all` boleh dibezakan; `sample()` tidak. Lihat `../gradients.md`.

## Bila guna yang mana

- Litar ≤ ~20 pbit atau untuk rujukan: `StateVectorSimulator` (diukur dengan `equinox.filter_jit(sv.density)`, `cookbook_11_train.py` bahagian 7: n=20, 2²⁰ keadaan: kira-kira 20 ms; n=24 kira-kira 0.6 s; kos naik kira-kira ×5–6 setiap 2 pbit (20 ms pada n=20 ke 0.6 s pada n=24). Panggilan tanpa jit didominasi overhead dispatch dan hampir rata. `jax.jit(sv.density)` gagal dengan `TypeError`; guna `filter_jit`).
- Litar besar: `BranchingSimulator` (kita guna 576 PNOT, jalan lancar).
- `PISING` sahaja dengan state-vector.
