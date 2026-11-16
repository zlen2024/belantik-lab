"""Write a human-readable ledger from the already audited MLflow summary."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_RUN = '3255f7ed95a04073b964fdfb55586eca'
summary = json.loads((ROOT / 'mlruns' / '8' / SUMMARY_RUN / 'artifacts' / 'config.json').read_text(encoding='utf-8'))
if summary['panel_size'] != 72 or summary['strict_six_corpus_promotion']:
    raise ValueError('This report is specific to the audited 72-run failed-gate panel')

lines = [
    '# Fresh synthetic Ising benchmark: results',
    '',
    'Follow-up: the [eight-permutation exploratory panel](ISING_CORPUS227_PERMUTATION_PANEL_RESULTS.md) '
    'has now been completed on selected corpus 227. It does not change this gate.',
    '',
    '16 November 2026. All 72 scheduled runs completed in 2,828.31 seconds; '
    'the frozen six-corpus promotion gate **failed**. Source and protocol were '
    'committed before this panel as `1178334`. See '
    '[protocol](ISING_FRESH_SYNTHETIC_BENCHMARK_PROTOCOL.md) and '
    '[independent corpus 227 critic](ISING_FRESH_CORPUS_227_CRITIC.md).',
    '',
    'Each cell below is the number of four paired model seeds passing all '
    'three joint100 thresholds: sense >= 10/12, lexical AUC >= 0.75, '
    'and mean context gain >= 0.03. The condition required spectral to '
    'exceed both controls on every corpus, so its tie with the fixed '
    'permutation on corpus 227 fails the gate.',
    '',
    '| Corpus | Random | Spectral | Permuted |',
    '|---|---:|---:|---:|',
]
for corpus, rates in summary['per_corpus_pass_rates'].items():
    lines.append(f"| {corpus} | {round(4*rates['random'])}/4 | {round(4*rates['spectral'])}/4 | {round(4*rates['permuted'])}/4 |")
lines += [
    '',
    'Spectral passed 24/24 runs; random passed 13/24; permuted passed 8/24. '
    'These seeds are nested within six corpora; do not interpret the 24 runs '
    'as 24 independent datasets. The one fixed permutation is shared numerically '
    'across corpora. The pass counts at lexical400 were also 13/24, 24/24, '
    'and 8/24 respectively, so these results do not establish that the joint '
    'stage was necessary. All three arms had lexical AUC 1.0 on every run, so '
    'AUC alone did not distinguish the controls on this toy problem.',
    '',
    'The source audit checks paired internal and interaction hashes, data and '
    'schedule hashes, initializer/control metadata, 400-to-100 handoff, '
    'budgets, run identity, and endpoint gates. GPT-6.1 Sol Ultra independently '
    'audited all 72 runs and recomputed 144 lexical400/joint100 endpoints '
    'using NumPy enumeration, with maximum absolute error 6.66e-16 and zero '
    'gate disagreements. Its focused corpus 227 analysis is in the separate '
    'critic note. Peak recorded process working set was 547.4 MiB. MLflow experiment '
    f'`word-energy-fresh-synthetic-v2`, summary run `{SUMMARY_RUN}`, contains '
    'all detailed configuration, metrics and links to the individual runs.',
    '',
    'This is a small synthetic development benchmark with the same '
    'oracle-assisted grammar; word types occur in training. It does not '
    'establish performance on unseen words or natural text, nor execution '
    'on thermodynamic hardware. No reserved test split was evaluated. '
    'The per-process working-set measurements are snapshot-based, not an OS '
    'hard limit; thread settings were requested via environment variables.',
    '',
    '## Run ledger',
    '',
    'Each run ID is searchable in MLflow. Sense, AUC and gain below are '
    'joint100 endpoints; the full record also contains lexical400 and '
    'per-family metrics, provenance and checkpoint details.',
    '',
    '| Corpus | Seed | Arm | MLflow run ID | Sense | AUC | Gain | Pass |',
    '|---|---:|---|---|---:|---:|---:|---|',
]
for key, run in summary['runs'].items():
    corpus, seed, arm = key.split(':')
    m = run['joint100']
    lines.append(f"| {corpus} | {seed} | {arm} | `{run['run_id']}` | {m['sense']:.3f} | {m['auc']:.3f} | {m['gain']:.6f} | {'yes' if run['joint100_pass'] else 'no'} |")
path = ROOT / 'notes' / 'ISING_FRESH_SYNTHETIC_BENCHMARK_RESULTS.md'
path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
print(path)
