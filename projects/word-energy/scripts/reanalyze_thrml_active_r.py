"""Reanalyze same-sweep R in the previous, failed THRML parity run."""
from pathlib import Path
import json
import sys

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.audit_thrml_four_spin_joint import EDGES, STATES, diag
from scripts.build_wikitext2_joint_context_data import load
from word_energy.contextual_substitution import ising_active_correlations
from word_energy.text8_development import sha_array
from word_energy.tracking import RunContract, log_contract, tracking_uri

SOURCE = 'ce2be3890c8e43bc9e74b4897feedc87'
DATA = 'f1b7db5231ed4a5788b566e388f8e731'


def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    _, metadata = load(DATA)
    root = artifact_dir(client, SOURCE)
    config = json.loads((root/'config.json').read_text(encoding='utf8'))
    if not config['run_complete'] or config['dataset_run_id'] != DATA:
        raise ValueError('source run not complete or data mismatch')
    if config['graph_edges'] != [list(edge) for edge in EDGES]:
        raise ValueError('source graph differs')
    checkpoint_root = artifact_dir(client, config['source_model_run'])
    checkpoint_config = json.loads((checkpoint_root/'config.json').read_text(encoding='utf8'))
    if checkpoint_config['dataset_hashes'] != metadata['hashes']:
        raise ValueError('checkpoint data mismatch')
    with np.load(checkpoint_root/'checkpoints/epoch3.npz', allow_pickle=False) as checkpoint:
        params = dict(checkpoint)
    hashes = {k: sha_array(v) for k,v in params.items()}
    if hashes != config['checkpoint_hashes'] or hashes != checkpoint_config['checkpoint_hashes']['epoch3']:
        raise ValueError('checkpoint hash mismatch')
    exact_observable = np.mean(STATES[:, :4] * STATES[:, 8:12], axis=1)
    results = {}
    with np.load(root/'samples/spin_samples.npz', allow_pickle=False) as samples:
        for fixture in ('river', 'finance'):
            source_fixture = config['results'][fixture]
            ids = source_fixture['ids']
            probability = np.asarray(source_fixture['exact_state_probability'])
            if probability.shape != (4096,) or not np.isclose(probability.sum(), 1, rtol=0, atol=1e-12):
                raise ValueError('invalid exact law')
            expected = float(probability @ exact_observable)
            production = float(ising_active_correlations(params, ids[0], [ids[1]], np.array([ids[2]]))[0])
            np.testing.assert_allclose(production, expected, rtol=0, atol=1e-12)
            observations = {}
            for seed, historical in zip((17, 29, 43), source_fixture['runs']):
                chain = samples[f'{fixture}_{seed}']
                if chain.shape != (4, 8000, 12) or not np.isin(chain, (-1, 1)).all() or historical['seed'] != seed:
                    raise ValueError('sample shape, encoding, or seed mismatch')
                state_id = np.sum((chain > 0)*(1 << np.arange(12)), axis=2)
                empirical = np.bincount(state_id.ravel(), minlength=4096)/state_id.size
                np.testing.assert_array_equal(empirical, np.asarray(historical['state_probability']))
                trace = np.mean(chain[:, :, :4]*chain[:, :, 8:12], axis=2)
                observations[str(seed)] = diag(trace, expected)
            results[fixture] = {'tokens': source_fixture['words'], 'ids': ids, 'exact_R': expected,
                                'production_R': production, 'seeds': observations}
    output = {'source_run': SOURCE, 'checkpoint_run': config['source_model_run'], 'checkpoint_hashes': hashes,
              'graph_edges': config['graph_edges'], 'results': results,
              'limitations': 'posthoc scalar reanalysis of old THRML simulator chains, original overall gates failed; no physical hardware, new simulation, or ranking claim'}
    mlflow.set_experiment('word-energy-thrml-four-spin-parity')
    with mlflow.start_run(run_name='posthoc-active-R-existing-chains') as run:
        log_contract(mlflow, RunContract('thrml_simulator', 'cpu_gpu_simulator', 'posthoc_active_R_reanalysis',
            'E=-sum b*s-sum J*s*s; R=mean target-candidate spin products', 'spins {-1,+1}', 'beta=1',
            'reused 4-color block Gibbs, warmup500, draws8000, step1', 17, DATA,
            'two illustrative triples; no test', '12 spins, 26 edges, degree5', output['limitations'],
            chain_count=4, burn_in=500, sample_count=8000), output)
        mlflow.log_dict(output, 'active_r.json')
        for fixture, item in results.items():
            for seed, values in item['seeds'].items():
                mlflow.log_metric(f'{fixture}.{seed}.R', values['sample'])
        for path in (Path(__file__), ROOT/'notes/THRML_ACTIVE_R_REANALYSIS_PROTOCOL.md',
                     ROOT/'word_energy/contextual_substitution.py', ROOT/'scripts/audit_thrml_four_spin_joint.py'):
            mlflow.log_artifact(str(path), 'source_snapshot')
        print('RUN', run.info.run_id, json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
