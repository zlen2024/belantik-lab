"""Matched uniform-other-candidate control for the shared-star development probe."""
from pathlib import Path
import json
import hashlib
import sys
import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_text8_canonical_evaluator import artifact_dir
from scripts.build_wikitext2_joint_context_data import load
from scripts.evaluate_swords_external_candidates import DATA, ISING, load_checkpoint
from scripts.evaluate_swords_active_r import auc
from scripts.evaluate_swords_shared_star import evaluate_summary
from word_energy.contextual_substitution import ising_shared_correlations
from word_energy.tracking import RunContract, log_contract, tracking_uri

def main():
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    origin = artifact_dir(client, '6bfbcadfb039410dbc206f963b50d2bf')
    shared = artifact_dir(client, '88afd4e4710a4da3b36080581905ff07')
    manifest = json.loads((origin/'input_manifest.json').read_text(encoding='utf8'))
    digest = hashlib.sha256(json.dumps(manifest['items'], sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()
    if digest != manifest['manifest_sha256'] or len(manifest['items']) != 219:
        raise ValueError('manifest changed')
    if len({item['target_id'] for item in manifest['items']}) != 219:
        raise ValueError('duplicate target IDs in manifest')
    _, metadata = load(DATA)
    models = {s: load_checkpoint(client, ISING[s], metadata) for s in (17,29,43)}
    predictions = []
    for item in manifest['items']:
        ids = item['candidate_ids']
        out = {'target_id': item['target_id'], 'seeds': {}}
        for seed, params in models.items():
            expanded = dict(params)
            expanded['biases'] = np.vstack((params['biases'], np.zeros((1,4))))
            expanded['internal'] = np.vstack((params['internal'], np.zeros((1,6))))
            dummy = len(params['biases'])
            values = []
            for candidate in ids:
                leaves = np.array([candidate]+[dummy]*(len(ids)-1), dtype=np.int64)
                values.append(float(ising_shared_correlations(expanded,item['target'],item['context'],leaves)[0]))
            if not np.isfinite(values).all() or np.any(np.abs(values) > 1 + 1e-12):
                raise FloatingPointError('invalid dummy correlation')
            out['seeds'][str(seed)] = values
        predictions.append(out)
    historical = {x['target_id']: x for x in json.loads((origin/'target_results.json').read_text(encoding='utf8'))}
    prior = {x['target_id']: x for x in json.loads((shared/'target_results.json').read_text(encoding='utf8'))}
    if len(historical) != 132 or set(historical) != set(prior):
        raise ValueError('historical subset mismatch')
    rows=[]
    for item, pred in zip(manifest['items'], predictions):
        tid=item['target_id']
        if tid not in historical: continue
        fractions=historical[tid]['true_fraction']
        if (historical[tid]['source'] != item['source'] or prior[tid]['source'] != item['source']
                or len(fractions) != len(item['candidate_ids'])):
            raise ValueError('historical source or candidate alignment mismatch')
        mask=np.array([v is not None for v in fractions])
        labels=np.array([v is not None and v>.5 for v in fractions])[mask]
        row={'target_id':tid,'source':item['source'],'scores':{}}
        for seed in (17,29,43):
            key=str(seed)
            prev=prior[tid]['scores'][key]
            row['scores'][key]={'shared':auc(np.asarray(pred['seeds'][key])[mask],labels),
                'uniform':prev['shared'], 'active_r':prev['active_r'], 'vector_contextual':prev['vector_contextual']}
        rows.append(row)
    if len(rows)!=132: raise ValueError('subset mismatch')
    result={'all':evaluate_summary(rows),'without_dominant_source':evaluate_summary([r for r in rows if r['source']!='Nathans_Bylichka.txt']),
            'mapping':'shared in summary is matched dummy; uniform in summary is original shared',
            'limitations':'repeated development; matched control uses candidate-specific systems, no new training or hardware'}
    mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='swords-matched-dummy-control') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','matched dummy Ising','raw spin correlation','four spins {-1,+1}','beta=1','exact star',23140,manifest['manifest_sha256'],'reused development','4(1+m+K) spins',result['limitations']),result)
        mlflow.log_dict(result,'summary.json'); mlflow.log_dict(rows,'target_results.json'); mlflow.log_dict(predictions,'predictions.json')
        for p in (Path(__file__),ROOT/'notes/SWORDS_MATCHED_DUMMY_PROTOCOL.md'): mlflow.log_artifact(str(p),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps(result),flush=True)

if __name__=='__main__': main()
