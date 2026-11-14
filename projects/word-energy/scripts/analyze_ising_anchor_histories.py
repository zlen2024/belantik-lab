"""Read-only per-token synthesis of saved Ising lexical histories."""
from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlflow
from scripts.analyze_ising_trajectory import STATES, snapshot_stats
from word_energy.ising.exact_training import _free_energy_factory, prepare_data
from word_energy.synthetic import POOLS, vocabulary
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

CASES = {
    'original29': ('local', 'ising-attribution-original-seed29.json'),
    'original43': ('local', 'ising-attribution-original-seed43.json'),
    'bank29_from43': ('mlflow', '1f1703f05c2345388878f7e6cf4a941c'),
    'bank43_from29': ('mlflow', 'e542fda97aa0442fa7b390f23c0b72b4'),
    'anchors29_from43': ('mlflow', '18a3b4c0af9248f58299234ea381f70f'),
    'anchors43_from29': ('mlflow', 'ec16dbc2d3294b10be99aa19eb9cd2d1'),
    'bias29_from43': ('mlflow', '0fa55ff18422464593bc755f0aba849a'),
    'bias43_from29': ('mlflow', 'ca6d88b60cd14aeb9b76d5d5eb285c9d'),
}

def source_path(kind, name):
    return ROOT / ('experiments/runs' if kind == 'local' else 'mlruns/3') / (name if kind == 'local' else f'{name}/artifacts/config.json')

def unary_all(raw):
    b = 2.5*np.tanh(np.asarray(raw['biases']))
    internal = 2*np.tanh(np.asarray(raw['internal']))
    logits = b @ STATES.T + internal[:,None]*STATES[:,0]*STATES[:,1]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)

if __name__ == '__main__':
    data = prepare_data()
    jax,jnp,energy,_,_ = _free_energy_factory()
    pairs=jnp.asarray(data['train_pairs']); neg=jnp.asarray(data['train_negatives'])
    def lexical(raw):
        pos=jax.vmap(lambda row:energy(raw,row))(pairs)
        negative=jax.vmap(lambda left,row:jax.vmap(lambda right:energy(raw,jnp.array([left,right],dtype=jnp.int32)))(row))(pairs[:,0],neg)
        return jnp.mean(jax.nn.softplus(1+pos[:,None]-negative))
    loss=jax.jit(lexical)
    counts=Counter(int(token) for row in data['train_windows'] for token in row[1:])
    ids=data['ids']; state_names=['--','+-','-+','++']; output={}
    for case,(kind,name) in CASES.items():
        source=source_path(kind,name)
        payload=json.loads(source.read_text(encoding='utf-8'))
        snapshots=payload['snapshots']
        expected=['initial_0']+[f'lexical_{step}' for step in range(20,401,20)]
        if any(label not in snapshots for label in expected): raise ValueError(f'{case}: missing required snapshot')
        history={}
        for label in expected:
            raw=snapshots[label]; p=unary_all(raw)
            family={}; tokens={}
            for group in ('F','R'):
                indices=[ids[token] for token in POOLS[group]]
                probabilities=p[indices]
                group_mean=probabilities.mean(axis=0)
                dominant_counts=Counter(state_names[int(i)] for i in np.argmax(probabilities,axis=1))
                family[group]={'unweighted_mean':group_mean.tolist(),'dominant_state_counts':dict(dominant_counts),'mean_max_probability':float(probabilities.max(axis=1).mean()),'mean_within_family_pair_tv':float(np.mean([np.abs(probabilities[i]-probabilities[j]).sum()/2 for i in range(8) for j in range(i+1,8)]))}
                for token,idx in zip(POOLS[group],indices):
                    tokens[token]={'family':group,'probabilities':p[idx].tolist(),'dominant_state':state_names[int(np.argmax(p[idx]))],'max_probability':float(p[idx].max()),'entropy':float(-np.sum(p[idx]*np.log(p[idx]))),'train_occurrences':counts[idx]}
            stats=snapshot_stats(raw,data)
            params={k:jnp.asarray(v,dtype=jnp.float64) for k,v in raw.items()}
            history[label]={'tokens':tokens,'family':family,'weighted_anchor_tv':stats['anchor_tv'],'context_tv':stats['context_tv'],'train_full_lexical_loss':float(loss(params))}
        output[case]={'source_sha256':sha256_file(source),'source_kind':kind,'source_id_or_file':name,'history':history}
    mlflow.set_tracking_uri(tracking_uri()); mlflow.set_experiment('word-energy-exact-training')
    contract=RunContract(execution_tier='classical',device_class='cpu_gpu_simulator',model_family='ising_anchor_history_synthesis',energy_convention='dimensionless beta E; normalized Ising',variable_encoding='spins {-1,+1}',beta_temperature_convention='beta=1',sampler_schedule='read-only exact enumeration on saved snapshots',seed=0,data_hash=sha256_file(ROOT/'experiments/runs/synthetic-split-manifest.json'),data_split='training anchors and validation diagnostics only',graph_summary='2 spins per word; synthetic lexical pairs',limitations='Exploratory descriptive synthesis of two model seeds and reciprocal interventions; no new training; no causal claim for an individual word; no hardware.')
    with mlflow.start_run(run_name='ising-per-anchor-history-synthesis-seeds29-43') as run:
        log_contract(mlflow,contract,{'state_order':state_names,'cases':output,'caution':'state labels are only comparable within the same run unless explicitly aligned; no individual causal inference'})
        for case,case_data in output.items():
            for label,entry in case_data['history'].items():
                mlflow.log_metrics({f'{case}.{label}.anchor_tv':entry['weighted_anchor_tv'],f'{case}.{label}.context_tv':entry['context_tv'],f'{case}.{label}.train_full_lexical_loss':entry['train_full_lexical_loss']})
        print(json.dumps({'run_id':run.info.run_id,'cases':len(output),'snapshots_per_case':21,'source_hashes':{k:v['source_sha256'] for k,v in output.items()}}))
