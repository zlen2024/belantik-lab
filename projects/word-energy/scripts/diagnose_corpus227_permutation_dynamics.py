"""Posthoc diagnostics of saved corpus-227 checkpoints; never trains or edits runs.

Reads MLflow SQLite in read-only mode and saved JSON artifacts. Gradients below
are independent NumPy enumeration gradients of the full canonical training
objective, not the stochastic minibatch gradient or saved Adam momentum.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from word_energy.fresh_synthetic import make_fresh_corpus
from word_energy.ising.exact_training import prepare_data

STAGES = ('stage1_initial_0', 'stage1_lexical_20', 'stage1_lexical_100',
          'stage1_lexical_400', 'stage2_joint_100')
FIELDS = ('biases', 'internal', 'interaction')
U = (2 * ((np.arange(4)[:, None] >> np.arange(2)) & 1) - 1).astype(float)
STATES = {n: (2 * ((np.arange(1 << (2*n))[:, None] >> np.arange(2*n)) & 1)
              - 1).reshape(-1, n, 2).astype(float) for n in (2, 3)}


def flatten(raw):
    return np.concatenate([np.asarray(raw[k]).reshape(-1) for k in FIELDS])


def unflatten(vector, template):
    result = {}; offset = 0
    for key in FIELDS:
        shape = np.asarray(template[key]).shape
        size = int(np.prod(shape))
        result[key] = vector[offset:offset+size].reshape(shape)
        offset += size
    return result


class Exact:
    def __init__(self, raw):
        self.raw = {k: np.asarray(raw[k], dtype=float) for k in FIELDS}
        self.b = 2.5*np.tanh(self.raw['biases'])
        self.k = 2*np.tanh(self.raw['internal'])
        self.g = 1.5/(1+np.exp(-self.raw['interaction']))
        score = U@self.b.T + (U[:, 0]*U[:, 1])[:, None]*self.k
        maximum = score.max(axis=0)
        weights = np.exp(score-maximum)
        self.logz = maximum + np.log(weights.sum(axis=0))
        self.prob = (weights/weights.sum(axis=0)).T
        self.mean = self.prob@U
        self.product = self.prob@(U[:, 0]*U[:, 1])

    def windows(self, tokens):
        tokens = np.asarray(tokens)
        states = STATES[tokens.shape[1]]
        products = states[:, :, 0]*states[:, :, 1]
        cross = (states[:, 0, None, :]*states[:, 1:, :]).sum(axis=1)/(tokens.shape[1]-1)
        score = np.einsum('skd,nkd->ns', states, self.b[tokens])
        score += np.einsum('sk,nk->ns', products, self.k[tokens])
        score += (cross@self.g)[None, :]
        maximum = score.max(axis=1)
        weights = np.exp(score-maximum[:, None])
        prob = weights/weights.sum(axis=1)[:, None]
        return dict(tokens=tokens,
            free=-maximum-np.log(weights.sum(axis=1))+self.logz[tokens].sum(axis=1),
            spin=np.einsum('ns,skd->nkd', prob, states),
            product=prob@products, cross=prob@cross)

    def gradient(self, stats, coefficient):
        tokens = stats['tokens']; coefficient = np.asarray(coefficient)
        gb = np.zeros_like(self.b); gk = np.zeros_like(self.k)
        for position in range(tokens.shape[1]):
            ids = tokens[:, position]
            np.add.at(gb, ids, coefficient[:, None]*(self.mean[ids]-stats['spin'][:, position]))
            np.add.at(gk, ids, coefficient*(self.product[ids]-stats['product'][:, position]))
        gg = -(coefficient[:, None]*stats['cross']).sum(axis=0)
        return {'biases': gb*2.5*(1-np.tanh(self.raw['biases'])**2),
                'internal': gk*2*(1-np.tanh(self.raw['internal'])**2),
                'interaction': gg*self.g*(1-self.g/1.5)}

    def ranking(self, positive, negatives, margin, mask=None):
        count, alternatives = negatives.shape[:2]
        a = self.windows(positive)
        b = self.windows(negatives.reshape(-1, negatives.shape[-1]))
        difference = margin+a['free'][:, None]-b['free'].reshape(count, alternatives)
        losses = np.logaddexp(0, difference)
        coefficient = 1/(1+np.exp(-difference))
        if mask is not None:
            coefficient = coefficient*mask
        first = self.gradient(a, coefficient.sum(axis=1)/difference.size)
        second = self.gradient(b, -coefficient.reshape(-1)/difference.size)
        grad = {k: first[k]+second[k] for k in FIELDS}
        return float(np.mean(losses)), grad, losses


def corpus_objects():
    data = prepare_data(corpus=make_fresh_corpus(227))
    left = np.repeat(data['train_pairs'][:, 0, None], 5, axis=1)
    data['negative_pairs'] = np.stack([left, data['train_negatives']], axis=-1)
    families = {data['ids'][t]: f for f, pool in data['pools'].items() for t in pool}
    data['cross_fr_mask'] = np.asarray([[families.get(int(a)) in ('F','R')
        and families.get(int(b)) in ('F','R') and families[int(a)] != families[int(b)]
        for b in row] for a, row in zip(data['train_pairs'][:, 0], data['train_negatives'])])
    data['anchors'] = {f: [int(t) for w,y in zip(data['train_windows'], data['train_families'])
                          if y == f for t in w[1:]] for f in ('F','R','M')}
    return data


def objectives(raw, data):
    model = Exact(raw)
    lexical, lg, ll = model.ranking(data['train_pairs'], data['negative_pairs'], 1.0)
    ranking, rg, _ = model.ranking(data['train_windows'], data['train_corruptions'], .25)
    joint = ranking+.25*lexical
    jg = {k: rg[k]+.25*lg[k] for k in FIELDS}
    return model, lexical, joint, lg, jg, ll


def diagnose(raw, data):
    model, lexical, joint, lg, jg, ll = objectives(raw, data)
    bank = model.prob[data['ids']['bank']]
    q = {f: model.prob[t].mean(axis=0) for f,t in data['anchors'].items()}
    cross = data['cross_fr_mask']
    _, cross_grad, _ = model.ranking(data['train_pairs'], data['negative_pairs'], 1.0, mask=cross)
    state = U[int(np.argmax(bank))]
    flip_costs = sorted((2*state*(model.b[0]+model.k[0]*state[::-1])).tolist())
    raw_centroids = {f: model.raw['biases'][[data['ids'][t] for t in pool]].mean(axis=0)
                     for f,pool in data['pools'].items()}
    return dict(bank_entropy=float(-sum(bank*np.log(bank))), bank_prior=bank.tolist(),
        bank_single_spin_flip_costs_sorted=flip_costs,
        raw_FR_centroid_distance=float(np.linalg.norm(raw_centroids['F']-raw_centroids['R'])),
        train_anchor_q={k:v.tolist() for k,v in q.items()},
        anchor_tv=float(abs(q['F']-q['R']).sum()/2),
        train_F2_loss=lexical, train_joint_total_loss=joint,
        cross_FR_negative_loss=float(ll[cross].mean()),
        cross_FR_loss_contribution=float((ll*cross).mean()),
        cross_FR_mean_ranking_slope=float((1-np.exp(-ll[cross])).mean()),
        cross_FR_full_objective_gradient_norm=float(np.linalg.norm(flatten(cross_grad))),
        full_lexical_gradient_norm=float(np.linalg.norm(flatten(lg))),
        full_joint_gradient_norm=float(np.linalg.norm(flatten(jg))),
        bank_lexical_gradient=[*lg['biases'][0].tolist(), float(lg['internal'][0])],
        bank_joint_gradient=[*jg['biases'][0].tolist(), float(jg['internal'][0])],
        bank_effective_bias=model.b[0].tolist(), bank_effective_internal=float(model.k[0]),
        raw_bias_norm=float(np.linalg.norm(model.raw['biases'])),
        raw_internal_norm=float(np.linalg.norm(model.raw['internal'])))


def load_saved():
    connection = sqlite3.connect((ROOT/'tracking/mlflow.db').as_uri()+'?mode=ro', uri=True)
    rows = connection.execute("SELECT r.run_uuid,r.artifact_uri,e.name FROM runs r JOIN experiments e "
        "ON r.experiment_id=e.experiment_id WHERE r.status='FINISHED' AND e.name IN "
        "('word-energy-corpus227-permutation-panel-v1','word-energy-fresh-synthetic-v2')").fetchall()
    result = {}
    for run_id, uri, experiment in rows:
        tags = dict(connection.execute('SELECT key,value FROM tags WHERE run_uuid=?', (run_id,)))
        if tags.get('replicate') != '227' or 'seed' not in tags:
            continue
        if experiment.endswith('permutation-panel-v1') and 'permutation_seed' in tags:
            name = 'perm'+tags['permutation_seed']
        elif experiment.endswith('synthetic-v2') and tags.get('arm') in ('spectral','random','permuted'):
            name = tags['arm'] if tags['arm'] != 'permuted' else 'perm6061'
        else:
            continue
        path = Path(uri.removeprefix('file:'))/'config.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        result[(name, int(tags['seed']))] = (run_id, config,
            hashlib.sha256(path.read_bytes()).hexdigest())
    connection.close()
    if len(result) != 44:
        raise ValueError(f'expected 44 saved runs, got {len(result)}')
    return result


def transform(raw, axes, signs):
    return {'biases': np.asarray(raw['biases'])[:, axes]*np.asarray(signs),
        'internal': np.asarray(raw['internal'])*int(np.prod(signs)),
        'interaction': np.asarray(raw['interaction'])[list(axes)]}


def numerical_checks(saved, data):
    rng = np.random.default_rng(81731)
    checks = []
    for name in ('perm7019','perm7043'):
        raw = saved[(name,251)][1]['snapshots']['stage2_joint_100']
        _, lexical, joint, lg, jg, _ = objectives(raw, data)
        base = flatten(raw); finite = []
        for _ in range(2):
            direction = rng.normal(size=base.shape); direction /= np.linalg.norm(direction)
            for eps in (1e-4, 1e-5):
                plus = objectives(unflatten(base+eps*direction, raw), data)
                minus = objectives(unflatten(base-eps*direction, raw), data)
                for index, gradient in ((1, lg), (2, jg)):
                    numerical = (plus[index]-minus[index])/(2*eps)
                    analytic = float(flatten(gradient)@direction)
                    finite.append(abs(numerical-analytic))
        symmetry = []
        for axes in ((0,1),(1,0)):
            for signs in itertools.product((-1,1), repeat=2):
                changed = transform(raw, axes, signs)
                _, a, b, ga, gb, _ = objectives(changed, data)
                symmetry.append(max(abs(a-lexical), abs(b-joint),
                    float(np.max(abs(flatten(ga)-flatten(transform(lg,axes,signs))))),
                    float(np.max(abs(flatten(gb)-flatten(transform(jg,axes,signs)))))))
        curvature = []
        for axis in (0,1):
            direction = {k:np.zeros_like(np.asarray(raw[k])) for k in FIELDS}
            for family, sign in (('F',1),('R',-1)):
                for token in data['pools'][family]:
                    direction['biases'][data['ids'][token],axis] = sign/4
            delta = flatten(direction)
            for eps in (1e-2,3e-3):
                plus = objectives(unflatten(base+eps*delta,raw),data)
                minus = objectives(unflatten(base-eps*delta,raw),data)
                curvature.append(dict(axis=axis, epsilon=eps,
                    lexical_directional_gradient=float(flatten(lg)@delta),
                    lexical_directional_curvature=float((plus[1]-2*lexical+minus[1])/eps**2),
                    joint_directional_gradient=float(flatten(jg)@delta),
                    joint_directional_curvature=float((plus[2]-2*joint+minus[2])/eps**2)))
        checks.append(dict(run=name, finite_difference_max_abs_error=max(finite),
            signed_axis_symmetry_max_abs_error=max(symmetry), FR_directional_curvatures=curvature))
        if max(finite) > 1e-7 or max(symmetry) > 1e-10:
            raise ValueError(f'numerical gradient/symmetry verification failed: {name}')
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/diagnostics/corpus227-permutation-dynamics-20261116.json')
    args = parser.parse_args()
    data = corpus_objects(); saved = load_saved(); output = {}
    for (name, seed), (run_id, config, sha) in sorted(saved.items()):
        output[f'{name}:{seed}'] = dict(run_id=run_id, config_sha256=sha,
            fixed_joint100=config['fixed_joint100'],
            stages={stage:diagnose(config['snapshots'][stage],data) for stage in STAGES})
    checks = numerical_checks(saved, data)
    result = dict(label='posthoc read-only diagnostic, no training',
        protocol_sha256=data['protocol_sha256'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations='Selected corpus and posthoc diagnostics; full canonical gradients exclude minibatch noise and Adam state. '
            'Directional finite differences are evaluations of perturbed frozen parameters, not optimization. '
            'No existing run, checkpoint or protocol was edited; no test evaluation.',
        source_runs=output, numerical_checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8', newline='\n')
    print(json.dumps(dict(output=str(args.output), runs=len(saved), checkpoints=len(saved)*len(STAGES), numerical_checks=checks)))


if __name__ == '__main__':
    main()
