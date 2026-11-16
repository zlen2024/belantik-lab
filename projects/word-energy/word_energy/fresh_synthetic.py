"""Fresh synthetic pair-combination corpora for split-level replication.

Opaque token identities are independent per replicate. This is an artificial
grammar control, not natural text or evaluation on unseen word types.
"""
from itertools import combinations
import numpy as np

from word_energy.synthetic import Window

REPLICATES = (211, 223, 227, 229, 233, 239)


def make_fresh_corpus(replicate: int) -> dict:
    if replicate not in REPLICATES:
        raise ValueError('replicate not in the preregistered set')
    opaque = tuple(f'v2_{replicate}_{i:02d}' for i in range(24))
    membership_rng = np.random.default_rng(np.random.SeedSequence([replicate, 10]))
    assignment = membership_rng.permutation(24)
    families=('F','R','M')
    pools = {family:tuple(opaque[int(i)] for i in assignment[k*8:(k+1)*8]) for k,family in enumerate(families)}
    splits = {'train':[], 'validation':[], 'test':[]}
    pair_index_splits = {}
    for index,family in enumerate(families):
        pair_list=list(combinations(range(8),2))
        rng=np.random.default_rng(np.random.SeedSequence([replicate,20,index]))
        while True:
            permutation=rng.permutation(len(pair_list))
            train=permutation[:16]
            degree=np.bincount(np.asarray([v for i in train for v in pair_list[int(i)]]),minlength=8)
            if np.min(degree)>=2:
                break
        indices={'train':permutation[:16], 'validation':permutation[16:22], 'test':permutation[22:]}
        pair_index_splits[family]={name:values.tolist() for name,values in indices.items()}
        center='song' if family=='M' else 'bank'
        for name,values in indices.items():
            splits[name].extend(Window(family,(center,*(pools[family][v] for v in pair_list[int(i)]))) for i in values)
    return {'replicate':replicate,'vocabulary':('bank','song',*opaque),'pools':pools,'splits':{name:tuple(rows) for name,rows in splits.items()},'pair_index_splits':pair_index_splits}
