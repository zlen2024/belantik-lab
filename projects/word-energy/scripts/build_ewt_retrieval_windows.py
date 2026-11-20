"""Build frozen EWT development windows for the out-of-domain retrieval replication."""
from pathlib import Path
import hashlib
import re

import numpy as np
from scripts.build_wikitext2_document_data import corpus_data, norm, parse, sample
from scripts.build_wikitext2_joint_context_data import windows
from word_energy.tracking import sha256_file
from word_energy.text8_development import sha_array

SOURCE = 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.15/en_ewt-ud-dev.conllu'
SHA = '14a82ce2c4c4648c3e7c4efc76872c85c5f4c39643fcb7f9c91a45a58157a270'


def build_ewt(root, vocabulary):
    source = Path(root)/'data/ewt/en_ewt-ud-dev.conllu'
    if sha256_file(source) != SHA:
        raise ValueError('EWT release hash differs')
    documents = []
    for line in source.read_text(encoding='utf8').splitlines():
        if line.startswith('# newdoc id = '):
            documents.append({'title':line.removeprefix('# newdoc id = ').casefold(),'rows':[]})
        elif line.startswith('# text = '):
            documents[-1]['rows'].append(line.removeprefix('# text = '))
    wiki = parse('train')+parse('validation')
    titles = {d['title'] for d in wiki}
    texts = {norm(' '.join(d['rows'])) for d in wiki}
    paragraphs = '\n'.join(norm(row) for d in wiki for row in d['rows'])
    kept = []
    for doc in documents:
        content = norm(' '.join(doc['rows']))
        if doc['title'] in titles or content in texts or any(len(norm(s))>=80 and norm(s) in paragraphs for s in doc['rows']):
            continue
        kept.append(doc)
        titles.add(doc['title'])
        texts.add(content)
    ids,docs = corpus_data(kept,{w:i for i,w in enumerate(vocabulary)})
    ids = np.asarray(ids,np.int32)
    pairs,positions,groups = sample(ids,docs,40000,np.random.default_rng(24101))
    words,pos,group,_,_ = windows(ids,docs,pairs,positions,groups,10000,24102)
    manifest = dict(source=SOURCE,source_sha256=SHA,total_docs=len(documents),kept_docs=len(kept),
                    evaluated_docs=len(np.unique(group)),included_tokens=int((ids>=0).sum()),
                    raw_tokens=int((docs>=0).sum()),unique_positions=len(np.unique(pos,axis=0)),
                    windows_sha256=sha_array(words),positions_sha256=sha_array(pos),groups_sha256=sha_array(group),
                    doc_hashes=[hashlib.sha256(norm(' '.join(d['rows'])).encode()).hexdigest() for d in kept])
    return words,group,manifest
