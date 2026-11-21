"""Summarize completed blinded AI reviews without treating them as gold labels."""
from collections import Counter,defaultdict
from pathlib import Path
import json,sys
import mlflow
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from word_energy.tracking import RunContract,log_contract,sha256_file,tracking_uri

def main():
    root=ROOT/'.tmp/twsi_blind_review'
    names=['cue.json','full.json','key.json','cue_review.json','full_review.json']
    data={name:json.loads((root/name).read_text(encoding='utf8')) for name in names}
    ids={r['id'] for r in data['key.json']}
    if len(ids)!=24 or any(len(a)!=24 or {r['id'] for r in a}!=ids for a in data.values()):raise ValueError('panel/review ID mismatch')
    key={r['id']:r for r in data['key.json']}
    cue={r['id']:r for r in data['cue_review.json']};full={r['id']:r for r in data['full_review.json']}
    groups=defaultdict(list)
    for i in ids:groups[key[i]['group']].append(i)
    if len(groups)!=12 or any(len(g)!=2 or {key[i]['weak_preference'] for i in g}!={'A','B'} for g in groups.values()):raise ValueError('invalid reversal groups')
    result={'limitations':'AI reviews are not human annotation; small selected train-only set, unverified sources, unknown cue sufficiency and non-independent items. Full-sentence reviewer agent exited with usage-limit error after writing its complete-looking file; its final completion status was not confirmed.',
            'panel_rows':24,'groups':12,'source_run':'16e0bf36c0574dcb92cd88cdf1a9b1d3','context_run':'8f2b8e9cfa8344d6bed4c2155e003cda',
            'hashes':{n:sha256_file(root/n) for n in names}}
    for name,review in [('cue',cue),('full',full)]:
        if any(x['choice'] not in ('A','B','U') or not isinstance(x.get('reason'),str) for x in review.values()):raise ValueError('invalid review')
        choices=Counter(x['choice'] for x in review.values());decisive=[i for i in ids if review[i]['choice']!='U']
        result[name]={'choices':dict(choices),'decisive':len(decisive),
          'matches_weak_among_all':sum(review[i]['choice']==key[i]['weak_preference'] for i in ids),
          'matches_weak_among_decisive':sum(review[i]['choice']==key[i]['weak_preference'] for i in decisive),
          'groups_both_match_weak':sum(all(review[i]['choice']==key[i]['weak_preference'] for i in g) for g in groups.values())}
    both=[i for i in ids if cue[i]['choice']!='U' and full[i]['choice']!='U']
    result['reviewer_agreement']={'joint_decisive':len(both),'same_choice_joint_decisive':sum(cue[i]['choice']==full[i]['choice'] for i in both),
      'same_choice_including_uncertain':sum(cue[i]['choice']==full[i]['choice'] for i in ids)}
    mlflow.set_tracking_uri(tracking_uri());mlflow.set_experiment('word-energy-natural-context-evaluation')
    with mlflow.start_run(run_name='twsi-blind-ai-cue-sufficiency-review') as run:
        log_contract(mlflow,RunContract('classical','cpu_gpu_simulator','blind AI review','no EBM','words and weak labels','not applicable','two AI reviewers; no sampling',88121,result['hashes']['key.json'],'TWSI train selected pairs only','no graph',result['limitations']),result)
        mlflow.log_dict(result,'summary.json')
        for name in names:mlflow.log_artifact(str(root/name),'blind_review')
        for path in (Path(__file__),ROOT/'scripts/build_twsi_blind_cue_panel.py',ROOT/'notes/TWSI_AI_CUE_REVIEW_PROTOCOL.md'):mlflow.log_artifact(str(path),'source_snapshot')
        print('RUN',run.info.run_id,json.dumps(result),flush=True)
if __name__=='__main__':main()
