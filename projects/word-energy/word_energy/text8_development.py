"""Bounded text8 development samples without grammar or semantic labels."""
from __future__ import annotations
from collections import Counter
import hashlib
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TEXT8_ZIP=ROOT.parents[1]/'word2energy'/'landscape_ebm'/'text8_data'/'text8.zip'
EXPECTED_MD5='3bea1919949baf155f99411df5fada7e'
TRAIN=(10_000_000,11_000_000)
VALIDATION=(16_000_000,16_500_000)
RESERVED=(18_000_000,20_000_000)
VOCAB_SIZE=1000
MIN_COUNT=5
TRAIN_PAIRS=50_000
VALIDATION_PAIRS=5_000
TRAIN_NEGATIVES=5
VALIDATION_NEGATIVES=20
EPOCHS=3


def sha_array(arr):
    arr=np.ascontiguousarray(arr)
    return hashlib.sha256(str(arr.dtype).encode()+repr(arr.shape).encode()+arr.tobytes()).hexdigest()


def verify_archive(path=TEXT8_ZIP):
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist()
        if len(infos)!=1 or infos[0].filename!='text8' or infos[0].file_size!=100_000_000:
            raise ValueError('unexpected text8 archive contents')
        md5=hashlib.md5(); sha=hashlib.sha256()
        with archive.open('text8') as source:
            for chunk in iter(lambda:source.read(1024*1024),b''):
                md5.update(chunk); sha.update(chunk)
        if md5.hexdigest()!=EXPECTED_MD5: raise ValueError('text8 content checksum mismatch')
        return {'content_md5':md5.hexdigest(),'content_sha256':sha.hexdigest(),
                'content_bytes':infos[0].file_size,'archive':str(Path(path).resolve())}


def region_tokens(path,start,end):
    """Trim partial boundary words; never bridge across OOV or split boundary."""
    if not (0<start<end<100_000_000): raise ValueError('invalid region')
    with zipfile.ZipFile(path) as archive, archive.open('text8') as source:
        source.seek(start-1); block=source.read(end-start+2)
    if len(block)!=end-start+2: raise ValueError('truncated region')
    middle=block[1:-1]
    if block[0]!=32 and middle[0]!=32:
        position=middle.find(b' ')
        if position<0: raise ValueError('no word boundary')
        middle=middle[position+1:]
    if block[-1]!=32 and middle[-1]!=32:
        position=middle.rfind(b' ')
        if position<0: raise ValueError('no word boundary')
        middle=middle[:position]
    if any(byte!=32 and not 97<=byte<=122 for byte in middle): raise ValueError('unexpected character')
    return middle.decode('ascii').split()


def make_pairs(ids,n_pairs,rng,window=2):
    ids=np.asarray(ids,dtype=np.int32); positions=np.arange(ids.size)
    offsets=np.asarray([-2,-1,1,2],dtype=np.int32)
    if window!=2: raise ValueError('this protocol freezes window=2')
    choices=positions[:,None]+offsets[None,:]
    inside=(choices>=0)&(choices<len(ids))
    valid=inside&(ids[:,None]>=0)&(ids[np.clip(choices,0,len(ids)-1)]>=0)
    eligible=np.flatnonzero(valid.any(axis=1))
    if not len(eligible): raise ValueError('no in-vocabulary context pairs')
    selected=rng.choice(eligible,size=n_pairs,replace=True)
    mask=valid[selected]; k=mask.sum(axis=1)
    target=(rng.random(n_pairs)*k).astype(int)
    chosen=np.argmax(np.cumsum(mask,axis=1)>target[:,None],axis=1)
    contexts=selected+offsets[chosen]
    return np.column_stack((ids[selected],ids[contexts])).astype(np.int32),np.column_stack((selected,contexts)).astype(np.int32),{
        'tokens':int(len(ids)),'oov_tokens':int((ids<0).sum()),'token_coverage':float(np.mean(ids>=0)),
        'eligible_center_positions':int(len(eligible)),'sampled_pairs':int(n_pairs)}


def noise_rows(q,positives,count,rng):
    """Draw unconditional noise; positives and semantically valid collisions may occur."""
    positives=np.asarray(positives,dtype=np.int32)
    return rng.choice(len(q),size=(len(positives),count),p=q).astype(np.int32)


def build(path=TEXT8_ZIP):
    source=verify_archive(path)
    train=region_tokens(path,*TRAIN); validation=region_tokens(path,*VALIDATION)
    counts=Counter(train)
    vocabulary=sorted((w for w,c in counts.items() if c>=MIN_COUNT),key=lambda w:(-counts[w],w))[:VOCAB_SIZE]
    if len(vocabulary)!=VOCAB_SIZE: raise ValueError('too few vocabulary words')
    ids={w:i for i,w in enumerate(vocabulary)}
    count=np.asarray([counts[w] for w in vocabulary],dtype=np.int64)
    q=count.astype(np.float64)**.75; q/=q.sum()
    encode=lambda words: np.fromiter((ids.get(w,-1) for w in words),dtype=np.int32,count=len(words))
    train_ids=encode(train); val_ids=encode(validation)
    train_pairs,train_positions,train_stats=make_pairs(train_ids,TRAIN_PAIRS,np.random.default_rng(11301))
    val_pairs,val_positions,val_stats=make_pairs(val_ids,VALIDATION_PAIRS,np.random.default_rng(11302))
    val_noise=noise_rows(q,val_pairs[:,1],VALIDATION_NEGATIVES,np.random.default_rng(11303))
    noise=np.stack([noise_rows(q,train_pairs[:,1],TRAIN_NEGATIVES,np.random.default_rng(11310+e)) for e in range(EPOCHS)])
    orders=np.stack([np.random.default_rng(11320+e).permutation(TRAIN_PAIRS).astype(np.int32) for e in range(EPOCHS)])
    shuffle=np.random.default_rng(11330).permutation(TRAIN_PAIRS).astype(np.int32)
    shuffled_pairs=train_pairs.copy(); shuffled_pairs[:,1]=train_pairs[shuffle,1]
    arrays={'train_pairs':train_pairs,'shuffled_train_pairs':shuffled_pairs,'shuffle_order':shuffle,
            'validation_pairs':val_pairs,'train_positions':train_positions,'validation_positions':val_positions,
            'train_noise':noise,'validation_noise':val_noise,'orders':orders,'counts':count,'noise_q':q}
    manifest={'source':source,'train_region_bytes':list(TRAIN),'validation_region_bytes':list(VALIDATION),
              'reserved_region_bytes':list(RESERVED),'reserved_region_evaluated':False,'vocabulary':vocabulary,
              'sampling':'train-only count^.75 unconditional noise with collisions; uniform eligible raw positions with one context sampled from offsets -2,-1,+1,+2; no OOV bridging',
              'no_subsampling':True,'min_count':MIN_COUNT,'vocab_size':len(vocabulary),
              'train_stats':train_stats,'validation_stats':val_stats,
              'train_unique_pairs':int(len(np.unique(train_pairs,axis=0))),
              'validation_unique_pairs':int(len(np.unique(val_pairs,axis=0))),
              'train_unique_occurrence_pairs':int(len(np.unique(train_positions,axis=0))),
              'validation_unique_occurrence_pairs':int(len(np.unique(val_positions,axis=0))),
              'validation_exact_noise_collisions':int((val_noise==val_pairs[:,1,None]).sum()),
              'hashes':{name:sha_array(value) for name,value in arrays.items()},
              'limitations':'text8 has no document boundaries; inspected historical byte ranges separate but cannot prove unseen; development only, no sense labels'}
    return arrays,manifest
