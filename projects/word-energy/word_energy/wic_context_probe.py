"""Label-free contextual probes for frozen pair-trained embeddings."""
import re
import numpy as np
from scipy.special import expit,softmax
STATES=np.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]])
TOKEN=re.compile(r'[A-Za-z]+')

def extract(sentence,start,end,vocab,stats=False):
    tokens=list(TOKEN.finditer(sentence));hits=[i for i,x in enumerate(tokens) if x.start()<end and x.end()>start]
    if len(hits)!=1 or start<0 or end>len(sentence) or start>=end:return None
    center=hits[0]
    if tokens[center].start()!=start or tokens[center].end()!=end:return None
    ctx=[];total=0
    for delta in (-2,-1,1,2):
        i=center+delta
        if 0<=i<len(tokens):
            total+=1;w=tokens[i].group().lower()
            if w in vocab:ctx.append(vocab[w])
    return (ctx,total) if stats else ctx

def unary(b,k):return softmax(STATES@(2.5*np.tanh(b))+(STATES[:,0]*STATES[:,1])*(2*np.tanh(k)))

def star(raw,target,contexts,role):
    if not contexts:raise ValueError('empty context')
    if role:bt,kt,bc,kc=raw['target_biases'],raw['target_internal'],raw['context_biases'],raw['context_internal']
    else:bt=bc=raw['biases'];kt=kc=raw['internal']
    p=unary(bt[target],kt[target]);g=1.5*expit(raw['interaction'])/len(contexts)
    pair=np.exp(np.sum(STATES[:,None,:]*STATES[None,:,:]*g,axis=-1))
    for c in contexts:p*=pair@unary(bc[c],kc[c])
    return p/p.sum()

def negative_js(p,q):
    p=np.asarray(p);q=np.asarray(q);m=(p+q)/2
    return -.5*(np.sum(p*np.log(p/m))+np.sum(q*np.log(q/m)))

def cosine(x,y):
    a=np.linalg.norm(x);b=np.linalg.norm(y)
    if not np.isfinite(a*b) or a<1e-14 or b<1e-14:raise ValueError('zero/nonfinite representation')
    return float(x@y/(a*b))

def auc(labels,scores):
    labels=np.asarray(labels);scores=np.asarray(scores);p=scores[labels==1];n=scores[labels==0]
    if len(p)==0 or len(n)==0:return None
    return float(((p[:,None]>n)+.5*(p[:,None]==n)).mean())
