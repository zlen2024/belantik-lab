"""Independent 4096-state fixtures for covariance sign; compare JAX scorer."""
from itertools import product
from math import exp, fsum, log
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from word_energy.wikitext_four_spin import score_windows

states=list(product((-1,1),repeat=4))
def fixture(opposite):
    signs=(1,-1 if opposite else 1)
    def q1(s,sign): return exp(sign*.6*s[0])/(16*np.cosh(.6))
    q=[1/16]*16
    qa=[q1(s,signs[0]) for s in states];qb=[q1(s,signs[1]) for s in states]
    def factor(s,t): return exp(.4*sum(x*y for x,y in zip(s,t)))
    A=[fsum(qa[j]*factor(s,t) for j,t in enumerate(states)) for s in states]
    B=[fsum(qb[j]*factor(s,t) for j,t in enumerate(states)) for s in states]
    joint=fsum(q[i]*A[i]*B[i] for i in range(16));p1=fsum(q[i]*A[i] for i in range(16));p2=fsum(q[i]*B[i] for i in range(16))
    direct=fsum(q[i]*qa[j]*qb[k]*factor(s,t)*factor(s,u) for i,s in enumerate(states) for j,t in enumerate(states) for k,u in enumerate(states))
    biases=np.zeros((3,4));biases[1,0]=np.arctanh(.6/2.5);biases[2,0]=signs[1]*biases[1,0]
    p={'biases':biases,'internal':np.zeros((3,6)),'interaction':np.full(4,log(.8/.7))}
    windows=np.array([[0,1,2]],np.int32);offset=np.zeros(3)
    actual_joint=float(score_windows(p,windows,offset,'joint4')[0]);actual_pair=float(score_windows(p,windows,offset,'pair_add4')[0])
    gap=log(joint)-log(p1)-log(p2);cov=joint-p1*p2
    assert abs(joint-direct)<1e-12 and abs(actual_joint-log(joint))<1e-12 and abs(actual_pair-log(p1)-log(p2))<1e-12
    assert (cov<0 and gap<0) if opposite else (cov>0 and gap>0)
    print(f"{'opposite' if opposite else 'same'}: covariance={cov:.10f}, log_gap={gap:.10f}, direct_minus_factorized={direct-joint:.3g}")
fixture(False);fixture(True)
