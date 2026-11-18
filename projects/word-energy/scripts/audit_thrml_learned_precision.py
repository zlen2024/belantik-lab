"""Repeat the same learned pair parity with more draws after precision failure."""
from pathlib import Path
import json,sys,tempfile,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mlflow
import scripts.audit_thrml_learned_pairs as original
from word_energy.ising.exact_training import _imports
from word_energy.ising.fft_diagnostics import split_rank_rhat_ess_fft
from word_energy.tracking import RunContract,log_contract,tracking_uri

def sample_long(bias,weights,seed):
    jax,jnp,_=_imports()
    from thrml import Block,SamplingSchedule,SpinNode,sample_states
    from thrml.models import IsingEBM,IsingSamplingProgram
    nodes=[SpinNode() for _ in range(4)];edges=[(nodes[0],nodes[1]),(nodes[2],nodes[3]),(nodes[0],nodes[2]),(nodes[1],nodes[3])]
    model=IsingEBM(nodes,edges,jnp.asarray(bias),jnp.asarray(weights),jnp.array(1.));blocks=[Block([nodes[0],nodes[3]]),Block([nodes[1],nodes[2]])]
    program=IsingSamplingProgram(model,blocks,[]);schedule=SamplingSchedule(n_warmup=500,n_samples=8000,steps_per_sample=1)
    starts=jnp.array([[False,False,False,False],[True,True,True,True],[False,True,False,True],[True,False,True,False]],dtype=jnp.bool_)
    init=[starts[:,[0,3]],starts[:,[1,2]]];runner=jax.jit(jax.vmap(lambda ini,key:sample_states(key,program,schedule,ini,[],[Block(nodes)])))
    return 2*np.asarray(runner(init,jax.random.split(jax.random.key(seed),4))[0],dtype=np.int8)-1

if __name__=='__main__':
    start=time.monotonic();mlflow.set_tracking_uri(tracking_uri());original.sample=sample_long;original.split_rank_rhat_ess=split_rank_rhat_ess_fft
    result,samples=original.audit();result['elapsed_seconds']=time.monotonic()-start;result['previous_failed_run']='219a97eea88f4a3487b0aaa87f29fb79';result['samples_per_chain']=8000;result['adaptive_followup']=True
    mlflow.set_experiment('word-energy-thrml-learned-pair-parity')
    contract=RunContract(execution_tier='thrml_simulator',device_class='cpu_gpu_simulator',model_family='wikitext_learned_pair_precision_followup',energy_convention='E=-sum b*s-sum J*s*s, beta=1',variable_encoding='THRML bool True=+1',beta_temperature_convention='beta=1',sampler_schedule='block Gibbs two colors, warmup500, draws8000, step1',seed=17,data_hash=original.DATA,data_split='trained checkpoint; two illustrative vocab pairs; no test',graph_summary='4 spins, 4 edges, max degree2',limitations=result['limitations']+'; adaptive precision follow-up after original gate failed',chain_count=4,burn_in=500,sample_count=8000)
    with mlflow.start_run(run_name='learned-bank-pair-precision-followup') as run:
        log_contract(mlflow,contract,result);tmp=ROOT/'.tmp';tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as d:
            p=Path(d)/'spin_samples.npz';np.savez_compressed(p,**samples);mlflow.log_artifact(str(p),'samples')
        for p in (Path(__file__),ROOT/'scripts/audit_thrml_learned_pairs.py',ROOT/'word_energy/ising/fft_diagnostics.py',ROOT/'tests/test_fft_diagnostics.py',ROOT/'notes/THRML_LEARNED_PAIR_PRECISION_FOLLOWUP.md'):mlflow.log_artifact(str(p),'source_snapshot')
        print(json.dumps({'run_id':run.info.run_id,'results':result['results']},indent=2))
