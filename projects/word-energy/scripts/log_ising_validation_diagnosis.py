"""Record a read-only validation diagnosis in the local MLflow tracker."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mlflow
from scripts.diagnose_ising_validation import diagnose
from word_energy.tracking import RunContract, log_contract, sha256_file, tracking_uri

if __name__ == '__main__':
    seed = int(sys.argv[1])
    result = diagnose(seed)
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment('word-energy-exact-training')
    contract = RunContract(execution_tier='classical', device_class='cpu_gpu_simulator', model_family='two_spin_ising_exact_selected_checkpoint', energy_convention='dimensionless beta E, normalized unary and joint Ising', variable_encoding='spins {-1,+1}', beta_temperature_convention='beta=1', sampler_schedule='exact enumeration; no sampling or training', seed=seed, data_hash=sha256_file(ROOT / 'experiments' / 'runs' / 'synthetic-split-manifest.json'), data_split='training anchors for region map; validation windows for diagnosis', graph_summary='3 active tokens, 2 spins per token, star graph', limitations='Exploratory diagnosis on model-selection validation data; synthetic toy; no causal proof or hardware claim.')
    with mlflow.start_run(run_name=f'ising-validation-collapse-diagnosis-seed{seed}') as run:
        log_contract(mlflow, contract, result)
        metrics = {'anchor_total_variation': result['anchor_total_variation'], 'validation.macro_sense_accuracy': result['validation_macro_sense_accuracy'], 'validation.joint_ranking_loss': result['validation_joint_ranking_loss']}
        for family, values in result['family_summary'].items():
            for key, value in values.items():
                metrics[f'validation.{family}.{key}'] = value
        mlflow.log_metrics(metrics)
        print(run.info.run_id)
