"""Test local MLflow tracking only; synthetic metrics are not model results."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / 'tracking'
ARTIFACTS = TRACKING / 'artifacts'
RUNS = ROOT / 'experiments' / 'runs'
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN_DIR = RUNS / f'mlflow-infrastructure-smoke-{STAMP}'
for folder in (TRACKING, ARTIFACTS, RUN_DIR):
    folder.mkdir(parents=True, exist_ok=True)
TEMP_ROOT = ROOT.parents[1] / '.wbt'
TEMP_ROOT.mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(TEMP_ROOT)
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'false'

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

uri = 'sqlite:///' + (TRACKING / 'mlflow.db').as_posix()
mlflow.set_tracking_uri(uri)
client = MlflowClient(tracking_uri=uri)
name = 'word-energy-infrastructure-smoke'
experiment = client.get_experiment_by_name(name)
if experiment is None:
    experiment_id = client.create_experiment(name, artifact_location=ARTIFACTS.as_uri())
else:
    experiment_id = experiment.experiment_id

config = {'purpose': 'infrastructure smoke test only', 'synthetic': True}
rows = [{'step': step, 'example_value': value} for step, value in enumerate([1.0, 0.5, 0.25])]
artifact = RUN_DIR / 'synthetic-config.json'
artifact.write_text(json.dumps(config, indent=2), encoding='utf-8')
(RUN_DIR / 'metrics.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows), encoding='utf-8')
with mlflow.start_run(experiment_id=experiment_id, run_name='synthetic-offline-smoke') as run:
    mlflow.set_tag('synthetic', 'true')
    mlflow.log_param('purpose', config['purpose'])
    for row in rows:
        mlflow.log_metric('example_value', row['example_value'], step=row['step'])
    mlflow.log_artifact(str(artifact))
    run_id = run.info.run_id

record = client.get_run(run_id)
history = client.get_metric_history(run_id, 'example_value')
assert record.info.status == 'FINISHED' and len(history) == 3
assert [x.value for x in history] == [1.0, 0.5, 0.25]
assert any(x.path == artifact.name for x in client.list_artifacts(run_id))
result = {'status': record.info.status, 'run_id': run_id, 'experiment_id': experiment_id,
          'mlflow_version': mlflow.__version__, 'tracking_uri': uri,
          'synthetic': True, 'metric_points': len(history), 'artifact_verified': True}
(RUN_DIR / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
