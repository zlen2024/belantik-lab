"""MLflow setup and reproducibility metadata for Word Energy experiments.

This module deliberately records whether an output came from ordinary code, a
THRML/Torx simulator, or a physical probabilistic chip.  It does not claim that
the local SQLite tracker or a simulator is hardware execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunContract:
    execution_tier: str
    device_class: str
    model_family: str
    energy_convention: str
    variable_encoding: str
    beta_temperature_convention: str
    sampler_schedule: str
    seed: int
    data_hash: str
    data_split: str
    graph_summary: str
    limitations: str
    chain_count: int | None = None
    burn_in: int | None = None
    sample_count: int | None = None

    def __post_init__(self) -> None:
        if self.execution_tier not in {"classical", "thrml_simulator", "torx_simulator", "hardware"}:
            raise ValueError(f"Unknown execution tier: {self.execution_tier}")
        if self.device_class not in {"cpu_gpu_simulator", "remote_simulator", "physical_probabilistic_chip"}:
            raise ValueError(f"Unknown device class: {self.device_class}")
        if self.execution_tier == "hardware" and self.device_class != "physical_probabilistic_chip":
            raise ValueError("hardware tier requires a physical probabilistic chip")
        if self.execution_tier != "hardware" and self.device_class == "physical_probabilistic_chip":
            raise ValueError("physical chip device class requires hardware tier")


def tracking_uri(project_root: Path = PROJECT_ROOT) -> str:
    """Return the explicit local SQLite URI mandated by TRACKING_DECISION.md."""
    database = (project_root / "tracking" / "mlflow.db").resolve().as_posix()
    return f"sqlite:///{database}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Return commit plus a hash of uncommitted tracked changes, without writing."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root.parents[1]), "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(project_root.parents[1]), "diff", "--binary", "HEAD"], text=False
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unavailable", "dirty_tree_sha256": "unavailable"}
    return {"git_commit": commit, "dirty_tree_sha256": hashlib.sha256(diff).hexdigest()}


def log_contract(mlflow: Any, contract: RunContract, config: dict[str, Any]) -> None:
    """Attach required metadata and reproducible JSON artifacts to an active run."""
    mlflow.set_tracking_uri(tracking_uri())
    tags = {key: str(value) for key, value in asdict(contract).items() if value is not None}
    tags.update(git_snapshot())
    mlflow.set_tags(tags)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlflow": _package_version("mlflow"),
        "numpy": _package_version("numpy"),
        "jax": _package_version("jax"),
        "thrml": _package_version("thrml"),
        "extro-torx": _package_version("extro-torx"),
    }
    # MLflow's log_dict uses the process temp directory. In this Windows sandbox
    # that directory is not writable, so materialise the same JSON under the
    # project-controlled `.tmp` and upload it as ordinary artifacts instead.
    artifact_root = PROJECT_ROOT / ".tmp"
    artifact_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=artifact_root) as temporary:
        temporary_path = Path(temporary)
        config_path = temporary_path / "config.json"
        contract_path = temporary_path / "run_contract.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        contract_path.write_text(
            json.dumps({"contract": asdict(contract), "environment": environment}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(str(contract_path))


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"

