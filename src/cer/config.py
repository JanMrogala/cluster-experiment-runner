from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(Exception):
    pass


@dataclass
class ClusterConfig:
    host: str
    base_dir: str
    repo_url: str
    repo_branch: str = "main"


@dataclass
class ContainerConfig:
    image: str
    bind_mounts: list[str] = field(default_factory=list)


@dataclass
class SlurmConfig:
    partition: str = "gpu"
    gres: str = "gpu:1"
    cpus_per_task: int = 8
    mem: str = "32G"
    time: str = "24:00:00"
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    entrypoint: str = "python train.py"
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_api_key: str = ""


@dataclass
class LocalConfig:
    db_path: str = "~/.local/share/cer/experiments.db"
    max_workspaces: int = 10


@dataclass
class CERConfig:
    cluster: ClusterConfig
    container: ContainerConfig
    slurm: SlurmConfig
    experiment: ExperimentConfig
    local: LocalConfig


def _find_config_file() -> Path:
    candidates = [
        Path("cer.yaml"),
        Path.home() / ".config" / "cer" / "cer.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise ConfigError(
        "No cer.yaml found. Looked in:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nCopy cer.yaml.example to cer.yaml and fill in your values."
    )


def _env_override(data: dict, prefix: str = "CER") -> dict:
    """Override nested dict values with CER_ environment variables.

    e.g. CER_CLUSTER_HOST overrides data["cluster"]["host"]
    """
    for key, value in os.environ.items():
        if not key.startswith(prefix + "_"):
            continue
        parts = key[len(prefix) + 1 :].lower().split("_", 1)
        if len(parts) == 2 and parts[0] in data and isinstance(data[parts[0]], dict):
            data[parts[0]][parts[1]] = value
    return data


def load_config() -> CERConfig:
    path = _find_config_file()
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"Invalid config file: {path}")

    raw = _env_override(raw)

    try:
        cluster = ClusterConfig(**raw.get("cluster", {}))
    except TypeError as e:
        raise ConfigError(f"Invalid cluster config: {e}") from e

    container = ContainerConfig(**raw.get("container", {}))
    slurm = SlurmConfig(**raw.get("slurm", {}))
    experiment = ExperimentConfig(**raw.get("experiment", {}))
    local = LocalConfig(**raw.get("local", {}))

    # Validate required fields
    if not cluster.host:
        raise ConfigError("cluster.host is required")
    if not cluster.base_dir:
        raise ConfigError("cluster.base_dir is required")
    if not cluster.repo_url:
        raise ConfigError("cluster.repo_url is required")
    # container.image is optional — if empty, experiments run without Singularity

    # Expand ~ in paths
    local.db_path = str(Path(local.db_path).expanduser())

    return CERConfig(
        cluster=cluster,
        container=container,
        slurm=slurm,
        experiment=experiment,
        local=local,
    )
