from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import keyring
import yaml

KEYRING_SERVICE = "cer"


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
    partition: str = "small-g"
    nodes: int = 1
    ntasks: int = 1
    gpus: int = 1
    cpus_per_task: int = 8
    mem: str = "64GB"
    time: str = "24:00:00"
    account: str = ""
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


SECRET_FIELDS = {"experiment.wandb_api_key"}


def _resolve_secret(section: str, key: str, value: str) -> str:
    """Resolve a config value that may be stored in the system keyring.

    If the value is empty or the literal string "keyring", look it up
    from the system keyring (GNOME Keyring / KDE Wallet / macOS Keychain).
    """
    field_name = f"{section}.{key}"
    if field_name not in SECRET_FIELDS:
        return value
    if value and value != "keyring":
        return value
    secret = keyring.get_password(KEYRING_SERVICE, key)
    if secret is None:
        raise ConfigError(
            f"{field_name} not found in config or system keyring.\n"
            f"Store it with:  cer-secret set {key} <value>"
        )
    return secret


def _resolve_secrets(raw: dict) -> dict:
    """Walk config and resolve any secret fields via keyring."""
    for section_name, section in raw.items():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if isinstance(value, str):
                section[key] = _resolve_secret(section_name, key, value)
    return raw


def load_config() -> CERConfig:
    path = _find_config_file()
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"Invalid config file: {path}")

    raw = _env_override(raw)
    raw = _resolve_secrets(raw)

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
