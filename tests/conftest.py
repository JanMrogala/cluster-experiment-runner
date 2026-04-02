import os
import tempfile
from pathlib import Path

import pytest
import yaml

from cer.db import Database


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    with db:
        yield db


@pytest.fixture
def sample_config_dict():
    return {
        "cluster": {
            "host": "testcluster",
            "base_dir": "/tmp/cer_test",
            "repo_url": "git@github.com:user/repo.git",
            "repo_branch": "main",
        },
        "container": {
            "image": "/tmp/test.sif",
            "bind_mounts": ["/data:/data"],
        },
        "slurm": {
            "partition": "gpu",
            "gres": "gpu:1",
            "cpus_per_task": 4,
            "mem": "16G",
            "time": "01:00:00",
            "extra_flags": [],
        },
        "experiment": {
            "entrypoint": "python train.py",
            "wandb_project": "test-project",
            "wandb_entity": "",
        },
        "local": {
            "db_path": "~/.local/share/cer/test.db",
        },
    }


@pytest.fixture
def config_file(tmp_path, sample_config_dict):
    config_path = tmp_path / "cer.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield config_path
    os.chdir(old_cwd)
