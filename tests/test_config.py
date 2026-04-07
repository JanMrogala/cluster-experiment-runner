import os

import pytest

from cer.config import ConfigError, load_config


def test_load_config(config_file):
    cfg = load_config()
    assert cfg.cluster.host == "testcluster"
    assert cfg.cluster.base_dir == "/tmp/cer_test"
    assert cfg.slurm.partition == "gpu"
    assert cfg.container.image == "/tmp/test.sif"


def test_load_config_missing_file(tmp_path, monkeypatch):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    # Point HOME to tmp_path so ~/.config/cer/cer.yaml isn't found
    monkeypatch.setenv("HOME", str(tmp_path))
    try:
        with pytest.raises(ConfigError, match="No cer.yaml found"):
            load_config()
    finally:
        os.chdir(old_cwd)


def test_env_override(config_file, monkeypatch):
    monkeypatch.setenv("CER_CLUSTER_HOST", "override-host")
    cfg = load_config()
    assert cfg.cluster.host == "override-host"
