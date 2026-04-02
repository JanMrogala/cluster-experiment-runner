import os
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from cer.cli import app
from cer.ssh import SSHResult

runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_env(tmp_path):
    """Create a temp config and DB for all CLI tests."""
    config_data = {
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
            "db_path": str(tmp_path / "test.db"),
        },
    }
    config_path = tmp_path / "cer.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


COMMIT = "abcdef1234567890abcdef1234567890abcdef12"


class TestSubmit:
    @patch("cer.cli.ssh_run_script")
    def test_submit_success(self, mock_ssh):
        mock_ssh.return_value = SSHResult(0, "Submitted batch job 99999", "")
        result = runner.invoke(app, ["submit", COMMIT])
        assert result.exit_code == 0, result.output
        assert "99999" in result.output
        assert "Submitted!" in result.output

    @patch("cer.cli.ssh_run_script")
    def test_submit_with_config(self, mock_ssh):
        mock_ssh.return_value = SSHResult(0, "Submitted batch job 88888", "")
        result = runner.invoke(app, [
            "submit", COMMIT,
            "--config", "lr=0.001", "--config", "batch_size=64",
        ])
        assert result.exit_code == 0, result.output
        assert "88888" in result.output

    def test_submit_invalid_commit(self):
        result = runner.invoke(app, ["submit", "not-a-hash"])
        assert result.exit_code == 1

    @patch("cer.cli.ssh_run_script")
    def test_submit_ssh_failure(self, mock_ssh):
        mock_ssh.return_value = SSHResult(1, "", "Connection refused")
        result = runner.invoke(app, ["submit", COMMIT])
        assert result.exit_code == 1


class TestStatus:
    @patch("cer.cli.ssh_run")
    @patch("cer.cli.ssh_run_script")
    def test_status(self, mock_ssh_script, mock_ssh):
        # First submit
        mock_ssh_script.return_value = SSHResult(0, "Submitted batch job 55555", "")
        runner.invoke(app, ["submit", COMMIT])

        # Then check status
        mock_ssh.return_value = SSHResult(0, "RUNNING", "")
        result = runner.invoke(app, ["status", "55555"])
        assert result.exit_code == 0, result.output
        assert "RUNNING" in result.output

    def test_status_unknown_job(self):
        result = runner.invoke(app, ["status", "99999"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestCancel:
    @patch("cer.cli.ssh_run")
    @patch("cer.cli.ssh_run_script")
    def test_cancel(self, mock_ssh_script, mock_ssh):
        mock_ssh_script.return_value = SSHResult(0, "Submitted batch job 77777", "")
        runner.invoke(app, ["submit", COMMIT])

        mock_ssh.return_value = SSHResult(0, "", "")
        result = runner.invoke(app, ["cancel", "77777"])
        assert result.exit_code == 0, result.output
        assert "Cancelled" in result.output


class TestList:
    @patch("cer.cli.ssh_run")
    @patch("cer.cli.ssh_run_script")
    def test_list_with_experiments(self, mock_ssh_script, mock_ssh):
        mock_ssh_script.return_value = SSHResult(0, "Submitted batch job 11111", "")
        runner.invoke(app, ["submit", COMMIT])

        mock_ssh.return_value = SSHResult(0, "11111 RUNNING", "")
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0, result.output
        assert "11111" in result.output

    def test_list_empty(self):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No experiments found" in result.output
