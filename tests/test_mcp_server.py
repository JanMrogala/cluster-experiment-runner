"""Test MCP server tools by calling them directly (no network, mocked SSH)."""

import json
import os
from unittest.mock import patch

import pytest
import yaml

from cer.ssh import SSHResult


@pytest.fixture(autouse=True)
def mock_env(tmp_path):
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
            "nodes": 1,
            "ntasks": 1,
            "gpus": 1,
            "cpus_per_task": 4,
            "mem": "16G",
            "time": "01:00:00",
            "extra_flags": [],
        },
        "experiment": {
            "entrypoint": "python train.py",
            "wandb_project": "test-project",
            "wandb_entity": "",
            "wandb_api_key": "fake-key",
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


class TestMCPTools:
    """Call MCP tool functions directly to test the logic."""

    @patch("cer.mcp_server.ssh_run_script")
    def test_submit(self, mock_ssh):
        from cer.mcp_server import submit

        mock_ssh.return_value = SSHResult(0, "Submitted batch job 12345", "")
        result = submit(COMMIT)
        assert "12345" in result
        assert "Submitted" in result

    @patch("cer.mcp_server.ssh_run_script")
    def test_submit_invalid_hash(self, mock_ssh):
        from cer.mcp_server import submit

        result = submit("not-a-hash")
        assert "Invalid commit hash" in result

    @patch("cer.mcp_server.ssh_run")
    @patch("cer.mcp_server.ssh_run_script")
    def test_status(self, mock_script, mock_ssh):
        from cer.mcp_server import status, submit

        mock_script.return_value = SSHResult(0, "Submitted batch job 12345", "")
        submit(COMMIT)

        mock_ssh.return_value = SSHResult(0, "RUNNING", "")
        result = status("12345")
        data = json.loads(result)
        assert data["status"] == "RUNNING"
        assert data["job_id"] == "12345"

    def test_status_unknown_job(self):
        from cer.mcp_server import status

        result = status("99999")
        assert "not found" in result

    @patch("cer.mcp_server.ssh_run")
    @patch("cer.mcp_server.ssh_run_script")
    def test_cancel(self, mock_script, mock_ssh):
        from cer.mcp_server import cancel, submit

        mock_script.return_value = SSHResult(0, "Submitted batch job 12345", "")
        submit(COMMIT)

        mock_ssh.return_value = SSHResult(0, "", "")
        result = cancel("12345")
        assert "Cancelled" in result

    @patch("cer.mcp_server.ssh_run_script")
    def test_list_empty(self, mock_ssh):
        from cer.mcp_server import list_experiments

        result = list_experiments()
        assert "No experiments" in result

    @patch("cer.mcp_server.ssh_run")
    @patch("cer.mcp_server.ssh_run_script")
    def test_list_with_experiments(self, mock_script, mock_ssh):
        from cer.mcp_server import list_experiments, submit

        mock_script.return_value = SSHResult(0, "Submitted batch job 12345", "")
        submit(COMMIT)

        mock_ssh.return_value = SSHResult(0, "12345 RUNNING", "")
        result = list_experiments()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["job_id"] == "12345"

    @patch("cer.mcp_server.ssh_run_script")
    def test_submit_ssh_failure(self, mock_ssh):
        from cer.mcp_server import submit

        mock_ssh.return_value = SSHResult(1, "", "Connection refused")
        result = submit(COMMIT)
        assert "failed" in result.lower()

    @patch("cer.mcp_server.ssh_run")
    @patch("cer.mcp_server.ssh_run_script")
    def test_status_failed_fetches_error_log(self, mock_script, mock_ssh):
        """When a job is FAILED, status should fetch the SLURM err log tail."""
        from cer.mcp_server import status, submit

        mock_script.return_value = SSHResult(0, "Submitted batch job 12345", "")
        submit(COMMIT)

        # Three SSH calls happen for a FAILED job:
        #   1. squeue/sacct status check       -> "FAILED"
        #   2. tail of *.err log               -> traceback
        # (the wandb_url grep is skipped because status != RUNNING/COMPLETED)
        mock_ssh.side_effect = [
            SSHResult(0, "FAILED", ""),
            SSHResult(0, "Traceback (most recent call last):\nRuntimeError: boom", ""),
        ]
        result = status("12345")
        data = json.loads(result)
        assert data["status"] == "FAILED"
        assert "RuntimeError: boom" in data["error_message"]

    @patch("cer.mcp_server.ssh_run")
    @patch("cer.mcp_server.ssh_run_script")
    def test_status_strips_ansi_from_wandb_url(self, mock_script, mock_ssh):
        from cer.mcp_server import status, submit

        mock_script.return_value = SSHResult(0, "Submitted batch job 12345", "")
        submit(COMMIT)

        mock_ssh.side_effect = [
            SSHResult(0, "RUNNING", ""),
            SSHResult(0, "https://wandb.ai/team/proj/runs/abc123\x1b[0m", ""),
        ]
        result = status("12345")
        data = json.loads(result)
        assert data["wandb_url"] == "https://wandb.ai/team/proj/runs/abc123"
        assert "\x1b" not in data["wandb_url"]

    @patch("cer.mcp_server.ssh_run")
    @patch("cer.mcp_server.ssh_run_script")
    def test_logs_tool(self, mock_script, mock_ssh):
        from cer.mcp_server import logs, submit

        mock_script.return_value = SSHResult(0, "Submitted batch job 12345", "")
        submit(COMMIT)

        mock_ssh.side_effect = [
            SSHResult(0, "stdout line 1\nstdout line 2", ""),
            SSHResult(0, "stderr line 1", ""),
        ]
        result = logs("12345")
        data = json.loads(result)
        assert data["job_id"] == "12345"
        assert "stdout line 2" in data["stdout"]
        assert "stderr line 1" in data["stderr"]

    def test_logs_unknown_job(self):
        from cer.mcp_server import logs

        result = logs("99999")
        assert "not found" in result

    def test_logs_invalid_stream(self):
        from cer.mcp_server import logs

        result = logs("99999", stream="bogus")
        assert "Error" in result and "stream" in result


class TestMCPOverNetwork:
    """Test the actual MCP server over HTTP using the MCP client SDK."""

    @patch("cer.mcp_server.ssh_run_script")
    @patch("cer.mcp_server.ssh_run")
    def test_full_workflow_over_http(self, mock_ssh, mock_script):
        """Simulate what the agent does: connect to MCP server, list tools, call them."""
        import asyncio
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        mock_script.return_value = SSHResult(0, "Submitted batch job 77777", "")
        mock_ssh.return_value = SSHResult(0, "77777 RUNNING", "")

        async def run():
            # Start server in background
            import uvicorn
            from cer.mcp_server import mcp as mcp_app

            server_app = mcp_app.streamable_http_app()
            config = uvicorn.Config(server_app, host="127.0.0.1", port=18932, log_level="error")
            server = uvicorn.Server(config)
            server_task = asyncio.create_task(server.serve())

            # Wait for server to be ready
            import httpx
            for _ in range(50):
                try:
                    async with httpx.AsyncClient() as client:
                        await client.get("http://127.0.0.1:18932/mcp")
                        break
                except (httpx.ConnectError, ConnectionError):
                    await asyncio.sleep(0.1)

            try:
                async with streamablehttp_client("http://127.0.0.1:18932/mcp") as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()

                        # 1. List available tools
                        tools = await session.list_tools()
                        tool_names = [t.name for t in tools.tools]
                        assert "submit" in tool_names
                        assert "status" in tool_names
                        assert "cancel" in tool_names
                        assert "results" in tool_names
                        assert "list_experiments" in tool_names

                        # 2. Submit an experiment
                        result = await session.call_tool("submit", {"commit_hash": COMMIT})
                        text = result.content[0].text
                        assert "77777" in text
                        assert "Submitted" in text

                        # 3. List experiments
                        result = await session.call_tool("list_experiments", {})
                        data = json.loads(result.content[0].text)
                        assert len(data) == 1
                        assert data[0]["job_id"] == "77777"

                        # 4. Check status
                        mock_ssh.return_value = SSHResult(0, "RUNNING", "")
                        result = await session.call_tool("status", {"job_id": "77777"})
                        data = json.loads(result.content[0].text)
                        assert data["status"] == "RUNNING"

                        # 5. Cancel
                        mock_ssh.return_value = SSHResult(0, "", "")
                        result = await session.call_tool("cancel", {"job_id": "77777"})
                        assert "Cancelled" in result.content[0].text

            finally:
                server.should_exit = True
                await server_task

        asyncio.run(run())
