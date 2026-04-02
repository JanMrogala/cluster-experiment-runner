from __future__ import annotations

import subprocess
from dataclasses import dataclass


class SSHError(Exception):
    pass


@dataclass
class SSHResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _ssh_base(host: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]


def ssh_run(host: str, command: str, timeout: int = 120) -> SSHResult:
    """Execute a command on the remote host via SSH."""
    try:
        result = subprocess.run(
            [*_ssh_base(host), command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise SSHError(f"SSH command timed out after {timeout}s: {command}") from e
    except FileNotFoundError as e:
        raise SSHError("ssh binary not found. Is OpenSSH installed?") from e

    return SSHResult(result.returncode, result.stdout.strip(), result.stderr.strip())


def ssh_run_script(host: str, script: str, timeout: int = 300) -> SSHResult:
    """Pipe a script to bash -s on the remote host."""
    try:
        result = subprocess.run(
            [*_ssh_base(host), "bash", "-s"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise SSHError(f"SSH script timed out after {timeout}s") from e
    except FileNotFoundError as e:
        raise SSHError("ssh binary not found. Is OpenSSH installed?") from e

    return SSHResult(result.returncode, result.stdout.strip(), result.stderr.strip())
