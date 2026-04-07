from __future__ import annotations

import re


class SubmitError(Exception):
    pass


SLURM_STATUS_MAP = {
    "PENDING": "PENDING",
    "RUNNING": "RUNNING",
    "COMPLETED": "COMPLETED",
    "FAILED": "FAILED",
    "CANCELLED": "CANCELLED",
    "CANCELLED+": "CANCELLED",
    "TIMEOUT": "TIMEOUT",
    "NODE_FAIL": "FAILED",
    "PREEMPTED": "FAILED",
    "OUT_OF_MEMORY": "FAILED",
}


def parse_sbatch_output(stdout: str) -> str:
    """Extract job ID from 'Submitted batch job 12345'."""
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if not match:
        raise SubmitError(f"Could not parse sbatch output: {stdout}")
    return match.group(1)


def parse_job_status(stdout: str) -> str | None:
    """Parse status from squeue or sacct output (single job)."""
    line = stdout.strip()
    if not line:
        return None
    # Take the first non-empty word (handles extra whitespace from sacct)
    state = line.split()[0].strip()
    return SLURM_STATUS_MAP.get(state, "UNKNOWN")


def build_status_command(job_id: str) -> str:
    """SSH command to check job status. Tries squeue first, falls back to sacct."""
    # squeue succeeds with empty output for completed jobs, so use bash to check
    return (
        f'STATUS=$(squeue -j {job_id} -h -o "%T" 2>/dev/null); '
        f'if [ -n "$STATUS" ]; then echo "$STATUS"; '
        f"else sacct -j {job_id} -n -o State -X 2>/dev/null; fi"
    )


def build_batch_status_command(job_ids: list[str]) -> str:
    """SSH command to check multiple jobs at once."""
    ids = ",".join(job_ids)
    return (
        f"squeue -j {ids} -h -o '%i %T' 2>/dev/null; "
        f"sacct -j {ids} -n -o JobID,State -X 2>/dev/null"
    )


def parse_batch_status(stdout: str) -> dict[str, str]:
    """Parse output from build_batch_status_command into {job_id: status}."""
    results: dict[str, str] = {}
    for line in stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            job_id = parts[0].strip()
            state = parts[1].strip()
            # sacct may include .batch or .extern suffixes
            job_id = job_id.split(".")[0]
            mapped = SLURM_STATUS_MAP.get(state, "UNKNOWN")
            # Prefer non-UNKNOWN status (squeue is more reliable for active jobs)
            if job_id not in results or results[job_id] == "UNKNOWN":
                results[job_id] = mapped
    return results
