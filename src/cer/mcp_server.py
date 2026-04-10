"""CER MCP Server — exposes CER commands as tools for AI agents."""

from __future__ import annotations

import json
import re
import shlex

from mcp.server.fastmcp import FastMCP

from cer.config import CERConfig, load_config
from cer.db import Database
from cer.slurm import (
    SubmitError,
    build_batch_status_command,
    build_status_command,
    parse_batch_status,
    parse_job_status,
    parse_sbatch_output,
)
from cer.ssh import SSHError, ssh_run, ssh_run_script

mcp = FastMCP("cer", instructions="Cluster Experiment Runner — submit, monitor, and query GPU experiments on a SLURM cluster.")

# Matches ANSI CSI escape sequences (e.g. color codes like \x1b[0m).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _log_paths(cfg: CERConfig, job_id: str) -> tuple[str, str]:
    base = f"{cfg.cluster.base_dir}/logs/{job_id}"
    return f"{base}.out", f"{base}.err"


def _fetch_log_tail(cfg: CERConfig, job_id: str, stream: str, tail: int) -> str | None:
    """Fetch the last `tail` lines of a SLURM log. Returns None if file missing or SSH fails."""
    out_path, err_path = _log_paths(cfg, job_id)
    path = err_path if stream == "err" else out_path
    try:
        result = ssh_run(
            cfg.cluster.host,
            f"tail -n {int(tail)} {shlex.quote(path)} 2>/dev/null",
        )
    except SSHError:
        return None
    if not result.ok or not result.stdout:
        return None
    return _strip_ansi(result.stdout).strip() or None


def _load() -> tuple[CERConfig, Database]:
    cfg = load_config()
    db = Database(cfg.local.db_path)
    return cfg, db


def _validate_commit(commit: str) -> str | None:
    if not re.match(r"^[0-9a-f]{7,40}$", commit):
        return f"Error: Invalid commit hash: {commit}"
    return None


def _build_submit_script(cfg: CERConfig, commit: str) -> str:
    commit_short = commit[:8]
    base_dir = cfg.cluster.base_dir
    log_dir = f"{base_dir}/logs"
    sbatch_dir = f"{base_dir}/sbatch"

    extra_sbatch = "\n".join(f"#SBATCH {f}" for f in cfg.slurm.extra_flags)
    bind_flags = " ".join(f"--bind {m}" for m in cfg.container.bind_mounts)
    account_line = f"\n#SBATCH --account={cfg.slurm.account}" if cfg.slurm.account else ""

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=cer-{commit_short}
#SBATCH --partition={cfg.slurm.partition}
#SBATCH --nodes={cfg.slurm.nodes}
#SBATCH --ntasks={cfg.slurm.ntasks}
#SBATCH --gpus={cfg.slurm.gpus}
#SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
#SBATCH --mem={cfg.slurm.mem}
#SBATCH --time={cfg.slurm.time}
#SBATCH --output={log_dir}/%j.out
#SBATCH --error={log_dir}/%j.err{account_line}
{extra_sbatch}

set -euo pipefail

export WANDB_API_KEY="{cfg.experiment.wandb_api_key}"
export WANDB_PROJECT="{cfg.experiment.wandb_project}"
export WANDB_ENTITY="{cfg.experiment.wandb_entity}"
export WANDB_RUN_NAME="cer-{commit_short}"
export WANDB_TAGS="{commit}"
export CER_COMMIT="{commit}"

# Clone repo and checkout exact commit
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT
git clone {cfg.cluster.repo_url} "$WORK"
cd "$WORK"
git fetch origin '+refs/heads/*:refs/remotes/origin/*'
git checkout {commit}

# Run experiment inside Singularity container
singularity exec \\
    --nv \\
    --writable-tmpfs \\
    --bind "$WORK":"$WORK" \\
    {bind_flags} \\
    --pwd "$WORK" \\
    {cfg.container.image} \\
    bash -c 'pip install -q -r requirements.txt 2>/dev/null || true; {cfg.experiment.entrypoint}'
"""

    return f"""set -euo pipefail

BASE_DIR="{base_dir}"
LOG_DIR="$BASE_DIR/logs"
SBATCH_DIR="$BASE_DIR/sbatch"

mkdir -p "$LOG_DIR" "$SBATCH_DIR"

# Write and submit sbatch script
cat > "$SBATCH_DIR/cer-{commit_short}.sbatch" << 'SBATCH_EOF'
{sbatch_content}SBATCH_EOF

sbatch "$SBATCH_DIR/cer-{commit_short}.sbatch"
"""


@mcp.tool()
def submit(commit_hash: str) -> str:
    """Submit an experiment to the SLURM cluster for a given git commit hash.

    The commit must be pushed to the remote repository first.
    Returns the SLURM job ID on success.
    """
    err = _validate_commit(commit_hash)
    if err:
        return err

    try:
        cfg, db = _load()
    except Exception as e:
        return f"Error: {e}"

    script = _build_submit_script(cfg, commit_hash)

    try:
        result = ssh_run_script(cfg.cluster.host, script, timeout=900)
    except SSHError as e:
        return f"SSH error: {e}"

    if not result.ok:
        return f"Submission failed: {result.stderr}"

    try:
        job_id = parse_sbatch_output(result.stdout)
    except SubmitError as e:
        return f"Error parsing sbatch output: {e}"

    with db:
        db.insert_experiment(
            job_id=job_id,
            commit_hash=commit_hash,
            config={},
            cluster_host=cfg.cluster.host,
            partition=cfg.slurm.partition,
        )

    return f"Submitted! Job ID: {job_id}"


@mcp.tool()
def status(job_id: str) -> str:
    """Check the SLURM status of an experiment by job ID.

    Returns job status, commit info, and W&B URL if available.
    If the job has FAILED, the tail of the SLURM error log is fetched
    and surfaced as `error_message`.
    """
    try:
        cfg, db = _load()
    except Exception as e:
        return f"Error: {e}"

    with db:
        exp = db.get_experiment(job_id)
        if not exp:
            return f"Job {job_id} not found in local database"

        try:
            result = ssh_run(cfg.cluster.host, build_status_command(job_id))
        except SSHError as e:
            return f"SSH error: {e}"

        slurm_status = parse_job_status(result.stdout)
        if slurm_status:
            db.update_status(job_id, slurm_status)
            exp["status"] = slurm_status

        if not exp.get("wandb_url") and exp["status"] in ("RUNNING", "COMPLETED"):
            try:
                log_path = f"{cfg.cluster.base_dir}/logs/{job_id}.out"
                wandb_result = ssh_run(
                    cfg.cluster.host,
                    f"grep -aoE 'https://wandb\\.ai/[^[:space:]]+' {log_path} 2>/dev/null | tail -1",
                )
                if wandb_result.ok and wandb_result.stdout:
                    url = _strip_ansi(wandb_result.stdout).strip()
                    # Trim any trailing punctuation that may have been captured.
                    url = url.rstrip(".,;:)\"'")
                    if url:
                        run_id = url.rstrip("/").split("/")[-1]
                        db.update_wandb(job_id, run_id, url)
                        exp["wandb_url"] = url
                        exp["wandb_run_id"] = run_id
            except SSHError:
                pass

        # If the job failed, fetch the tail of the error log and persist it.
        if exp["status"] in ("FAILED", "TIMEOUT") and not exp.get("error_message"):
            err_tail = _fetch_log_tail(cfg, job_id, "err", 50)
            if not err_tail:
                # Some failures only print to stdout (e.g. early Python errors).
                err_tail = _fetch_log_tail(cfg, job_id, "out", 50)
            if err_tail:
                db.update_status(job_id, exp["status"], error_message=err_tail)
                exp["error_message"] = err_tail

    return json.dumps(exp, indent=2, default=str)


@mcp.tool()
def cancel(job_id: str) -> str:
    """Cancel a running experiment on the SLURM cluster."""
    try:
        cfg, db = _load()
    except Exception as e:
        return f"Error: {e}"

    with db:
        exp = db.get_experiment(job_id)
        if not exp:
            return f"Job {job_id} not found in local database"

        try:
            result = ssh_run(cfg.cluster.host, f"scancel {job_id}")
        except SSHError as e:
            return f"SSH error: {e}"

        if not result.ok:
            return f"Cancel failed: {result.stderr}"

        db.update_status(job_id, "CANCELLED")

    return f"Cancelled job {job_id}"


@mcp.tool()
def logs(job_id: str, stream: str = "both", tail: int = 200) -> str:
    """Fetch SLURM stdout/stderr logs for a job.

    Args:
        job_id: SLURM job ID.
        stream: 'out', 'err', or 'both' (default 'both').
        tail: Number of trailing lines to return per stream (default 200).
    """
    if stream not in ("out", "err", "both"):
        return f"Error: stream must be 'out', 'err', or 'both' (got {stream!r})"

    try:
        cfg, db = _load()
    except Exception as e:
        return f"Error: {e}"

    with db:
        exp = db.get_experiment(job_id)
        if not exp:
            return f"Job {job_id} not found in local database"

    payload: dict[str, str | None] = {"job_id": job_id}
    if stream in ("out", "both"):
        payload["stdout"] = _fetch_log_tail(cfg, job_id, "out", tail)
    if stream in ("err", "both"):
        payload["stderr"] = _fetch_log_tail(cfg, job_id, "err", tail)

    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
def results(job_id: str, history: bool = False, keys: list[str] | None = None) -> str:
    """Query Weights & Biases results for an experiment.

    Args:
        job_id: SLURM job ID
        history: If True, include full metric history (all logged steps)
        keys: Optional list of specific metric keys to fetch (e.g. ["train/loss", "val/acc"])
    """
    try:
        cfg, db = _load()
    except Exception as e:
        return f"Error: {e}"

    if not cfg.experiment.wandb_project:
        return "Error: wandb_project not set in cer.yaml"

    with db:
        exp = db.get_experiment(job_id)
        if not exp:
            return f"Job {job_id} not found in local database"

    from cer.wandb_query import find_run, get_run_history, get_run_summary

    run = find_run(
        cfg.experiment.wandb_project,
        exp["commit_hash"],
        cfg.experiment.wandb_entity,
        cfg.experiment.wandb_api_key,
    )

    if not run:
        return "No W&B run found for this experiment. It may not have started logging yet."

    summary = get_run_summary(run)

    with db:
        db.update_wandb(job_id, summary["run_id"], summary["url"])

    output = {
        "job_id": job_id,
        "commit": exp["commit_hash"],
        "wandb": summary,
    }
    if history:
        output["history"] = get_run_history(run, keys=keys)

    return json.dumps(output, indent=2, default=str)


@mcp.tool()
def list_experiments() -> str:
    """List all tracked experiments with their current status."""
    try:
        cfg, db = _load()
    except Exception as e:
        return f"Error: {e}"

    with db:
        experiments = db.list_experiments()

        if not experiments:
            return "No experiments found."

        active = [e for e in experiments if e["status"] in ("SUBMITTED", "PENDING", "RUNNING")]
        if active:
            active_ids = [e["job_id"] for e in active]
            try:
                result = ssh_run(cfg.cluster.host, build_batch_status_command(active_ids))
                if result.ok and result.stdout:
                    statuses = parse_batch_status(result.stdout)
                    for exp in experiments:
                        if exp["job_id"] in statuses:
                            new_status = statuses[exp["job_id"]]
                            if new_status != exp["status"]:
                                db.update_status(exp["job_id"], new_status)
                                exp["status"] = new_status
            except SSHError:
                pass

    return json.dumps(experiments, indent=2, default=str)



def main():
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
