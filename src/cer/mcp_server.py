"""CER MCP Server — exposes CER commands as tools for AI agents."""

from __future__ import annotations

import json
import re

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


def _load() -> tuple[CERConfig, Database]:
    cfg = load_config()
    db = Database(cfg.local.db_path)
    return cfg, db


def _validate_commit(commit: str) -> str | None:
    if not re.match(r"^[0-9a-f]{7,40}$", commit):
        return f"Error: Invalid commit hash: {commit}"
    return None


def _build_run_command(cfg: CERConfig, worktree_dir: str, bind_flags: str) -> str:
    """Build the command that runs the experiment. Uses Singularity if image is set."""
    if cfg.container.image:
        return (
            f"singularity exec \\\n"
            f"    --nv \\\n"
            f"    --bind {worktree_dir} \\\n"
            f"    {bind_flags} \\\n"
            f"    --pwd {worktree_dir} \\\n"
            f"    {cfg.container.image} \\\n"
            f"    bash -c 'pip install -q -r requirements.txt 2>/dev/null; {cfg.experiment.entrypoint}'"
        )
    return cfg.experiment.entrypoint


def _build_submit_script(cfg: CERConfig, commit: str) -> str:
    commit_short = commit[:8]
    base_dir = cfg.cluster.base_dir
    log_dir = f"{base_dir}/logs"
    worktree_dir = f"{base_dir}/worktrees/{commit}"

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

export WANDB_API_KEY="{cfg.experiment.wandb_api_key}"
export WANDB_PROJECT="{cfg.experiment.wandb_project}"
export WANDB_ENTITY="{cfg.experiment.wandb_entity}"
export WANDB_RUN_NAME="cer-{commit_short}"
export WANDB_TAGS="{commit}"
export CER_COMMIT="{commit}"

cd "{worktree_dir}"

{_build_run_command(cfg, worktree_dir, bind_flags)}
"""

    return f"""set -euo pipefail

BASE_DIR="{base_dir}"
REPO_URL="{cfg.cluster.repo_url}"
COMMIT="{commit}"
BRANCH="{cfg.cluster.repo_branch}"

REPO_DIR="$BASE_DIR/repo"
WORKTREE_DIR="$BASE_DIR/worktrees/$COMMIT"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$BASE_DIR" "$LOG_DIR"

# Clone bare repo if not present
if [ ! -e "$REPO_DIR/HEAD" ]; then
    git clone --bare "$REPO_URL" "$REPO_DIR"
fi

# Fetch with retry until commit is available
MAX_RETRIES=30
RETRY_DELAY=10
for i in $(seq 1 $MAX_RETRIES); do
    git -C "$REPO_DIR" fetch origin "$BRANCH" --quiet 2>/dev/null || true
    if git -C "$REPO_DIR" cat-file -t "$COMMIT" >/dev/null 2>&1; then
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "ERROR: Commit $COMMIT not found after $MAX_RETRIES retries" >&2
        exit 1
    fi
    sleep $RETRY_DELAY
done

# Create worktree (remove stale one if exists)
if [ -d "$WORKTREE_DIR" ]; then
    git -C "$REPO_DIR" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
fi
git -C "$REPO_DIR" worktree add --detach "$WORKTREE_DIR" "$COMMIT"

# Auto-rebuild container if experiment.def changed
CONTAINER_IMAGE="{cfg.container.image}"
if [ -n "$CONTAINER_IMAGE" ] && [ -f "$WORKTREE_DIR/experiment.def" ]; then
    CONTAINER_DIR=$(dirname "$CONTAINER_IMAGE")
    CHECKSUM_FILE="$CONTAINER_DIR/experiment.def.md5"
    NEW_CHECKSUM=$(md5sum "$WORKTREE_DIR/experiment.def" | cut -d' ' -f1)
    OLD_CHECKSUM=""
    if [ -f "$CHECKSUM_FILE" ]; then
        OLD_CHECKSUM=$(cat "$CHECKSUM_FILE")
    fi
    if [ "$NEW_CHECKSUM" != "$OLD_CHECKSUM" ]; then
        echo "experiment.def changed — rebuilding container..."
        module load LUMI/23.09 partition/G systools/23.09 2>/dev/null || true
        singularity build --force "$CONTAINER_IMAGE" "$WORKTREE_DIR/experiment.def"
        echo "$NEW_CHECKSUM" > "$CHECKSUM_FILE"
        echo "Container rebuilt."
    fi
fi

# Write sbatch script
cat > "$WORKTREE_DIR/_run.sbatch" << 'SBATCH_EOF'
{sbatch_content}SBATCH_EOF

sbatch "$WORKTREE_DIR/_run.sbatch"
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
                    f"grep -o 'https://wandb.ai/[^ ]*' {log_path} 2>/dev/null | tail -1",
                )
                if wandb_result.ok and wandb_result.stdout:
                    url = wandb_result.stdout.strip()
                    run_id = url.rstrip("/").split("/")[-1]
                    db.update_wandb(job_id, run_id, url)
                    exp["wandb_url"] = url
            except SSHError:
                pass

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
