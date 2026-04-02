from __future__ import annotations

import json
import re
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from cer.config import CERConfig, ConfigError, load_config
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

app = typer.Typer(name="cer", help="Cluster Experiment Runner", no_args_is_help=True)
console = Console()


def _load() -> tuple[CERConfig, Database]:
    cfg = load_config()
    db = Database(cfg.local.db_path)
    return cfg, db


def _validate_commit(commit: str) -> str:
    if not re.match(r"^[0-9a-f]{7,40}$", commit):
        console.print(f"[red]Invalid commit hash: {commit}[/red]")
        raise typer.Exit(1)
    return commit


def _build_submit_script(cfg: CERConfig, commit: str, config: dict) -> str:
    commit_short = commit[:8]
    base_dir = cfg.cluster.base_dir
    log_dir = f"{base_dir}/logs"
    worktree_dir = f"{base_dir}/worktrees/{commit}"

    extra_sbatch = "\n".join(f"#SBATCH {f}" for f in cfg.slurm.extra_flags)
    bind_flags = " ".join(f"--bind {m}" for m in cfg.container.bind_mounts)

    config_cli_args = " ".join(f"--{k} {v}" for k, v in config.items())
    config_json = json.dumps(config)

    # Build the sbatch content separately (no shell variable conflicts)
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=cer-{commit_short}
#SBATCH --partition={cfg.slurm.partition}
#SBATCH --gres={cfg.slurm.gres}
#SBATCH --cpus-per-task={cfg.slurm.cpus_per_task}
#SBATCH --mem={cfg.slurm.mem}
#SBATCH --time={cfg.slurm.time}
#SBATCH --output={log_dir}/%j.out
#SBATCH --error={log_dir}/%j.err
{extra_sbatch}

export WANDB_PROJECT="{cfg.experiment.wandb_project}"
export WANDB_ENTITY="{cfg.experiment.wandb_entity}"
export WANDB_RUN_NAME="cer-{commit_short}"
export WANDB_TAGS="{commit}"
export CER_COMMIT="{commit}"
export CER_CONFIG='{config_json}'

cd "{worktree_dir}"

singularity exec \\
    --nv \\
    {bind_flags} \\
    {cfg.container.image} \\
    bash -c 'pip install -q -r requirements.txt 2>/dev/null; {cfg.experiment.entrypoint} {config_cli_args}'
"""

    # Use $ for shell variables in the bash orchestration part
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
if [ ! -d "$REPO_DIR/HEAD" ]; then
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

# Write sbatch script
cat > "$WORKTREE_DIR/_run.sbatch" << 'SBATCH_EOF'
{sbatch_content}SBATCH_EOF

sbatch "$WORKTREE_DIR/_run.sbatch"
"""


@app.command()
def submit(
    commit_hash: Annotated[str, typer.Argument(help="Git commit hash to run")],
    config: Annotated[
        Optional[list[str]],
        typer.Option(
            "--config", "-c", help="Config overrides as key=value pairs"
        ),
    ] = None,
):
    """Submit an experiment for a given commit."""
    commit_hash = _validate_commit(commit_hash)
    config_dict: dict[str, str] = {}
    for item in config or []:
        if "=" not in item:
            console.print(f"[red]Invalid config format: {item} (expected key=value)[/red]")
            raise typer.Exit(1)
        k, v = item.split("=", 1)
        config_dict[k] = v

    try:
        cfg, db = _load()
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    script = _build_submit_script(cfg, commit_hash, config_dict)

    console.print(f"Submitting experiment for commit [cyan]{commit_hash[:8]}[/cyan]...")

    try:
        result = ssh_run_script(cfg.cluster.host, script)
    except SSHError as e:
        console.print(f"[red]SSH error: {e}[/red]")
        raise typer.Exit(1)

    if not result.ok:
        console.print(f"[red]Submission failed:[/red]\n{result.stderr}")
        raise typer.Exit(1)

    try:
        job_id = parse_sbatch_output(result.stdout)
    except SubmitError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    with db:
        db.insert_experiment(
            job_id=job_id,
            commit_hash=commit_hash,
            config=config_dict,
            cluster_host=cfg.cluster.host,
            partition=cfg.slurm.partition,
        )

    console.print(f"[green]Submitted![/green] Job ID: [bold]{job_id}[/bold]")


@app.command()
def status(
    job_id: Annotated[str, typer.Argument(help="SLURM job ID")],
):
    """Check the status of an experiment."""
    try:
        cfg, db = _load()
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    with db:
        exp = db.get_experiment(job_id)
        if not exp:
            console.print(f"[red]Job {job_id} not found in local database[/red]")
            raise typer.Exit(1)

        # Query SLURM
        try:
            result = ssh_run(cfg.cluster.host, build_status_command(job_id))
        except SSHError as e:
            console.print(f"[red]SSH error: {e}[/red]")
            raise typer.Exit(1)

        slurm_status = parse_job_status(result.stdout)
        if slurm_status:
            db.update_status(job_id, slurm_status)
            exp["status"] = slurm_status

        # Try to find W&B URL if not yet known
        if not exp.get("wandb_url") and exp["status"] in ("RUNNING", "COMPLETED"):
            try:
                log_path = f"{cfg.cluster.base_dir}/logs/{job_id}.out"
                wandb_result = ssh_run(
                    cfg.cluster.host,
                    f"grep -o 'https://wandb.ai/[^ ]*' {log_path} 2>/dev/null | tail -1",
                )
                if wandb_result.ok and wandb_result.stdout:
                    url = wandb_result.stdout.strip()
                    # Extract run ID from URL (last path segment)
                    run_id = url.rstrip("/").split("/")[-1]
                    db.update_wandb(job_id, run_id, url)
                    exp["wandb_url"] = url
            except SSHError:
                pass  # Non-critical

        # Display
        status_colors = {
            "RUNNING": "green",
            "COMPLETED": "blue",
            "FAILED": "red",
            "CANCELLED": "yellow",
            "PENDING": "cyan",
            "SUBMITTED": "cyan",
            "TIMEOUT": "red",
        }
        color = status_colors.get(exp["status"], "white")

        console.print(f"Job ID:    [bold]{exp['job_id']}[/bold]")
        console.print(f"Commit:    [cyan]{exp['commit_short']}[/cyan] ({exp['commit_hash']})")
        console.print(f"Status:    [{color}]{exp['status']}[/{color}]")
        console.print(f"Submitted: {exp['submitted_at']}")
        if exp.get("config_json") and exp["config_json"] != "{}":
            console.print(f"Config:    {exp['config_json']}")
        if exp.get("wandb_url"):
            console.print(f"W&B:       {exp['wandb_url']}")
        if exp.get("error_message"):
            console.print(f"Error:     [red]{exp['error_message']}[/red]")


@app.command()
def cancel(
    job_id: Annotated[str, typer.Argument(help="SLURM job ID to cancel")],
):
    """Cancel a running experiment."""
    try:
        cfg, db = _load()
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    with db:
        exp = db.get_experiment(job_id)
        if not exp:
            console.print(f"[red]Job {job_id} not found in local database[/red]")
            raise typer.Exit(1)

        try:
            result = ssh_run(cfg.cluster.host, f"scancel {job_id}")
        except SSHError as e:
            console.print(f"[red]SSH error: {e}[/red]")
            raise typer.Exit(1)

        if not result.ok:
            console.print(f"[red]Cancel failed: {result.stderr}[/red]")
            raise typer.Exit(1)

        db.update_status(job_id, "CANCELLED")
        console.print(f"[yellow]Cancelled job {job_id}[/yellow]")


@app.command(name="list")
def list_experiments():
    """List recent experiments."""
    try:
        cfg, db = _load()
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    with db:
        experiments = db.list_experiments()

        if not experiments:
            console.print("No experiments found.")
            return

        # Batch-refresh active jobs
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
                pass  # Show stale status rather than failing

        table = Table(title="Experiments")
        table.add_column("Job ID", style="bold")
        table.add_column("Commit", style="cyan")
        table.add_column("Status")
        table.add_column("Submitted")
        table.add_column("Config")
        table.add_column("W&B")

        status_colors = {
            "RUNNING": "green",
            "COMPLETED": "blue",
            "FAILED": "red",
            "CANCELLED": "yellow",
            "PENDING": "cyan",
            "SUBMITTED": "cyan",
            "TIMEOUT": "red",
        }

        for exp in experiments:
            color = status_colors.get(exp["status"], "white")
            config_str = exp.get("config_json", "{}")
            if config_str == "{}":
                config_str = ""
            wandb = exp.get("wandb_url") or ""

            table.add_row(
                exp["job_id"],
                exp["commit_short"],
                f"[{color}]{exp['status']}[/{color}]",
                exp["submitted_at"][:19],
                config_str,
                wandb,
            )

        console.print(table)
