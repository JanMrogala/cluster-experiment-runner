# Cluster Experiment Runner — High-Level Specification

## Goal

Enable a coding agent (e.g. PlotCode) to submit, monitor, and cancel GPU experiments on a remote SLURM cluster — without any persistent service running on the cluster.

## Architecture

Two components connected via SSH:

1. **Local Control Layer** — runs on the developer's machine, exposes a simple API to the agent
2. **Cluster Execution Layer** — stateless bash scripts triggered via SSH

## Core Principle

**Commit = Experiment.** Every experiment is identified by a git commit hash, ensuring reproducibility.

## Workflow

1. Agent modifies code, commits, and pushes to GitHub
2. Agent calls local API: `submit_experiment(commit_hash, config) → job_id`
3. Local layer SSHs into cluster, triggers `run_experiment.sh <commit_hash>`
4. Cluster script:
   - Fetches from GitHub (with retry loop until commit is available)
   - Creates a git worktree at `/worktrees/<commit_hash>`
   - Submits SLURM job via `sbatch`
5. SLURM runs the experiment inside a Singularity container from the worktree
6. Experiment logs to Weights & Biases (the local agents will need to know some identifier in Wandb to know which experiment is which. this can can be coded in the experiment script)

## Local API (Agent-facing)

```
submit_experiment(commit_hash, config) → job_id
get_status(job_id) → status
cancel_experiment(job_id)
list_experiments()
```

## Cluster-Side Layout

```
repo/                     # single shared git repo
worktrees/<commit_hash>/  # one worktree per experiment
logs/<job_id>.{out,err}   # SLURM output
```

## Key Design Decisions

- **SSH-only** — no daemons or open ports on the cluster
- **Git worktrees** — fast, space-efficient, isolated per experiment
- **Cluster-side polling** — retry `git fetch` until pushed commit appears (handles push race condition)
- **Singularity containers** — reproducible execution environment
- **W&B for results** — agent reads results via W&B API, not from cluster filesystem

## Future Extensions (not in scope now)

- Experiment queueing / priorities
- Auto-cleanup of old worktrees
- Job retry on failure
- Hyperparameter sweeps
- Metadata DB mapping job_id ↔ commit ↔ W&B run
