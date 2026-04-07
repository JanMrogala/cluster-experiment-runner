"""Query W&B API for experiment results."""
from __future__ import annotations

import os

import wandb


class WandbError(Exception):
    pass


def _get_api(api_key: str = "") -> wandb.Api:
    """Get a W&B API client, using the provided key if set."""
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    return wandb.Api()


def find_run(project: str, commit_hash: str, entity: str = "", api_key: str = "") -> wandb.apis.public.Run | None:
    """Find a W&B run by commit hash tag."""
    api = _get_api(api_key)
    path = f"{entity}/{project}" if entity else project

    # Search by tag (we set WANDB_TAGS=commit_hash in sbatch)
    runs = api.runs(path, filters={"tags": {"$in": [commit_hash]}})
    runs_list = list(runs)

    if not runs_list:
        # Fallback: search by run name pattern cer-<commit_short>
        commit_short = commit_hash[:8]
        runs = api.runs(path, filters={"display_name": f"cer-{commit_short}"})
        runs_list = list(runs)

    if not runs_list:
        return None

    # Return most recent match
    return runs_list[0]


def get_run_summary(run: wandb.apis.public.Run) -> dict:
    """Extract key information from a W&B run."""
    return {
        "run_id": run.id,
        "run_name": run.name,
        "url": run.url,
        "state": run.state,
        "config": dict(run.config),
        "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
        "tags": list(run.tags),
        "created_at": run.created_at,
    }


def get_run_history(run: wandb.apis.public.Run, keys: list[str] | None = None) -> list[dict]:
    """Get the full metric history for a run.

    Args:
        run: W&B run object
        keys: Optional list of metric keys to fetch. If None, fetches all.
    """
    df = run.history(keys=keys) if keys else run.history()
    # history() returns a pandas DataFrame
    return df.to_dict("records")
