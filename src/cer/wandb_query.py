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


def _extract_config(run: wandb.apis.public.Run) -> dict:
    """Extract the run config as a plain dict.

    The wandb public API exposes `run.config` as a dict-like object whose
    representation can vary across wandb versions. Try the most reliable
    paths first and fall back to a manual copy.
    """
    cfg = run.config
    # Newer wandb versions expose `as_dict()`.
    as_dict = getattr(cfg, "as_dict", None)
    if callable(as_dict):
        try:
            return dict(as_dict())
        except Exception:
            pass
    # `dict(cfg)` works when cfg implements `keys()` (most wandb versions).
    try:
        out = dict(cfg)
        if out:
            return out
    except Exception:
        pass
    # Last resort: iterate items().
    items = getattr(cfg, "items", None)
    if callable(items):
        try:
            return {k: v for k, v in items()}
        except Exception:
            pass
    return {}


def get_run_summary(run: wandb.apis.public.Run) -> dict:
    """Extract key information from a W&B run."""
    return {
        "run_id": run.id,
        "run_name": run.name,
        "url": run.url,
        "state": run.state,
        "config": _extract_config(run),
        "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
        "tags": list(run.tags),
        "created_at": run.created_at,
    }


def get_run_history(run: wandb.apis.public.Run, keys: list[str] | None = None) -> list[dict]:
    """Get the full metric history for a run.

    Uses `run.scan_history` which streams every logged step. Unlike
    `run.history(keys=...)`, scan_history returns each individual logged
    event rather than only rows where every requested key is present —
    important for frameworks like PyTorch Lightning that log train and
    val metrics on different steps.

    Args:
        run: W&B run object.
        keys: Optional list of metric keys to fetch. If None, fetches all.
    """
    if keys:
        rows = list(run.scan_history(keys=keys))
        if rows:
            return rows
        # Some wandb backends return nothing from scan_history when keys
        # are passed (e.g. if a key was never logged). Fall back to a full
        # scan and filter client-side so the caller still gets data.
        wanted = set(keys)
        return [
            {k: v for k, v in row.items() if k in wanted or k == "_step"}
            for row in run.scan_history()
            if any(k in row for k in wanted)
        ]
    return list(run.scan_history())
