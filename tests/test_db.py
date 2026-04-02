import pytest

from cer.db import Database


def test_insert_and_get(tmp_db):
    tmp_db.insert_experiment(
        job_id="12345",
        commit_hash="abcdef1234567890abcdef1234567890abcdef12",
        config={"lr": "0.001"},
        cluster_host="testhost",
        partition="gpu",
    )
    exp = tmp_db.get_experiment("12345")
    assert exp is not None
    assert exp["job_id"] == "12345"
    assert exp["commit_short"] == "abcdef12"
    assert exp["status"] == "SUBMITTED"
    assert '"lr": "0.001"' in exp["config_json"]


def test_update_status(tmp_db):
    tmp_db.insert_experiment(
        job_id="12345",
        commit_hash="abcdef1234567890abcdef1234567890abcdef12",
        config={},
        cluster_host="testhost",
        partition="gpu",
    )
    tmp_db.update_status("12345", "RUNNING")
    exp = tmp_db.get_experiment("12345")
    assert exp["status"] == "RUNNING"


def test_update_wandb(tmp_db):
    tmp_db.insert_experiment(
        job_id="12345",
        commit_hash="abcdef1234567890abcdef1234567890abcdef12",
        config={},
        cluster_host="testhost",
        partition="gpu",
    )
    tmp_db.update_wandb("12345", "run_abc", "https://wandb.ai/team/project/runs/run_abc")
    exp = tmp_db.get_experiment("12345")
    assert exp["wandb_run_id"] == "run_abc"
    assert "wandb.ai" in exp["wandb_url"]


def test_list_experiments(tmp_db):
    for i in range(3):
        tmp_db.insert_experiment(
            job_id=str(10000 + i),
            commit_hash=f"{'a' * 8}{i:032d}",
            config={},
            cluster_host="testhost",
            partition="gpu",
        )
    results = tmp_db.list_experiments()
    assert len(results) == 3


def test_list_active(tmp_db):
    tmp_db.insert_experiment(
        job_id="100", commit_hash="a" * 40, config={},
        cluster_host="h", partition="gpu",
    )
    tmp_db.insert_experiment(
        job_id="101", commit_hash="b" * 40, config={},
        cluster_host="h", partition="gpu",
    )
    tmp_db.update_status("101", "COMPLETED")

    active = tmp_db.list_active()
    assert len(active) == 1
    assert active[0]["job_id"] == "100"


def test_duplicate_job_id_raises(tmp_db):
    tmp_db.insert_experiment(
        job_id="12345", commit_hash="a" * 40, config={},
        cluster_host="h", partition="gpu",
    )
    with pytest.raises(Exception):
        tmp_db.insert_experiment(
            job_id="12345", commit_hash="b" * 40, config={},
            cluster_host="h", partition="gpu",
        )
