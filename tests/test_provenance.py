"""Provenance guard: every registered model version records its training run id.

Source-level assertion (the trainers import mlflow/pandas/pyspark and can't be
imported here) that both trainers stamp a ``training_run_id`` tag on each model
version, so a served prediction can be traced back to the exact artifact.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(REPO_ROOT, "airflow", "jobs")


def _read(job_filename):
    with open(os.path.join(JOBS_DIR, job_filename), encoding="utf-8") as fh:
        return fh.read()


def test_both_trainers_tag_training_run_id():
    for trainer in ("train_models_pandas.py", "train_models.py"):
        src = _read(trainer)
        assert "set_model_version_tag" in src, (
            f"{trainer} must tag each version for provenance"
        )
        assert "training_run_id" in src, (
            f"{trainer} must record the training_run_id tag"
        )
