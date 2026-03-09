"""
mlflow_tracking/tracker.py
───────────────────────────
MLflow integration for EarnML.

What this logs:
  • Every training run: hyperparameters, metrics, feature importances
  • Model artifacts: serialised model files, plots
  • Dataset info: n_samples, feature count, class balance
  • Cross-validation scores per fold

Why JPMorgan cares:
  Production ML teams live and die by experiment reproducibility.
  If a model degraded in prod 6 months ago, you need to know
  exactly what changed — hyperparameters, data distribution, features.
  MLflow gives you that audit trail.

Usage:
  from mlflow_tracking.tracker import MLTracker

  tracker = MLTracker("earnings_classifier")
  with tracker.start_run(ticker="AAPL"):
      tracker.log_params({"n_estimators": 400, "max_depth": 4})
      tracker.log_metrics({"roc_auc": 0.68, "accuracy": 0.63})
      tracker.log_model(model, "xgboost_model")
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

# Default to local tracking if no server configured
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
EXPERIMENT_NAME = "EarnML"


class MLTracker:
    """
    Thin wrapper around MLflow that adds finance-specific conveniences
    and sensible defaults for our models.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        self._run = None
        self._start_time = None

    @contextmanager
    def start_run(self, ticker: str, tags: dict = {}):
        """Context manager — automatically ends run and logs duration."""
        self._start_time = time.time()
        with mlflow.start_run(
            run_name=f"{self.model_name}_{ticker}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            tags={
                "model":    self.model_name,
                "ticker":   ticker,
                "env":      os.getenv("ENVIRONMENT", "dev"),
                **tags,
            }
        ) as run:
            self._run = run
            print(f"[mlflow] Run started: {run.info.run_id[:8]}…")
            try:
                yield self
            finally:
                duration = round(time.time() - self._start_time, 2)
                mlflow.log_metric("training_duration_sec", duration)
                print(f"[mlflow] Run complete ({duration}s) → {MLFLOW_URI}")

    def log_params(self, params: dict):
        """Log model hyperparameters."""
        # MLflow has a param value length limit
        clean = {k: str(v)[:250] for k, v in params.items()}
        mlflow.log_params(clean)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log evaluation metrics."""
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_cv_scores(self, scores: list[float], metric_name: str = "roc_auc"):
        """Log individual cross-validation fold scores."""
        for i, s in enumerate(scores):
            mlflow.log_metric(f"cv_{metric_name}_fold_{i+1}", s)
        mlflow.log_metrics({
            f"cv_{metric_name}_mean": float(np.mean(scores)),
            f"cv_{metric_name}_std":  float(np.std(scores)),
        })

    def log_dataset_info(self, X: pd.DataFrame, y: pd.Series, split: str = "train"):
        """Log dataset statistics — important for reproducibility."""
        mlflow.log_params({
            f"{split}_n_samples":  len(X),
            f"{split}_n_features": X.shape[1],
            f"{split}_pos_rate":   round(float(y.mean()), 4),
            f"{split}_date_range": f"{X.index[0] if hasattr(X,'index') else 'N/A'} → {X.index[-1] if hasattr(X,'index') else 'N/A'}",
        })

    def log_feature_importance(self, model, feature_names: list[str], top_n: int = 30):
        """Log top-N feature importances as a JSON artifact."""
        try:
            imp = dict(zip(feature_names, model.feature_importances_))
            top = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n])
            # Save as artifact
            path = Path("/tmp/feature_importance.json")
            path.write_text(json.dumps(top, indent=2))
            mlflow.log_artifact(str(path), "feature_importance")
            # Also log top 10 as metrics for quick comparison
            for i, (feat, val) in enumerate(list(top.items())[:10]):
                mlflow.log_metric(f"top_feat_{i+1}_score", round(val, 6))
        except Exception as e:
            print(f"[mlflow] Feature importance logging failed: {e}")

    def log_model_xgb(self, model, X_sample: pd.DataFrame, model_name: str = "model"):
        """Log XGBoost model with input signature."""
        try:
            signature = infer_signature(X_sample, model.predict(X_sample))
            mlflow.xgboost.log_model(model, model_name, signature=signature)
        except Exception as e:
            print(f"[mlflow] XGB model logging failed: {e}")

    def log_model_sklearn(self, model, X_sample: pd.DataFrame, model_name: str = "model"):
        """Log sklearn-compatible model with input signature."""
        try:
            signature = infer_signature(X_sample, model.predict(X_sample))
            mlflow.sklearn.log_model(model, model_name, signature=signature)
        except Exception as e:
            print(f"[mlflow] sklearn model logging failed: {e}")

    def log_plot(self, fig, filename: str):
        """Log a matplotlib figure as an artifact."""
        import matplotlib.pyplot as plt
        path = Path(f"/tmp/{filename}")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(str(path), "plots")
        plt.close(fig)

    def log_prediction(self, ticker: str, result: dict):
        """Log an inference result for monitoring."""
        flat = {
            f"pred_{k}": v for k, v in result.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(flat)
        mlflow.log_param("prediction_ticker", ticker)
        mlflow.log_param("prediction_ts", datetime.utcnow().isoformat())

    @property
    def run_id(self) -> Optional[str]:
        return self._run.info.run_id if self._run else None


# ─────────────────────────────────────────────────────────────────
# Convenience functions for quick logging outside context manager
# ─────────────────────────────────────────────────────────────────

def log_training_run(
    ticker: str,
    model_name: str,
    params: dict,
    metrics: dict,
    cv_scores: list[float] = None,
    model=None,
    feature_names: list[str] = None,
) -> str:
    """
    One-shot function to log a complete training run.
    Returns the MLflow run ID.

    Example:
        run_id = log_training_run(
            ticker="AAPL",
            model_name="xgboost_classifier",
            params={"n_estimators": 400, "max_depth": 4},
            metrics={"roc_auc": 0.68, "accuracy": 0.63},
            cv_scores=[0.65, 0.70, 0.67, 0.71, 0.66],
        )
    """
    tracker = MLTracker(model_name)
    with tracker.start_run(ticker=ticker) as t:
        t.log_params(params)
        t.log_metrics(metrics)
        if cv_scores:
            t.log_cv_scores(cv_scores)
        if model is not None and feature_names is not None:
            t.log_feature_importance(model, feature_names)
        return t.run_id


# ─────────────────────────────────────────────────────────────────
# CLI: view recent runs
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=10,
            )
            print(f"\n── Recent MLflow Runs ({EXPERIMENT_NAME}) ────────────────")
            for r in runs:
                print(f"  {r.info.run_id[:8]}  {r.data.tags.get('model','?'):30s}  "
                      f"{r.data.tags.get('ticker','?'):6s}  "
                      f"AUC={r.data.metrics.get('cv_roc_auc_mean','?'):.3f}  "
                      f"{r.info.status}")
        else:
            print(f"No experiment '{EXPERIMENT_NAME}' found. Run a training first.")
    except Exception as e:
        print(f"MLflow connection error: {e}")
        print(f"Tracking URI: {MLFLOW_URI}")
