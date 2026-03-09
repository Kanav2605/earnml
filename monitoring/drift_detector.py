"""
monitoring/drift_detector.py
──────────────────────────────
Production model monitoring for EarnML.

Monitors two things:
  1. Data drift    — have feature distributions shifted? (PSI)
  2. Model drift   — has prediction accuracy degraded? (rolling metrics)

PSI (Population Stability Index):
  PSI < 0.1   → No significant drift
  PSI 0.1–0.2 → Moderate drift, investigate
  PSI > 0.2   → Significant drift, retrain recommended

Why JPMorgan cares:
  A model trained on 2021-2023 data may fail in a high-rate 2024
  environment. Drift detection catches this before users notice.
  This is one of the most asked-about topics in ML system design
  interviews at banks.

Usage:
  detector = DriftDetector("AAPL")
  detector.set_reference(training_features_df)
  report = detector.check(current_features_df)
  if report["retrain_recommended"]:
      trigger_retraining("AAPL")
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import joblib

warnings.filterwarnings("ignore")

MONITOR_DIR = Path(__file__).parent / "reference_data"
MONITOR_DIR.mkdir(exist_ok=True)

ALERT_THRESHOLDS = {
    "psi_warning":  0.10,   # investigate
    "psi_critical": 0.20,   # retrain
    "prediction_accuracy_min": 0.52,  # below this → model may have degraded
}


# ─────────────────────────────────────────────────────────────────
# PSI (Population Stability Index)
# ─────────────────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Compute PSI between reference (training) and current distributions.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    A PSI > 0.2 suggests the current data has drifted significantly
    from the training distribution — model retraining likely needed.
    """
    # Create bins from reference data
    breakpoints = np.nanpercentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates

    # Count observations per bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current,   bins=breakpoints)[0]

    # Convert to percentages, clip to avoid log(0)
    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(current),   1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_ks_statistic(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov test for distribution shift.
    Returns statistic and p-value.
    """
    from scipy import stats
    stat, pval = stats.ks_2samp(reference, current)
    return {"ks_statistic": round(float(stat), 4), "ks_pvalue": round(float(pval), 4)}


# ─────────────────────────────────────────────────────────────────
# Drift Detector class
# ─────────────────────────────────────────────────────────────────

class DriftDetector:

    def __init__(self, ticker: str):
        self.ticker   = ticker
        self.ref_path = MONITOR_DIR / f"{ticker}_reference.joblib"
        self._reference: Optional[pd.DataFrame] = None
        self._load_reference()

    # ── Reference management ──────────────────────────────────────

    def set_reference(self, df: pd.DataFrame, feature_cols: list[str] = None):
        """
        Save the training distribution as reference baseline.
        Call this once after training, then check() on each inference.
        """
        if feature_cols:
            df = df[feature_cols]
        numeric = df.select_dtypes(include=[np.number])
        stats = {
            "mean":  numeric.mean().to_dict(),
            "std":   numeric.std().to_dict(),
            "p25":   numeric.quantile(0.25).to_dict(),
            "p50":   numeric.quantile(0.50).to_dict(),
            "p75":   numeric.quantile(0.75).to_dict(),
            "min":   numeric.min().to_dict(),
            "max":   numeric.max().to_dict(),
            "cols":  list(numeric.columns),
            "n":     len(df),
            "saved_at": datetime.utcnow().isoformat(),
        }
        # Store raw sample for PSI computation (sample 2000 rows max)
        sample = numeric.sample(min(len(numeric), 2000), random_state=42)
        joblib.dump({"stats": stats, "sample": sample}, self.ref_path)
        self._reference = {"stats": stats, "sample": sample}
        print(f"[drift] Reference set for {self.ticker} ({len(df)} rows, {len(numeric.columns)} features)")

    def _load_reference(self):
        if self.ref_path.exists():
            self._reference = joblib.load(self.ref_path)

    # ── Drift check ───────────────────────────────────────────────

    def check(self, current_df: pd.DataFrame) -> dict:
        """
        Compare current feature distributions against reference.
        Returns a full drift report.
        """
        if self._reference is None:
            return {
                "status": "NO_REFERENCE",
                "message": "Call set_reference() after training first.",
                "drift_detected": False,
                "retrain_recommended": False,
            }

        ref_sample = self._reference["sample"]
        ref_stats  = self._reference["stats"]

        # Only check features present in both
        common_cols = [c for c in ref_stats["cols"] if c in current_df.columns]
        current_numeric = current_df[common_cols].select_dtypes(include=[np.number])

        feature_reports = {}
        drift_count     = 0
        critical_count  = 0

        for col in common_cols:
            if col not in ref_sample.columns:
                continue

            ref_vals = ref_sample[col].dropna().values
            cur_vals = current_numeric[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 2:
                continue

            psi  = compute_psi(ref_vals, cur_vals)
            ks   = compute_ks_statistic(ref_vals, cur_vals)

            # Z-score shift: how many std devs has the mean moved?
            ref_mean = float(ref_stats["mean"].get(col, 0))
            ref_std  = float(ref_stats["std"].get(col, 1)) or 1
            cur_mean = float(np.nanmean(cur_vals))
            z_shift  = abs(cur_mean - ref_mean) / ref_std

            level = "OK"
            if psi > ALERT_THRESHOLDS["psi_critical"]:
                level = "CRITICAL"
                critical_count += 1
                drift_count += 1
            elif psi > ALERT_THRESHOLDS["psi_warning"] or z_shift > 2.0:
                level = "WARNING"
                drift_count += 1

            feature_reports[col] = {
                "psi":          round(psi, 4),
                "ks_stat":      ks["ks_statistic"],
                "ks_pvalue":    ks["ks_pvalue"],
                "z_shift":      round(z_shift, 3),
                "ref_mean":     round(ref_mean, 4),
                "cur_mean":     round(cur_mean, 4),
                "level":        level,
            }

        # Sort by PSI descending
        sorted_feats = dict(sorted(
            feature_reports.items(),
            key=lambda x: x[1]["psi"], reverse=True
        ))

        total_checked = len(feature_reports)
        drift_rate    = drift_count / max(total_checked, 1)
        retrain       = critical_count > 0 or drift_rate > 0.3

        report = {
            "ticker":               self.ticker,
            "checked_at":           datetime.utcnow().isoformat(),
            "features_checked":     total_checked,
            "features_drifted":     drift_count,
            "features_critical":    critical_count,
            "drift_rate":           round(drift_rate, 3),
            "drift_detected":       drift_count > 0,
            "retrain_recommended":  retrain,
            "overall_status":       "CRITICAL" if critical_count > 0 else ("WARNING" if drift_count > 0 else "OK"),
            "top_drifted_features": list(sorted_feats.keys())[:5],
            "feature_details":      sorted_feats,
        }

        self._log_report(report)
        return report

    def _log_report(self, report: dict):
        """Append report to local log file."""
        log_path = MONITOR_DIR / f"{self.ticker}_drift_log.jsonl"
        with open(log_path, "a") as f:
            # Only write summary (not all feature details)
            summary = {k: v for k, v in report.items() if k != "feature_details"}
            f.write(json.dumps(summary) + "\n")

    def drift_history(self, days: int = 30) -> pd.DataFrame:
        """Load historical drift reports."""
        log_path = MONITOR_DIR / f"{self.ticker}_drift_log.jsonl"
        if not log_path.exists():
            return pd.DataFrame()
        records = []
        cutoff  = datetime.utcnow() - timedelta(days=days)
        with open(log_path) as f:
            for line in f:
                r = json.loads(line)
                if datetime.fromisoformat(r["checked_at"]) >= cutoff:
                    records.append(r)
        return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────
# Prediction accuracy monitoring
# ─────────────────────────────────────────────────────────────────

class PredictionMonitor:
    """
    Tracks rolling prediction accuracy by comparing past predictions
    against actual outcomes.

    Workflow:
        1. predict()     → save prediction to DB
        2. (5 days later) → fetch actual price, update outcome
        3. check_accuracy() → compute rolling win rate
    """

    def __init__(self, db_url: str = None):
        self.db_url = db_url or "sqlite:///monitoring/predictions.db"

    def record_prediction(self, ticker: str, prediction: dict) -> int:
        """Insert a new prediction record. Returns row ID."""
        import sqlite3
        Path("monitoring").mkdir(exist_ok=True)
        conn = sqlite3.connect("monitoring/predictions.db")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                ticker TEXT, predicted_at TEXT,
                signal TEXT, score REAL,
                beat_probability REAL, clf_prediction TEXT,
                current_price REAL, predicted_return REAL,
                sentiment TEXT, vol_regime TEXT,
                actual_return REAL, outcome_correct INTEGER,
                outcome_recorded_at TEXT
            )
        """)
        cur = conn.execute("""
            INSERT INTO predictions
              (ticker, predicted_at, signal, score, beat_probability,
               clf_prediction, current_price, predicted_return, sentiment, vol_regime)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            ticker,
            datetime.utcnow().isoformat(),
            prediction.get("composite", {}).get("signal"),
            prediction.get("composite", {}).get("score"),
            prediction.get("models", {}).get("classifier", {}).get("beat_probability"),
            prediction.get("models", {}).get("classifier", {}).get("prediction"),
            prediction.get("models", {}).get("price_predictor", {}).get("current_price"),
            prediction.get("models", {}).get("price_predictor", {}).get("predicted_return"),
            prediction.get("models", {}).get("sentiment", {}).get("overall_sentiment"),
            prediction.get("models", {}).get("volatility", {}).get("regime"),
        ))
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id

    def compute_rolling_accuracy(self, ticker: str = None, days: int = 30) -> dict:
        """
        Compute win rate for predictions that have outcomes recorded.
        """
        import sqlite3
        try:
            conn = sqlite3.connect("monitoring/predictions.db")
            query = """
                SELECT ticker, clf_prediction, actual_return, outcome_correct
                FROM predictions
                WHERE outcome_recorded_at IS NOT NULL
                  AND predicted_at >= date('now', ?)
                  {}
            """.format("AND ticker = ?" if ticker else "")
            params = [f"-{days} days"] + ([ticker] if ticker else [])
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if df.empty:
                return {"status": "NO_OUTCOMES", "message": "No outcomes recorded yet"}

            accuracy = df["outcome_correct"].mean()
            n        = len(df)
            return {
                "ticker":         ticker or "ALL",
                "window_days":    days,
                "n_predictions":  n,
                "accuracy":       round(float(accuracy), 3),
                "status":         "DEGRADED" if accuracy < ALERT_THRESHOLDS["prediction_accuracy_min"] else "OK",
                "retrain_flag":   accuracy < ALERT_THRESHOLDS["prediction_accuracy_min"],
            }
        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    parser = argparse.ArgumentParser(description="Drift Detection")
    parser.add_argument("--ticker",  default="AAPL")
    parser.add_argument("--set-ref", action="store_true", help="Set reference from training data")
    parser.add_argument("--check",   action="store_true", help="Run drift check")
    args = parser.parse_args()

    from ml.data.pipeline import build_master_dataset
    df = build_master_dataset(args.ticker)

    detector = DriftDetector(args.ticker)

    if args.set_ref:
        feature_cols = [c for c in df.columns if c not in {"target","fwd_ret","Open","High","Low","Close","Volume"}]
        detector.set_reference(df, feature_cols)
        print(f"Reference saved for {args.ticker}")

    if args.check or not args.set_ref:
        # Simulate drift by checking recent 60 days vs full history
        recent = df.tail(60)
        report = detector.check(recent)
        print(f"\n── Drift Report: {args.ticker} ──────────────────────────")
        print(f"  Status:              {report['overall_status']}")
        print(f"  Features checked:    {report['features_checked']}")
        print(f"  Features drifted:    {report['features_drifted']}")
        print(f"  Retrain recommended: {report['retrain_recommended']}")
        print(f"\n  Top drifted features:")
        for feat in report["top_drifted_features"]:
            d = report["feature_details"][feat]
            print(f"    {feat:<35} PSI={d['psi']:.4f}  [{d['level']}]")
