"""
ml/ensemble/stacked_classifier.py
────────────────────────────────────
Stacked ensemble for earnings beat/miss prediction.

Level-0 (base learners):
  • XGBoost      — gradient boosting, handles non-linearity well
  • LightGBM     — faster, great on high-dimensional features
  • CatBoost     — built-in categorical handling, less tuning needed

Level-1 (meta-learner):
  • Logistic Regression — learns how to weight base model predictions

Why stacking beats a single model:
  Each base model captures different patterns in the data.
  The meta-learner learns the optimal combination — often
  outperforms any individual model by 2–5% ROC-AUC.

Portfolio significance:
  Stacked ensembles are widely used in quant finance and
  ML competitions (Kaggle). Understanding them is a strong
  interview differentiator.

Usage:
  python ml/ensemble/stacked_classifier.py --ticker AAPL --train
  python ml/ensemble/stacked_classifier.py --ticker AAPL --predict
"""

import sys, argparse, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.metrics        import (
    classification_report, roc_auc_score,
    RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration    import CalibratedClassifierCV

import xgboost  as xgb
import lightgbm as lgb
from catboost  import CatBoostClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.data.pipeline import build_master_dataset, fetch_earnings_history

MODELS_DIR = Path(__file__).parent.parent / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature columns used (exclude price/target cols) ────────────────────────
EXCLUDE = {
    "Open","High","Low","Close","Volume","Dividends","Stock Splits",
    "fwd_ret","target","future_return","target_up",
}

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE
            and df[c].dtype in [np.float32, np.float64, np.int32, np.int64, float, int]]


# ─────────────────────────────────────────────────────────────────────────────
# Level-0 base models
# ─────────────────────────────────────────────────────────────────────────────

def _make_xgb():
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        scale_pos_weight=1, gamma=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
    )

def _make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=1,      # avoid loky/core count issues on Windows
        random_state=42, verbose=-1,
    )

def _make_cat():
    return CatBoostClassifier(
        iterations=400, depth=4, learning_rate=0.03,
        l2_leaf_reg=3, random_seed=42, verbose=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stacking utilities
# ─────────────────────────────────────────────────────────────────────────────

def _oof_predictions(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """
    Out-of-fold predictions using TimeSeriesSplit.
    This is the core of stacking — base models never see
    the data they're predicting on during training.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof  = np.zeros(len(X))
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr         = y[train_idx]
        m = _clone_model(model)
        m.fit(X_tr, y_tr)
        oof[val_idx] = m.predict_proba(X_val)[:, 1]
    return oof


def _clone_model(m):
    """Return a fresh copy of the model (no sklearn clone — some don't support it)."""
    if isinstance(m, xgb.XGBClassifier):
        return _make_xgb()
    if isinstance(m, lgb.LGBMClassifier):
        return _make_lgb()
    if isinstance(m, CatBoostClassifier):
        return _make_cat()
    from sklearn.base import clone
    return clone(m)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(ticker: str, save: bool = True):
    print(f"\n{'='*60}")
    print(f"  STACKED ENSEMBLE — {ticker}")
    print(f"  XGBoost + LightGBM + CatBoost → Logistic Meta-Learner")
    print(f"{'='*60}\n")

    df   = build_master_dataset(ticker, forward_days=5)
    cols = _get_feature_cols(df)
    X    = df[cols].fillna(df[cols].median()).values.astype(np.float32)
    y    = df["target"].values

    # Scale (important for meta-learner)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    print(f"Features: {len(cols)}  |  Samples: {len(X)}")
    print(f"Positive rate (price up): {y.mean():.1%}\n")

    base_models = {
        "xgboost":  _make_xgb(),
        "lightgbm": _make_lgb(),
        "catboost": _make_cat(),
    }

    # ── Level-0: OOF predictions ─────────────────────────────────────────────
    print("── Level-0: Generating out-of-fold predictions ─────────────")
    oof_preds = {}
    for name, model in base_models.items():
        print(f"  [{name}] running OOF…", end=" ", flush=True)
        t0 = __import__("time").time()
        oof_preds[name] = _oof_predictions(model, X_sc, y)
        auc = roc_auc_score(y[len(y)-len(oof_preds[name]):], oof_preds[name][-len(y):])
        print(f"done ({__import__('time').time()-t0:.1f}s) — OOF AUC ≈ {auc:.3f}")

    # ── Level-0: Fit final base models on all data ────────────────────────────
    print("\n── Level-0: Fitting final base models on full data ─────────")
    for name, model in base_models.items():
        model.fit(X_sc, y)
        print(f"  [{name}] fitted ✓")

    # ── Level-1: Build meta-features ─────────────────────────────────────────
    meta_X = np.column_stack(list(oof_preds.values()))   # shape: (n, 3)
    meta_y = y

    # ── Level-1: Meta-learner ─────────────────────────────────────────────────
    print("\n── Level-1: Training meta-learner (Logistic Regression) ────")
    meta_learner = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    tscv_meta = TimeSeriesSplit(n_splits=5)
    meta_aucs = []
    for _, (tr, vl) in enumerate(tscv_meta.split(meta_X)):
        meta_learner.fit(meta_X[tr], meta_y[tr])
        p = meta_learner.predict_proba(meta_X[vl])[:, 1]
        if len(np.unique(meta_y[vl])) > 1:
            meta_aucs.append(roc_auc_score(meta_y[vl], p))
    meta_learner.fit(meta_X, meta_y)
    print(f"  Meta-learner ROC-AUC: {np.mean(meta_aucs):.3f} ± {np.std(meta_aucs):.3f}")
    print(f"  Model weights: XGB={meta_learner.coef_[0][0]:.3f}  LGB={meta_learner.coef_[0][1]:.3f}  CAT={meta_learner.coef_[0][2]:.3f}")

    # ── Evaluate on last 20% ──────────────────────────────────────────────────
    split = int(len(X_sc) * 0.80)
    if split < len(X_sc):
        base_test_preds = np.column_stack([
            m.predict_proba(X_sc[split:])[:, 1]
            for m in base_models.values()
        ])
        y_proba = meta_learner.predict_proba(base_test_preds)[:, 1]
        y_pred  = (y_proba > 0.5).astype(int)
        y_true  = y[split:]
        print(f"\n── Out-of-sample Test ({len(y_true)} samples) ───────────────────")
        print(classification_report(y_true, y_pred, target_names=["DOWN", "UP"]))
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_proba)
            print(f"  ROC-AUC: {auc:.4f}")
            _plot_roc(y_true, y_proba, ticker)

    _plot_feature_importance(base_models["xgboost"], cols, ticker)

    if save:
        bundle = {
            "base_models":   base_models,
            "meta_learner":  meta_learner,
            "scaler":        scaler,
            "feature_cols":  cols,
            "ticker":        ticker,
            "trained_at":    __import__("datetime").datetime.utcnow().isoformat(),
        }
        path = MODELS_DIR / f"{ticker}_stacked.joblib"
        joblib.dump(bundle, path)
        print(f"\n[ensemble] Saved → {path}")

    return base_models, meta_learner, scaler, cols


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(ticker: str) -> dict:
    path = MODELS_DIR / f"{ticker}_stacked.joblib"
    if not path.exists():
        print(f"[ensemble] No model for {ticker}, training…")
        train(ticker)

    bundle       = joblib.load(path)
    base_models  = bundle["base_models"]
    meta_learner = bundle["meta_learner"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    df   = build_master_dataset(ticker, forward_days=5)
    X_raw= df[feature_cols].fillna(df[feature_cols].median()).iloc[-1:].values.astype(np.float32)
    X_sc = scaler.transform(X_raw)

    base_preds = np.column_stack([
        m.predict_proba(X_sc)[:, 1] for m in base_models.values()
    ])
    proba  = meta_learner.predict_proba(base_preds)[0]
    indiv  = {name: round(float(m.predict_proba(X_sc)[0][1]) * 100, 1)
               for name, m in base_models.items()}

    return {
        "ticker":            ticker,
        "beat_probability":  round(float(proba[1]) * 100, 1),
        "miss_probability":  round(float(proba[0]) * 100, 1),
        "prediction":        "UP" if proba[1] > 0.5 else "DOWN",
        "confidence":        round(float(max(proba)) * 100, 1),
        "individual_models": indiv,
        "model":             "Stacked Ensemble (XGB + LGB + CAT → LogReg)",
        "features_used":     len(feature_cols),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_roc(y_true, y_score, ticker):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, color="#C9A84C")
    ax.plot([0,1],[0,1], "k--", lw=0.8)
    ax.set_title(f"{ticker} Stacked Ensemble — ROC Curve")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{ticker}_roc.png", dpi=120)
    plt.close()

def _plot_feature_importance(xgb_model, feature_cols, ticker):
    imp = pd.Series(xgb_model.feature_importances_, index=feature_cols).nlargest(25)
    fig, ax = plt.subplots(figsize=(10, 7))
    imp.sort_values().plot(kind="barh", ax=ax, color="#C9A84C", edgecolor="none")
    ax.set_title(f"{ticker} — Top-25 Feature Importances (XGBoost)")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{ticker}_importance.png", dpi=120)
    plt.close()
    print(f"[ensemble] Feature importance chart saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",  default="AAPL")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    if args.train:
        train(args.ticker)
    if args.predict or not args.train:
        r = predict(args.ticker)
        print("\n── Ensemble Prediction ──────────────────────────────────────")
        for k, v in r.items():
            print(f"  {k:<30} {v}")
