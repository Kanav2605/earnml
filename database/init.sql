-- ─────────────────────────────────────────────────────────────────
-- EarnML Database Schema
-- ─────────────────────────────────────────────────────────────────

-- Create mlflow database (separate from app db)
CREATE DATABASE mlflow;

-- ── Predictions table ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10)  NOT NULL,
    predicted_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Composite signal
    signal          VARCHAR(20),
    score           FLOAT,

    -- Classifier
    beat_probability    FLOAT,
    clf_prediction      VARCHAR(10),
    clf_confidence      FLOAT,

    -- LSTM
    current_price       FLOAT,
    predicted_price     FLOAT,
    predicted_return    FLOAT,
    lstm_direction      VARCHAR(10),

    -- Sentiment
    sentiment_label     VARCHAR(20),
    sentiment_score     FLOAT,

    -- Volatility
    vol_regime          VARCHAR(20),
    conditional_vol     FLOAT,
    garch_forecast_5d   FLOAT,
    var_99_1d           FLOAT,

    -- Macro context
    vix                 FLOAT,
    t10y_rate           FLOAT,
    spx_ret_20d         FLOAT,

    -- Metadata
    elapsed_seconds     FLOAT,
    model_version       VARCHAR(50),
    features_used       INTEGER,

    -- Outcome tracking (filled in later)
    actual_beat         BOOLEAN,
    actual_price_5d     FLOAT,
    actual_return_5d    FLOAT,
    outcome_recorded_at TIMESTAMPTZ
);

CREATE INDEX idx_predictions_ticker ON predictions(ticker);
CREATE INDEX idx_predictions_predicted_at ON predictions(predicted_at DESC);

-- ── Model performance tracking ────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_metrics (
    id              SERIAL PRIMARY KEY,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker          VARCHAR(10),
    model_name      VARCHAR(50) NOT NULL,
    metric_name     VARCHAR(50) NOT NULL,
    metric_value    FLOAT       NOT NULL,
    split           VARCHAR(20),          -- 'train', 'val', 'test'
    mlflow_run_id   VARCHAR(100)
);

CREATE INDEX idx_metrics_model ON model_metrics(model_name, recorded_at DESC);

-- ── Feature drift tracking ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feature_drift (
    id              SERIAL PRIMARY KEY,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker          VARCHAR(10) NOT NULL,
    feature_name    VARCHAR(100) NOT NULL,
    reference_mean  FLOAT,
    reference_std   FLOAT,
    current_mean    FLOAT,
    current_std     FLOAT,
    psi_score       FLOAT,          -- Population Stability Index
    drift_detected  BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_drift_ticker ON feature_drift(ticker, recorded_at DESC);

-- ── Backtest results ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS backtest_results (
    id              SERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker          VARCHAR(10) NOT NULL,
    strategy        VARCHAR(50) NOT NULL,
    start_date      DATE,
    end_date        DATE,
    total_trades    INTEGER,
    win_rate        FLOAT,
    total_return    FLOAT,
    sharpe_ratio    FLOAT,
    max_drawdown    FLOAT,
    calmar_ratio    FLOAT,
    params          JSONB
);

-- ── Convenience view: latest prediction per ticker ────────────────
CREATE OR REPLACE VIEW latest_predictions AS
SELECT DISTINCT ON (ticker)
    *
FROM predictions
ORDER BY ticker, predicted_at DESC;

COMMENT ON TABLE predictions       IS 'All ML model predictions with outcomes for accuracy tracking';
COMMENT ON TABLE model_metrics     IS 'Model evaluation metrics logged per training run';
COMMENT ON TABLE feature_drift     IS 'Feature distribution monitoring for concept drift detection';
COMMENT ON TABLE backtest_results  IS 'Strategy backtesting performance results';
