# 🧠 EarnML v3 — Production-Grade ML Earnings Intelligence

> A ML engineering portfolio project demonstrating the full production lifecycle.

[![CI/CD](https://github.com/YOUR_USERNAME/earnml/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/earnml/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)

---

## 🎯 What This Demonstrates

| Hiring Criteria | Where Demonstrated |
|---|---|
| **ML Models** | Stacked ensemble (XGB+LGB+CAT), LSTM, FinBERT, GARCH |
| **Feature Engineering** | 40+ technical + macro features, regime-aware |
| **Model Evaluation** | TimeSeriesSplit CV, ROC-AUC, directional accuracy |
| **MLOps** | MLflow experiment tracking, model versioning |
| **Monitoring** | PSI drift detection, rolling accuracy tracking |
| **Software Engineering** | Type hints, pytest suite, Docker, CI/CD |
| **Data Engineering** | Multi-source pipeline (Yahoo Finance + FRED macro) |
| **LLM Integration** | Claude AI generates institutional analyst commentary |
| **Backtesting** | Sharpe, Sortino, max drawdown vs benchmark |
| **API Design** | FastAPI REST + WebSocket, SQLite persistence |
| **Cloud-Ready** | Docker Compose, Render + Vercel deploy configs |

---

## 🏗️ Architecture

```
earnml-v3/
├── ml/
│   ├── data/
│   │   └── pipeline.py          # Multi-source data: Yahoo Finance + FRED macro
│   ├── ensemble/
│   │   └── stacked_classifier.py # XGBoost + LightGBM + CatBoost → LogReg meta
│   └── models/
│       ├── price_predictor.py    # 2-layer LSTM (60-day sequences)
│       ├── sentiment_analyzer.py # FinBERT + VADER ensemble
│       └── volatility_model.py   # GARCH(1,1)-t
│
├── api/
│   ├── main.py                   # FastAPI + WebSocket + background tasks
│   └── db.py                     # SQLAlchemy persistence layer
│
├── mlflow_tracking/
│   └── tracker.py                # MLflow experiment logging
│
├── monitoring/
│   └── drift_detector.py         # PSI drift + prediction accuracy tracking
│
├── backtesting/
│   └── engine.py                 # Sharpe/Sortino/drawdown strategy simulation
│
├── llm_layer/
│   └── analyst.py                # Claude AI analyst commentary generation
│
├── tests/
│   └── test_pipeline.py          # pytest suite with 25+ tests
│
├── docker/
│   ├── Dockerfile                # Multi-stage Python image
│   ├── Dockerfile.frontend       # Nginx React image
│   ├── docker-compose.yml        # Full stack: API + MLflow + Postgres + Redis
│   └── nginx.conf
│
└── .github/
    └── workflows/ci.yml          # GitHub Actions: test → build → deploy
```

---

## ⚡ Quickstart

### Option A: Local dev (fastest)
```bash
git clone https://github.com/YOUR_USERNAME/earnml && cd earnml-v3
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Test a single model (no GPU, fast)
python ml/models/volatility_model.py --ticker AAPL
python ml/ensemble/stacked_classifier.py --ticker AAPL --train

# 2. Start API
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs

# 3. Start frontend
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

### Option B: Full Docker stack
```bash
cd docker
docker-compose up --build
# API     → http://localhost:8000
# Frontend → http://localhost:3000
# MLflow  → http://localhost:5000
```

---

## 🧪 Running Tests

```bash
# Full suite with coverage
pytest tests/ -v --cov=ml --cov=api --cov=monitoring --cov-report=html

# Just data pipeline tests
pytest tests/ -k "TestFeatureEngineering" -v

# Just drift detection tests
pytest tests/ -k "TestDriftDetection" -v

# Open coverage report
open htmlcov/index.html
```

---

## 📊 ML Models

### Stacked Ensemble (earnings direction)
```
Level 0: XGBoost + LightGBM + CatBoost
         ↓ out-of-fold predictions (TimeSeriesSplit)
Level 1: Logistic Regression meta-learner
Output:  Beat probability % + individual model votes
```

### LSTM Price Predictor
- Input: 60-day sliding windows × 17 features
- Architecture: LSTM(128) → BatchNorm → LSTM(64) → Dense(32) → output
- Target: % return over next N trading days

### Sentiment Pipeline
- VADER: fast rule-based baseline
- FinBERT (ProsusAI): fine-tuned BERT on financial text
- Blended: 40% VADER + 60% FinBERT

### GARCH(1,1) Volatility
- Student-t innovations (fat tails)
- AR(1) mean model
- Outputs: conditional vol, 10d forecast, VaR 95/99%

---

## 🔍 Monitoring & MLOps

```bash
# Set reference distribution after training
python monitoring/drift_detector.py --ticker AAPL --set-ref

# Check for drift on recent data
python monitoring/drift_detector.py --ticker AAPL --check

# View MLflow experiments
mlflow ui --port 5000
# → http://localhost:5000

# Run backtest
python backtesting/engine.py --ticker AAPL
```

---

## 🤖 LLM Analyst Commentary

Set `ANTHROPIC_API_KEY` in your `.env` and enable with `use_llm=true` in the API request.
The commentary tab shows Claude-generated institutional-style analysis of the ML outputs.
Falls back to rule-based commentary if the API key is not set.

```bash
# Test commentary directly
python llm_layer/analyst.py AAPL
```

---

## 🌐 Deployment

### Backend → Render.com
1. Connect GitHub repo → New Web Service
2. Root dir: `.`  |  Build: `pip install -r requirements.txt`
3. Start: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
4. Env vars: `ANTHROPIC_API_KEY`, `MLFLOW_TRACKING_URI`
5. Add `RENDER_DEPLOY_HOOK` to GitHub secrets for auto-deploy

### Frontend → Vercel
1. Import repo → Root dir: `frontend`
2. Add env var: `VITE_API_URL=https://your-render-url.onrender.com`
3. Add `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID` to GitHub secrets


---

## ⚠️ Disclaimer
Educational portfolio project. Not financial advice.
