# NY Jets Diagnostic Lab

> **Research question:** Why do the New York Jets chronically underperform in the NFL — and which measurable levers would most improve their win probability?

A full-stack sports analytics project: ingestion pipeline → feature engineering → ML models → calibrated probability outputs → interactive Streamlit dashboard with What-if simulator. Data covers all NFL seasons 2016–2025.

---

## Key Findings

The Jets finished 2025 **3–14 (17.6% win rate)** — **31.2 percentage points below the league average**. Since 2016, they average **4.9 wins per season** and have achieved **zero winning seasons** in a decade. The underperformance is structural, not random.

Top root causes identified by the model:

| Driver | Jets | League Avg | Problem Score |
|---|---|---|---|
| Avg Cap Share per Contract | 0.004 | 0.006 | 102.0 |
| Offense Sack Rate | 4.5% | 3.0% | 44.7 |
| Defense Takeaway Rate | 0.2% | 1.5% | 24.7 |
| Active Roster Contract Slots | 86.0 | 82.8 | 23.0 |
| Offense Success Rate | 45.3% | 46.8% | 18.0 |

---

## Model Results

Rolling-origin backtests — each season trained exclusively on prior seasons (no data leakage):

| Model | ROC-AUC | Log Loss | Brier Score | ECE | Beat Benchmark |
|---|---:|---:|---:|---:|---:|
| **Gradient Boosting (LightGBM)** | **0.6764** | **0.6476** | **0.2278** | 0.0683 | **7 / 7 seasons** |
| Logistic Regression (baseline) | 0.6417 | 0.6761 | 0.2391 | 0.0785 | 6 / 7 seasons |
| Naive home-advantage benchmark | 0.5000 | 0.6913 | 0.2491 | 0.0329 | — |

The Gradient Boosting model beats the naive benchmark in all 7 test seasons on both Log Loss and Brier Score. Calibration is sufficient for decision support (ECE 0.068).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | `nflreadpy` / nflverse, SQLite, Parquet |
| Feature engineering | pandas, numpy, pyarrow |
| ML models | LightGBM, scikit-learn (LogisticRegression) |
| Evaluation | Rolling-origin CV, ROC-AUC, Brier, ECE |
| Visualization | Streamlit, Altair |
| Package management | `uv`, `pyproject.toml` |
| Testing | pytest |

---

## Project Structure

```
ny-jets-diagnostic-lab/
├── jets_project/            # Core Python package
│   └── __init__.py          #   ingest · features · train · evaluate
├── app/
│   └── dashboard.py         # Streamlit app (League Overview, Tracking, What-if)
├── src/
│   └── jets_project/        # Importable package for pipeline stages
├── data/
│   └── reference/
│       └── staff_tenure.csv # Manually curated coaching/FO tenure data
├── artifacts/
│   ├── models/
│   │   └── model_bundle.joblib  # Trained model + feature config
│   └── app/
│       └── jets_app.sqlite  # Slim deployment DB for Streamlit Cloud
├── reports/
│   ├── final_report.md      # Full written analysis with all findings
│   ├── slide_deck_outline.md
│   └── deployment_guide.md
├── notebooks/               # Exploratory analysis
├── tests/                   # pytest suite for pipeline and model logic
├── pyproject.toml
└── uv.lock
```

---

## Pipeline

```bash
# 1. Install dependencies
uv sync

# 2. Ingest NFL data (2016–2025)
uv run python -m jets_project.ingest --start-season 2016 --end-season 2025

# 3. Build features (game, team, weekly + injury/cap context)
uv run python -m jets_project.features

# 4. Train models (Logistic Regression + Gradient Boosting)
uv run python -m jets_project.train

# 5. Evaluate (rolling-origin backtests)
uv run python -m jets_project.evaluate

# 6. Launch Streamlit dashboard
uv run streamlit run app/dashboard.py
```

---

## Streamlit Dashboard

Three views:

- **League Overview** — Jets vs. all 32 teams across process metrics (protection, efficiency, explosive plays)
- **Tracking Lens** — Season-by-season trend of key drivers
- **What-if Simulator** — Adjust individual metrics (sack rate, takeaway rate, cap allocation) and see the modelled win-probability impact

---

## Deployment

The app can be deployed to **Streamlit Community Cloud** directly from this repo — no local pipeline run needed. The pre-built artifacts are committed:
- `artifacts/app/jets_app.sqlite` — curated app database
- `artifacts/models/model_bundle.joblib` — trained model bundle

Set Root Directory to `.` in Streamlit Cloud. No additional secrets required.

---

## Notes

- The full working database (`db/jets_nfl.sqlite`) and raw data caches are excluded from version control (large files).
- `nflreadpy` and `lightgbm` are installed automatically via `uv sync`.
- The ingestion layer exits with clear error messages if optional dependencies are missing.
- Detailed write-up in `reports/final_report.md`.
