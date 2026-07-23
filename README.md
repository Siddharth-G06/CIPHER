# CIPHER
### Continuous Intelligent Platform for High-risk Event Recognition

> Real-time fraud detection system built with a 10-module layered architecture —
> graph-based feature engineering, ensemble ML, concept drift detection,
> SHAP explainability, and human-in-the-loop retraining.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green?style=flat-square)
![Kafka](https://img.shields.io/badge/Kafka-KRaft-red?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.8-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-TreeSHAP-purple?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-50%2B%20passing-brightgreen?style=flat-square)

---

## What It Does

CIPHER processes streaming financial transactions through a complete ML pipeline:

- Detects fraud in real time using a supervised + unsupervised ensemble
- Captures card velocity and merchant network patterns via a time-windowed transaction graph
- Monitors its own performance and retrains automatically when concept drift is detected
- Explains every flagged transaction using TreeSHAP and generates a PDF investigation report
- Allows fraud analysts to review flags, submit corrections, and improve the model over time

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYERED ARCHITECTURE                      │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Data Layer  │ Feature Layer│   ML Layer   │   Serving Layer    │
│  loader.py   │preprocess.py │  model.py    │ kafka_producer.py  │
│  schema.py   │graph_feat.py │  trainer.py  │ kafka_consumer.py  │
│  config.yaml │              │drift_detect. │ report_generator.py│
├──────────────┴──────────────┴──────────────┴────────────────────┤
│              Feedback Layer          │    Presentation Layer     │
│  feedback_store.py (SQLite/ES)       │    app.py (Streamlit)     │
│  retrain_pipeline.py                 │    5 tabs, live monitor   │
│  model_update_signal.py              │    analyst review queue   │
└──────────────────────────────────────┴───────────────────────────┘
```

---

## Key Technical Decisions

**Graph Features over pure tabular ML**
Standard row-level models see each transaction in isolation. CIPHER builds a bipartite transaction graph (NetworkX) with time-windowed features — card velocity, merchant degree, amount z-score — capturing fraud network patterns invisible to tabular models.

**Supervised + Unsupervised Ensemble**
LightGBM (0.7 weight) detects known fraud patterns from labeled data. Isolation Forest (0.3 weight) catches novel anomalies without needing labels. Together they cover both known fraud signatures and previously unseen attack vectors.

**Three-Layer Drift Monitoring**
PSI on graph features (early warning) → rolling F1 over 1000-transaction windows (gradual drift) → ADWIN on binary error signal (definitive trigger). Each layer catches a different drift speed: feature-level, gradual, and sudden.

**Champion/Challenger Model Promotion**
Retrained models start in MLflow Staging. Promotion to Production requires AUC-PR improvement of ≥0.005 over the current champion without F1 regression. Every promotion decision is logged with full metrics — audit trail for compliance.

**Human-in-the-Loop with Event Sourcing**
Analyst decisions (CONFIRM\_FRAUD / FALSE\_POSITIVE / ESCALATE) are stored as immutable append-only records in SQLite. Feedback is confidence-weighted during retraining: `weight = (base_size / feedback_size) × analyst_confidence`. The audit trail is permanent and regulatorily compliant.

---

## Module Overview

| # | Module | File | Design Pattern |
|---|--------|------|----------------|
| 1 | Data Preprocessing | `feature_layer/preprocessing.py` | Temporal split, leakage prevention |
| 2 | Graph Features | `feature_layer/graph_features.py` | Bipartite graph, time-windowed |
| 3 | Ensemble Model | `ml_layer/model.py` + `trainer.py` | Strategy Pattern, MLflow tracking |
| 4 | Drift Detection | `ml_layer/drift_detector.py` | Observer Pattern, ADWIN + PSI |
| 5 | SHAP Explainability | `ml_layer/explainer.py` | Facade Pattern, async cache |
| 6 | PDF Reports | `serving_layer/report_generator.py` | Template Method Pattern |
| 7 | Kafka Streaming | `serving_layer/kafka_producer/consumer.py` | Event-driven, at-least-once |
| 8 | Feedback + Retraining | `feedback_layer/retrain_pipeline.py` | Event Sourcing, Champion/Challenger |
| 9 | Dashboard | `app.py` | Streamlit, background threads |
| 10 | Containerization | `docker-compose.yml` + `Dockerfile` | KRaft Kafka, health checks |

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| ML Models | LightGBM, Isolation Forest, scikit-learn |
| Explainability | SHAP (TreeSHAP — exact, not approximate) |
| Drift Detection | river (ADWIN), PSI |
| Experiment Tracking | MLflow (tracking + model registry) |
| Graph Features | NetworkX |
| Streaming | Apache Kafka (KRaft, confluent-kafka) |
| Report Generation | ReportLab Platypus |
| Dashboard | Streamlit, Plotly |
| Data Validation | Pydantic |
| Feedback Storage | SQLite (append-only event sourcing) |
| Containerization | Docker, Docker Compose |
| Testing | pytest (50+ unit tests) |
| Config | PyYAML, python-dotenv |

---

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle)

- 590,000 transactions across 6 months
- ~3.5% fraud rate (severe class imbalance)
- Two files: `train_transaction.csv` + `train_identity.csv`

Download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place in `data/`.

> The dataset is not included in this repository.

---

## Getting Started

### Prerequisites

- Docker Desktop (8GB RAM recommended)
- Python 3.11+
- Kaggle account (for dataset download)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Siddharth-G06/CIPHER.git
cd CIPHER

# 2. Copy environment template
cp .env.example .env

# 3. Place dataset files
# Download from Kaggle and place:
# data/train_transaction.csv
# data/train_identity.csv

# 4. Start infrastructure (Kafka + MLflow + App)
make up

# 5. Run data pipeline
python src/data_layer/pipeline.py

# 6. Train initial model
make train

# 7. Open the dashboard
# Streamlit: http://localhost:8501
# MLflow:    http://localhost:5000
```

### Stream transactions

Once the dashboard is open, click **▶ Start Stream** in the sidebar. Transactions begin flowing through the pipeline within seconds.

### Operational commands

```bash
make up            # start all services
make down          # stop all services
make clean         # stop + delete all data volumes (full reset)
make train         # train model inside container
make test          # run full test suite
make logs          # tail application logs
make kafka-lag     # check consumer group lag
make kafka-topics  # list all Kafka topics
make rebuild       # rebuild app image after code changes
```

---

## Project Structure

```
cipher/
├── config/
│   └── config.yaml              # all parameters — no hardcoded values
├── data/
│   └── processed/               # preprocessed pickles (generated)
├── models/                      # trained model artifacts (generated)
├── src/
│   ├── data_layer/              # loader, schema, Pydantic validation
│   ├── feature_layer/           # preprocessing, graph features
│   ├── ml_layer/                # model, trainer, drift, explainer
│   ├── serving_layer/           # Kafka producer/consumer, report generator
│   ├── feedback_layer/          # feedback store, retraining pipeline
│   └── utils/                   # logger, config loader, MLflow client
├── tests/                       # 10 test files, 50+ unit tests
├── app.py                       # Streamlit dashboard entry point
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Design Principles

- **Config-driven** — all parameters in `config/config.yaml`, read via `utils/config_loader.py`
- **Structured logging** — `utils/logger.py` used across all modules, no bare `print()` statements
- **Type hints** — every function signature annotated
- **Google-style docstrings** — every class and method documented
- **Unit tested** — 50+ tests covering temporal leakage, SHAP efficiency axiom, drift detection, Champion/Challenger logic, and Kafka offset handling

---

## Status

| Component | Status |
|-----------|--------|
| Data preprocessing + graph features | ✅ Complete |
| Ensemble model + MLflow tracking | ✅ Complete |
| Drift detection (ADWIN + PSI) | ✅ Complete |
| SHAP explainability + PDF reports | ✅ Complete |
| Kafka streaming (producer + consumer) | ✅ Complete |
| Human feedback + retraining pipeline | ✅ Complete |
| Streamlit dashboard (5 tabs) | ✅ Complete |
| Docker Compose + Makefile | ✅ Complete |
| Full end-to-end integration | 🔄 In progress |

---

## Author

**Siddharth** — B.Tech + M.Tech (Dual Degree), SNU + IIT Madras
[GitHub](https://github.com/Siddharth-G06)

---

*Built end-to-end as a production-grade ML engineering portfolio project. June – July 2025.*