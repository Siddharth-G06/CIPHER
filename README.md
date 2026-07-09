# CIPHER — Continuous Intelligent Platform for High-risk Event Recognition

> Real-time fraud detection system with graph-based feature engineering,
> ensemble ML, concept drift detection, and SHAP explainability.

## Architecture
[layered diagram here]

## Modules
| Module | Description | Status |
|--------|-------------|--------|
| Data Preprocessing | Temporal split, encoding, validation | ✅ |
| Graph Features | Time-windowed bipartite graph | ✅ |
| Ensemble Model | LightGBM + Isolation Forest | ✅ |
| Drift Detection | ADWIN + PSI + Observer Pattern | ✅ |
| Explainability | TreeSHAP + waterfall plots | ✅ |
| Report Generator | PDF investigation reports with ReportLab | ✅ |
| Kafka Streaming | Event-driven serving layer with 4 topics + DLQ | ✅ |
| ... | ... | ⏳ |

## Sample Output
Every flagged transaction produces a color-coded PDF investigation report containing:
- Risk tier classification (LOW / MEDIUM / HIGH / CRITICAL)
- Ensemble + component risk scores
- SHAP waterfall plot with plain-English explanation
- Full audit trail with model version and drift status
## Setup
## Usage
## Dataset
## Results
## Tech Stack