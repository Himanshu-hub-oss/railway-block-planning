# AI Railway Maintenance Block & Disconnection Optimization System

An **SIH Hackathon Judge-Ready Railway Maintenance Decision Support & Block Coordination Platform** designed to unify multi-department maintenance disconnection planning for **Engineering (Track)**, **Traction Distribution (OHE)**, and **Signal & Telecommunication (S&T)** on Indian Railways.

---

## 🌟 Architecture & System Overview

```
SIH/
│
├── data/
│   └── raw/                  # Real reference datasets (stations.json, trains.json, schedules.json)
│
├── src/
│   ├── data_processing.py      # Station network indexing, train density analytics & Judge Demo scenario generator
│   ├── train_impact.py         # Train schedule impact engine (direct/near-window conflicts, delay estimates)
│   ├── feature_engineering.py  # ML feature matrix & complexity index calculations
│   ├── ml_models.py            # Random Forest & Gradient Boosting duration regression + disruption risk models
│   ├── conflict_detection.py   # Multi-department spatial-temporal conflict engine & conflict heatmap matrix
│   ├── optimization.py         # Google OR-Tools CP-SAT constraint programming solver engine
│   ├── recommendation_engine.py # Explainable AI recommendation engine & Explain This Decision modal data
│   ├── predictive_priority.py  # Maintenance Priority Scoring Engine (0-100 score & risk levels)
│   ├── emergency_mode.py       # Fast-path Emergency Maintenance Block engine (🚨 EMERGENCY BLOCK)
│   └── report_generator.py     # 12-Section Executive ReportLab PDF exporter
│
├── reports/                    # Generated PDF reports
├── tests/                      # Pytest verification suite
├── app.py                      # Interactive Railway Operations Control Center Streamlit UI
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## 🎯 15 Key Capabilities & Features

1. **Executive Command Center**: Professional dark control-room UI with prominent `AI PLANNING ENGINE: ONLINE` banner, 7 operational KPIs, and live-style operational timeline.
2. **Realistic Maintenance Request Workflow**: Dynamic 12-field submission form with instant validation, schedule check, risk scoring, and optimizer re-run.
3. **Train Impact Engine**: Queries 417,000+ real train schedules to identify direct conflicts, near-window conflicts, delay estimates, and impact levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
4. **AI Block Recommendation Engine**: Generates coordinated joint shadow blocks with confidence scores, alternative windows, and expected time savings.
5. **Conflict Matrix & Heatmap**: Dynamic cross-department matrix (`Engineering` x `Traction` x `S&T`) and "WHY NOT COMBINE?" incompatibility explanations.
6. **Railway Network Map**: Visualizes station GPS coordinates from `stations.json` with department activity overlays and conflict markers.
7. **Emergency Maintenance Mode (`🚨 EMERGENCY BLOCK`)**: Fast-path for unexpected track/power failures finding the least-conflicting off-peak maintenance window.
8. **Predictive Maintenance Priority**: Multi-criteria priority scoring engine (0-100 score) evaluating asset condition, safety criticality, and maintenance backlog.
9. **What-If Simulation Lab**: Parameter sliders (train density multiplier, max block duration) comparing baseline vs simulated scenario performance.
10. **Before vs After Operational Impact**: Dynamic comparison matrix showing quantifiable reductions in line disconnections, total block hours, and department conflicts.
11. **Digital Approval Workflow**: 7-stage operational approval pipeline (`Request` → `AI Analysis` → `Conflict Check` → `Train Impact` → `Optimizer` → `Review` → `Approval`).
12. **AI Explainability ("Explain This Decision")**: Structured explanations detailing decision, rationale, constraints enforced, and expected impact.
13. **Data Quality & Model Transparency**: Clear disclosures separating `REAL DATA` (stations, trains, schedules) from `SIMULATED SCENARIO DATA` (maintenance requests, failure events).
14. **Model Monitoring**: ML evaluation metrics (MAE, RMSE, R², Accuracy, F1, Confusion Matrix, Feature Importances).
15. **12-Section Executive PDF Report Generator**: One-click ReportLab PDF report generation covering executive summary, conflict analysis, train impact, and final coordinated block schedule.

---

## ⭐ Deterministic Judge Demo Workflow

Click the **`⭐ Load Judge Demo Scenario`** button in the sidebar to execute a 100% deterministic demo containing:
- 1 Engineering request (Deep Screening BCM)
- 1 Traction request (OHE Periodic Overhauling)
- 1 S&T / Track request (Turnout Sleepers Replacement with machine collision)
- Real scheduled train traffic on section
- 1 spatial/temporal conflict
- 1 compatible shadow block combination
- 1 high-priority emergency request

The complete operational flow runs in 30 seconds:
`REQUESTS` → `AI PREDICTION` → `CONFLICT DETECTION` → `TRAIN IMPACT` → `OPTIMIZATION` → `JOINT BLOCK` → `EXPLANATION` → `BEFORE VS AFTER` → `PDF REPORT`

---

## 🛡️ Data Honesty & Transparency Disclosure

- **REAL REFERENCE DATA**: Stations (`stations.json`), Trains (`trains.json`), Schedules (`schedules.json`).
- **SIMULATED SCENARIO DATA**: Maintenance requests, emergency failure events, estimated train delays, and asset failure history.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Control Center Dashboard
```bash
streamlit run app.py
```
Open browser at `http://localhost:8501`.
