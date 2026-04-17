# NOC Zoom-Call Prediction — PRD

## Original Problem
Production-grade end-to-end ML system to predict, for each NOC incident, whether a Zoom bridge is required — reducing manual toil for engineers triaging the ~7k open events shown in the Oracle Event Manager. Must cover ingestion → preprocessing → feature engineering → training → inference → storage → UI, with OCI API handling, PyTorch ANN, business rules, and an Event-Manager-style dashboard.

## User Choices (frozen)
- **1b** Working prototype: FastAPI + React + PyTorch ANN
- **2b** OCI creds to be supplied later → OCI client falls back to Excel mock
- **3a** PyTorch ANN only (binary Zoom Yes/No)
- **4a** UI mirrors uploaded Oracle "Event Manager → NOC Incidents" screenshot, with added **Zoom Call Prediction** column + confidence
- **5 yes** Architecture doc at `/app/memory/ARCHITECTURE.md`, rendered in a **Docs** tab

## Architecture (high level)
8 components: ingestion · preprocessing · feature engineering · model training · inference · storage · UI · automation. Full blueprint with diagrams, PyTorch layer spec, OCI auth/pagination flow, drift monitoring and MLOps loop lives in `/app/memory/ARCHITECTURE.md` and is served at `GET /api/docs/architecture`.

## What's Implemented (2026-04-17)
- Backend (FastAPI) with 12 `/api` routes: `/`, `/health`, `/seed`, `/train`, `/predict`, `/predict/batch`, `/incidents`, `/incidents/{alias}`, `/kpis`, `/model`, `/oci/health`, `/oci/pull`, `/docs/architecture`
- **PyTorch ANN (`ZoomNet`)** — 56 inputs → 256→128→64→32→1 (BCEWithLogitsLoss + pos_weight, AdamW + cosine LR, early stopping on PR-AUC)
- Feature engineering: 56 features (severity ordinal, 16-way region one-hot, 6-way status one-hot, 33 numerics: workstream counts, text flags, engagement/people, attachments, CSV-content signals, prior-zoom historical rates)
- Auto-bootstrap on first boot: 3000 synthetic incidents (mirrors real Excel schema with hidden non-linear labelling + 8% noise) → train → write predictions
- Business rules: SEV1 → hard Yes; `Yes` if p ≥ T_high, `No` if p ≤ T_low, else `Review` (threshold cost-optimised on FN:FP = 5:1)
- OCI client with Signer-based auth, `opc-next-page` pagination loop, tenacity-style retry on 429/5xx, auth/timeout taxonomy; Excel fallback when no creds
- MongoDB collections: `incidents`, `predictions`, `training_runs`; all responses strip `_id`
- Model artefacts persisted at `backend/data_store/models/zoom_net/latest/` (model.pt + scaler + encoders + metadata.json)
- Frontend (React 19 + Tailwind) Event Manager clone: top bar, green ribbon, 10 tabs (added **Model** + **Docs**), 6 KPI cards (added red **Zoom Calls Predicted**), actions bar (search, severity, zoom, active-only, Retrain), 20-column table with styled badges, **Zoom Call Prediction** cell (decision + probability + bar + confidence), drawer with key features, Model tab with metrics, Docs tab rendering ARCHITECTURE.md
- Test suite: `/app/backend/tests/test_noc_api.py` + pytest JUnit report — 19/19 backend + all frontend critical checks ✅
- Metrics on seed=42: **PR-AUC 0.84 · ROC-AUC 0.83 · F1 0.77 · Brier 0.17**

## Core Requirements (static)
- All APIs prefixed `/api` and reachable via `REACT_APP_BACKEND_URL`
- MongoDB via `MONGO_URL` + `DB_NAME` — no hardcoding
- Use `emergentintegrations` / Emergent LLM key only if LLM needed (not needed for v1)

## Prioritised Backlog
**P1**
- [ ] Real OCI credentials wired via `OCI_*` env → replace Excel fallback
- [ ] Historical ground-truth Zoom label backfill (from `07_commChannels` where `communicationChannelType == 'ZOOM'`)
- [ ] SHAP explanation values in the drawer (replace static feature list)
- [ ] Threshold slider in UI (operator can tune per shift)

**P2**
- [ ] Streaming path: Kafka `noc.events.v1` → Faust stream processor → websocket push to UI
- [ ] MLflow run tracking + model registry promotion gate
- [ ] Evidently drift dashboard + PagerDuty alerts on PSI > 0.2
- [ ] Role-based access (read-only vs operator override)

**P3**
- [ ] Multi-tenancy per realm (oc1, oc2, …)
- [ ] Auto-retraining Airflow DAG (`noc_ml_daily`)
- [ ] ONNX export for edge serving

## Acceptance Criteria (met for v1)
- End-to-end flow works without manual intervention on first boot ✅
- PR-AUC ≥ 0.80 on validation ✅ (0.84)
- Event Manager UI renders 20 columns with Zoom prediction and drawer ✅
- Docs tab serves the full architecture markdown ✅

## Key Files
- Architecture: `/app/memory/ARCHITECTURE.md`
- Backend: `/app/backend/server.py`, `/app/backend/ml/*`
- Frontend: `/app/frontend/src/App.js`
- Data: `/app/backend/data_store/raw/noc_incidents.xlsx`
- Models: `/app/backend/data_store/models/zoom_net/latest/`
- Tests: `/app/backend/tests/test_noc_api.py`, `/app/test_reports/pytest/pytest_results.xml`
