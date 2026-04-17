# NOC Zoom-Call Prediction — PRD

## Original Problem
Production-grade end-to-end ML system to predict, per NOC incident, whether a Zoom bridge is required — reducing manual toil for engineers triaging ~7k open events in the Oracle Event Manager. Covers ingestion → preprocessing → feature engineering → training → inference → storage → UI, with OCI API handling, PyTorch ANN, business rules, and an Event-Manager-style dashboard.

## User Choices (frozen)
- **1b** Working prototype: FastAPI + React + PyTorch ANN
- **2b** OCI creds to be supplied later → client falls back to Excel mock
- **3a** PyTorch ANN only (binary Zoom Yes/No)
- **4a** UI mirrors the uploaded Oracle "Event Manager → NOC Incidents" screenshot, with added Zoom Call Prediction column + confidence
- **5 yes** Architecture doc at `/app/memory/ARCHITECTURE.md`, rendered in a Docs tab

## Architecture (high level)
8 components: ingestion · preprocessing · feature engineering · model training · inference · storage · UI · automation. Full blueprint with diagrams, PyTorch layer spec, OCI auth/pagination flow, drift monitoring and MLOps loop lives in `/app/memory/ARCHITECTURE.md` and is served at `GET /api/docs/architecture`.

## What's Implemented

### 2026-04-17 (v1.0) — MVP
- Backend FastAPI with 12 `/api` routes for model + predictions + health
- PyTorch ANN (ZoomNet, 56 → 256→128→64→32→1, BCE+pos_weight, AdamW + cosine LR, early stop on PR-AUC)
- Feature engineering (56 features), hidden-truth synthetic generator (mirrors real Excel schema)
- Auto-bootstrap on first boot: seeds 3000 incidents + trains → **PR-AUC 0.84 · F1 0.77**
- Business rules: SEV1 → hard Yes; Yes/No/Review thresholds cost-optimised (FN:FP = 5:1)
- OCI client with Signer auth + `opc-next-page` pagination + retry/429/5xx/401 taxonomy; Excel fallback when no creds
- MongoDB collections: `incidents`, `predictions`, `training_runs`; `_id` stripped in all responses
- Model artefacts at `backend/data_store/models/zoom_net/latest/`
- React Event Manager clone: top bar, ribbon, tabs, 6 KPI cards (added red Zoom-Predicted), 20-column table with Zoom Prediction badges, drawer with feature breakdown, Model tab, Docs tab
- 19/19 backend + frontend critical checks passed

### 2026-04-17 (v1.1) — Full tab completion
- **Customer View** — aggregated by reporter + queue with SEV1-4 pills + Zoom-Yes counts (20 rows)
- **Instance View** — aggregated by region with SEV breakdown + avg age + Zoom-Yes (15 rows, one per region)
- **All Events** — full incident table across all statuses (50-row page)
- **Service Requests** — SEV3/SEV4 customer-impact subset shaped as SRs with SR-id + P3/P4 priority
- **Blackouts** — 12 deterministic synthetic maintenance windows with UPCOMING/ACTIVE/COMPLETED status + impact badge
- **Alarm Lens** — 4 × 15 Severity × Region heat-map (color intensity scales with count, open-count sub-label)
- **Cluster Events** — multi-region clustered incidents with regions-affected pill + zoom prediction badge
- 6 new `/api/views/*` endpoints (customer, instance, alarm-lens, cluster-events, service-requests, blackouts)
- Reusable `SimpleTable`, `Section`, `useFetch` frontend helpers
- 15/15 backend + 7/7 frontend tab tests passed (iteration_2.json)
- OCI partial creds saved (base/tenancy/user/region); client remains in mock mode (missing fingerprint + private key)

## Core Requirements (static)
- All APIs prefixed `/api` and reachable via `REACT_APP_BACKEND_URL`
- MongoDB via `MONGO_URL` + `DB_NAME`

## Prioritised Backlog
**P1**
- [ ] Complete OCI creds (`OCI_FINGERPRINT` + `OCI_PRIVATE_KEY_PEM` or `_PATH`) → flip to live mode
- [ ] Real historical Zoom-label backfill (from `07_commChannels.communicationChannelType == 'ZOOM'`)
- [ ] SHAP explanation values in the drawer (replace static feature list)
- [ ] Threshold slider in UI

**P2**
- [ ] Kafka `noc.events.v1` → Faust streaming path → websocket to UI
- [ ] MLflow run tracking + registry promotion gate
- [ ] Evidently drift dashboard + PagerDuty on PSI > 0.2
- [ ] Real blackout feed (replace deterministic synthetic)

**P3**
- [ ] Multi-tenancy per realm (oc1, oc2, …)
- [ ] Airflow `noc_ml_daily` DAG
- [ ] ONNX export for edge serving
- [ ] Operator override loop → relabelling pipeline

## Key Files
- Architecture: `/app/memory/ARCHITECTURE.md`
- Backend routes: `/app/backend/server.py`
- ML modules: `/app/backend/ml/{synthetic,feature_engineering,model,train,inference,oci_client}.py`
- Frontend: `/app/frontend/src/App.js`
- Data: `/app/backend/data_store/raw/noc_incidents.xlsx`
- Models: `/app/backend/data_store/models/zoom_net/latest/`
- Tests: `/app/backend/tests/test_noc_api.py`, `/app/test_reports/iteration_1.json`, `/app/test_reports/iteration_2.json`

### 2026-04-17 (v1.2) — LLM Explanation column
- Integrated **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) via `emergentintegrations` + Emergent Universal Key
- New MongoDB collection `explanations` (cache keyed by alias)
- Endpoints: `POST /api/explain/{alias}` (compute-or-cache, supports `?force=true`), `GET /api/explain/{alias}` (404 if miss), cached text auto-joined onto `/api/incidents` and `/api/incidents/{alias}` responses
- Backend module: `backend/ml/llm_explain.py` with system prompt capped at 35 words / no bullets / references concrete features
- Frontend: new last column **"LLM Explanation"** — shows ✨ Explain button on uncached rows; on click fires POST and replaces button with a 2-3 line text snippet (tooltip = full text) + model label footer; cached rows render text immediately on next refresh
- 12/12 backend + frontend tests passed (iteration_3.json)
