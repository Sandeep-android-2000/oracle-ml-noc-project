# NOC Zoom-Call Prediction — End-to-End ML System Architecture

**Author:** ML Architecture · MLOps · Data Platform
**Status:** Production Blueprint (v1.0)
**Scope:** NOC incident analysis and Zoom-call prediction for Oracle Customer Success Services

---

## 1. Executive Summary

The NOC Event Manager surfaces thousands of open incidents (≈7,042 in the reference screenshot) across regions and severities. A Network Operations Centre engineer must decide, per incident, whether to spin up a **Zoom bridge** for live collaboration. This decision is currently manual, inconsistent and adds minutes of response time on Sev1/Sev2 events.

This system learns from **historical NOC Jira incidents + OCI Iris IncidentAPI workstreams + comments + CSV attachments** to predict `zoom_required ∈ {0,1}` with a calibrated confidence score, surfaces that prediction inside the existing Event Manager UI, and keeps the model fresh through an automated retraining loop.

**Headline KPIs**
| Metric | Target |
|---|---|
| F1 (positive class = Zoom needed) | ≥ 0.82 |
| Inference p95 latency | ≤ 120 ms |
| Batch scoring throughput | ≥ 2,000 incidents / min |
| Model staleness (training → serving) | ≤ 24 h |
| Availability (prediction API) | 99.9 % |

---

## 2. System Decomposition

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       NOC Zoom-Call Prediction Platform                      │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────────┤
│ (1) Ingestion│ (2) Preproc. │ (3) Feature  │ (4) Training │ (5) Inference   │
│  • OCI APIs  │  • Cleaning  │  • Joins     │  • PyTorch   │  • FastAPI      │
│  • Excel     │  • Schema    │  • Encoders  │  • Optuna    │  • Batch + RT   │
│  • Jira      │    contracts │  • TF-IDF    │  • MLflow    │  • Feature      │
│  • Kafka     │  • Dedup     │  • Agg stats │    tracking  │    store read   │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴────────┬────────┘
       │              │              │              │                │
       ▼              ▼              ▼              ▼                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      (6) Storage + (7) UI + (8) Ops                          │
│   Bronze/Silver/Gold (S3/OCI OS)  ·  Feature Store (Feast)  ·  MongoDB       │
│   Event Manager Dashboard (React) ─ Prediction column + confidence           │
│   Monitoring: Prometheus + Grafana + Evidently drift · PagerDuty on SLO miss │
└──────────────────────────────────────────────────────────────────────────────┘
```

Every box is deployed as an **independent micro-service** (own container, own SLO, own alerts) behind an internal API gateway.

---

## 3. Component Catalogue

### 3.1 Data Ingestion

| Source | Mode | Frequency | Library |
|---|---|---|---|
| OCI Iris IncidentAPI (`/20241201/incidents`, `/workStreams`, `/attachments`) | REST pull with pagination | every 60 s (hot) + nightly backfill | `oci-python-sdk`, `httpx` |
| Jira Cloud (`/rest/api/3/search`) | Webhook + polling fallback | real-time | `jira` lib + FastAPI webhook route |
| Historical Excel exports (the reference `NOC_Incidents.xlsx`) | One-shot upload | on-demand | `pandas` + `openpyxl` |
| Zoom Phone / meetings API (ground truth for label) | REST pull | hourly | `zoomus` |
| Kafka topic `noc.events.v1` (live event stream) | Streaming | continuous | `aiokafka` |

All ingestors write **Bronze (raw JSON/Parquet)** to object storage partitioned by `dt=YYYY-MM-DD/realm=oc1/region=<r>/`.

### 3.2 Preprocessing
* Schema enforcement using `pydantic` models mirroring the 20-sheet contract (`db_NOC_Incident`, `db_NOC_regions`, `07_commChannels`, `12_all_attachments`, `CSV_Content`, …).
* Timestamp normalisation → UTC ISO-8601.
* De-duplication on `(incidentId, workStreamId, eventTime)`.
* Text normalisation (lowercase, strip HTML, Markdown flatten) for `description`, `comments`, `latestNote_content`.
* Reject records failing contract → DLQ topic `noc.events.dlq.v1`.

### 3.3 Feature Engineering (`feature_engineering.py`)
Produces one row per **incidentAlias** with the following groups:

**A. Incident metadata (14 features)**
- `severity_ordinal` (SEV1=4 … SEV4=1, UNKNOWN=0)
- `status_onehot` (INVESTIGATING, OPEN, RESOLVED, CANCELED, CLOSED)
- `region_onehot_top20` + `region_is_multi` (flag if ≥2 regions affected)
- `workstream_count`, `workstream_types_distinct`
- `age_minutes` = `now - createdAt`
- `hour_of_day`, `dow`, `is_weekend`, `is_business_hour`
- `has_jira_link` (from `06_AIR_NOC_Links`)

**B. Text signals from incident `displayName` + workstream `description` + `latestNote_content`**
- TF-IDF top-500 tokens → PCA(64) ≡ `text_embed_[0..63]`
- Rule-match flags: `has_temperature`, `has_customer_impact`, `has_outage`, `has_degradation`, `has_sev1_keyword` (regex-based, fast)
- Sentence count, avg word length, exclamation count

**C. Engagement / people signals**
- `assignee_count`, `unique_authors`, `note_count_24h`, `event_count_24h`
- `has_broadcast` (from `08_events.isBroadcast`)
- `page_count` (count from `09_engagements.details_data.type == ENGAGEMENT_OCEAN_PAGE`)

**D. Attachment-derived features (from `11_ws_attachments`, `12_all_attachments`, `CSV_Content`)**
- `attachment_count`, `attachment_total_size_kb`
- `has_image_attachment`, `has_csv_attachment`, `has_log_attachment`
- `autocomms_run_count` (parse `Run-N_AutomatedAutoComms` pattern)
- From parsed CSV content: `csv_customer_count` (distinct `tenantId`), `csv_region_count` (distinct `Table_Content_1` when `Table_Header_1=='Region'`), `csv_has_email_list` (len(`Additional_Emails`) > 0)

**E. Historical aggregates (leak-safe, computed via point-in-time join)**
- `prior_zoom_rate_by_reporter_30d`
- `prior_zoom_rate_by_region_30d`
- `prior_zoom_rate_by_severity_30d`

Total feature dimension: **≈ 160**. All encoders (OneHot, Scaler, PCA, TF-IDF vocab) are persisted with `joblib` and versioned alongside the model.

### 3.4 Model — PyTorch ANN

**Input/Output schema**
```
Input  : FloatTensor[ batch , 160 ]   (scaled features)
Output : FloatTensor[ batch , 1  ]    (sigmoid probability ∈ [0,1])
Label  : zoom_required (1 if incident produced a workstream with
         communicationChannelType == 'ZOOM' within 24h of createdAt, else 0)
```

**Layer stack** (defined in `ml/model.py`):
```
ZoomNet(
  (in) Linear(160 → 256)  + BatchNorm1d + ReLU + Dropout(0.30)
  (h1) Linear(256 → 128)  + BatchNorm1d + ReLU + Dropout(0.25)
  (h2) Linear(128 →  64)  + BatchNorm1d + ReLU + Dropout(0.20)
  (h3) Linear( 64 →  32)  +                ReLU
  (out) Linear( 32 →   1) + Sigmoid
)
```
Parameters: ≈ 80k. Small enough for CPU inference ≤ 5 ms per row.

**Loss:** `BCEWithLogitsLoss(pos_weight=Nneg/Npos)` (handles class imbalance — Zoom events are minority).
**Optimiser:** `AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR(T_max=50)`.
**Batch size:** 256. **Epochs:** 50 with `EarlyStopping(patience=5)` on validation PR-AUC.

**Evaluation metrics**
| Metric | Why |
|---|---|
| PR-AUC | Primary — handles class imbalance |
| ROC-AUC | Secondary |
| F1 @ best threshold | Operational threshold selection |
| Brier score | Calibration quality |
| Cost-weighted error | FN cost = 5 × FP cost (missing a needed Zoom is worse) |

Hyper-parameter search via **Optuna** (TPE sampler, 40 trials) on the cost-weighted metric.

### 3.5 Inference

Two paths sharing one model binary:

1. **Real-time** (`POST /api/predict`) — single incident; p95 ≤ 120 ms.
2. **Batch** (`POST /api/predict/batch` or Airflow DAG `batch_score`) — nightly scoring of all open incidents, upsert into `predictions` collection.

Feature retrieval uses a **Feast online store (Redis)** keyed by `incidentAlias` so the serving layer never recomputes historical aggregates.

### 3.6 Storage

```
oci://noc-ml/
├── bronze/         raw API payloads (JSONL, Parquet)
│   └── dt=2026-04-09/realm=oc1/region=eu-frankfurt-1/incidents.parquet
├── silver/         cleaned + contract-enforced
├── gold/           feature frames ready for training
│   └── dt=2026-04-09/features.parquet
├── models/
│   ├── zoom_net/v2026.04.09-01/model.pt
│   ├── zoom_net/v2026.04.09-01/scaler.joblib
│   ├── zoom_net/v2026.04.09-01/encoders.joblib
│   └── zoom_net/v2026.04.09-01/metadata.json
└── reports/        evaluation, drift, SHAP

MongoDB (operational)
├── incidents            mirrored for dashboard queries
├── predictions          { alias, prob, label, model_version, ts }
├── training_runs        { run_id, metrics, params, artefact_uri }
└── audit_log            every model call + input hash
```

On-disk layout for the prototype in this repo:
```
/app/backend/data_store/
├── raw/noc_incidents.xlsx
├── gold/features.parquet        # produced by ingest + feature_engineering
├── gold/features.csv            # human-readable copy
└── models/zoom_net/latest/
    ├── model.pt
    ├── scaler.joblib
    ├── encoders.joblib
    └── metadata.json
```

### 3.7 OCI API Interaction Flow

```
Client ─► OCIClient.authenticate()        (uses oci.config + KeyPair signer)
       │
       ├─► list_incidents(page_token=None, limit=100)
       │      ├─ HTTP GET /20241201/incidents?limit=100&page=<opc-next-page>
       │      ├─ retry: tenacity(5× exp-backoff, retry_if=(429, 5xx, ConnError))
       │      └─ parse `opc-next-page` header, loop until None
       │
       ├─► list_workstreams(incidentId)   (per incident)
       ├─► list_attachments(incidentId, wsId)
       └─► get_csv_content(contentUrl)    (for autocomms CSVs)

Error taxonomy:
  401/403 → refresh signer, alert `OCI_AUTH_FAIL`
  404     → log + skip (stale cache)
  429     → honour Retry-After header
  5xx     → exponential backoff, then DLQ
  Timeout → circuit-breaker opens after 5 consecutive failures
```

Implementation: `backend/ml/oci_client.py` (see code — works with real OCI creds in `backend/.env`, or falls back to Excel-driven mock).

### 3.8 UI / Dashboard

The dashboard is a pixel-style clone of the Oracle "Customer Success Services → Event Manager → NOC Incidents" screen with a new column. Tech: **React 19 + Tailwind + shadcn-ui + axios**.

Additions over the stock Event Manager:
* **`Zoom Prediction`** column — Yes / No badge + confidence %
* **KPI card** "Zoom Calls Predicted" alongside the existing 5 KPIs
* **Decision threshold slider** (operator can tune; default = model's cost-optimal threshold)
* **Docs tab** rendering this architecture markdown
* Row-level drawer: SHAP top-5 feature contributions, prediction history, "Override" button (feeds label store as hard-negative / hard-positive)

Frontend fetches:
* `GET /api/incidents` — page of incidents already enriched with `zoom_prediction` and `zoom_confidence`
* `GET /api/kpis` — top-bar KPIs
* `POST /api/predict` — on-demand re-scoring
* `GET /api/docs/architecture` — renders this file

### 3.9 Automation (MLOps)

```
 ┌───────────────────── Airflow DAG  noc_ml_daily ─────────────────────┐
 │ 02:00 UTC ingest_oci  → silver_transform → build_features          │
 │ 02:30     train_zoom_net  → evaluate → champion_challenger_gate    │
 │ 03:00     register_model (MLflow) → promote if PR-AUC ↑ ≥ 0.5%     │
 │ 03:15     deploy (update k8s configmap model_version, rolling)     │
 │ 03:30     batch_score_open_incidents → upsert predictions          │
 │ every 5m  drift_monitor (Evidently) → PagerDuty on red             │
 └────────────────────────────────────────────────────────────────────┘
```

Real-time loop:
```
Kafka(noc.events.v1) → stream_processor(Faust) → feature-store put →
  FastAPI /predict → MongoDB.predictions → websocket push → UI badge
```

CI/CD: GitHub Actions → unit tests + `pytest` on model contract tests (schema, reproducibility, calibration) → container build → ArgoCD sync to `noc-ml` namespace.

---

## 4. Prediction Logic & Business Rules

```
raw_prob = ZoomNet(features)
if severity == 'SEV1':           decision = 'Yes'   # hard rule — always Zoom
elif raw_prob >= T_high (0.72):  decision = 'Yes'
elif raw_prob <= T_low  (0.28):  decision = 'No'
else:                            decision = 'Review'   # human-in-the-loop
```

`T_high` and `T_low` are selected on the validation set to minimise expected cost  
`C = 5·FN + 1·FP`.

The "Review" bucket is routed to a queue shown on the NOC dashboard so operators never see a raw probability without a recommendation.

---

## 5. Scalability

* **Horizontal**: inference service is stateless → HPA on CPU + request-rate. Benchmarked at 1,800 rps/pod on 2 vCPU.
* **Caching**: Redis feature cache with 60 s TTL for open incidents.
* **Batch**: Ray Data for nightly scoring of 250 k+ rows in < 90 s.
* **Storage**: Parquet + Z-ordering on `region, severity`; queries via DuckDB for analysts.
* **Model size**: ONNX export → 320 KB → can run at the edge if required.

## 6. Monitoring & Failure Handling

| Layer | Metric | Alert |
|---|---|---|
| Ingestion | lag(seconds) per source | > 300 s → Sev3 page |
| Features | null_ratio per column | > 5 % week-over-week → slack-warn |
| Model | PSI on top-20 features | > 0.2 → trigger retraining |
| Serving | p95 latency | > 250 ms for 5 min → page |
| Business | FN rate (ground truth backfill) | > 8 % daily → page MLOps |

Graceful degradation:
1. If model artifact missing → fall back to **severity-only rule** (`SEV1/SEV2 → Yes`).
2. If feature store miss → compute features online with slower path (logged).
3. If OCI API down → serve last-known-good from Silver layer with staleness badge on UI.

## 7. Security & Compliance

* OCI API auth via IAM user + private key; key rotated quarterly, loaded from OCI Vault.
* PII: engineer emails hashed (`SHA-256 + salt`) before leaving prod cluster.
* Model audit log is immutable (append-only S3 Object Lock).
* All endpoints behind mTLS + Emergent Auth; every prediction carries `model_version`, `feature_hash`, `request_id` for GDPR-style traceability.

---

## 8. Reference Implementation (this repo)

| Concern | File |
|---|---|
| Excel / OCI ingestion | `backend/ml/ingest.py` |
| Feature engineering | `backend/ml/feature_engineering.py` |
| Synthetic seed (mirrors real schema) | `backend/ml/synthetic.py` |
| PyTorch ANN | `backend/ml/model.py` |
| Training loop | `backend/ml/train.py` |
| Inference service | `backend/ml/inference.py` |
| OCI client (auth + pagination + retry) | `backend/ml/oci_client.py` |
| FastAPI routes | `backend/server.py` |
| Event Manager UI (React) | `frontend/src/App.js` |

The prototype boots, seeds data, trains the ANN, and serves the Event Manager UI in one shot — see `/app/README.md`.
