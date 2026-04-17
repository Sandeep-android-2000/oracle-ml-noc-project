# NOC Zoom-Call Prediction — Local Run Guide

End-to-end ML system that predicts whether a Zoom call is required for each NOC incident.
Stack: **FastAPI · PyTorch ANN · MongoDB · React (CRA + Tailwind) · Claude Haiku 4.5** (via Emergent Universal Key).

---

## 0. Prerequisites

| Tool | Version | Install hint |
|---|---|---|
| Python | 3.11 | `pyenv` / system package |
| Node.js | 18 or 20 | [nodejs.org](https://nodejs.org) |
| Yarn | 1.22+ | `npm i -g yarn` |
| MongoDB | 5.x or 6.x | `brew install mongodb-community` / [docker](https://hub.docker.com/_/mongo) |
| git | any | — |

Quick MongoDB via Docker:
```bash
docker run -d --name mongo -p 27017:27017 mongo:6
```

---

## 1. Clone & layout

```
your-workdir/
├── backend/      FastAPI + PyTorch + OCI client
│   ├── ml/       model, features, training, inference, OCI, LLM
│   ├── data_store/
│   │   ├── raw/noc_incidents.xlsx     (sample Excel fallback)
│   │   └── models/zoom_net/latest/    (model artefacts, generated on first boot)
│   ├── server.py
│   ├── requirements.txt
│   └── .env
└── frontend/     React Event Manager clone
    ├── src/App.js
    ├── package.json
    └── .env
```

---

## 2. Backend setup

```bash
cd backend

# create & activate venv
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install deps (torch is the heaviest ~600 MB; pick CPU wheel on non-GPU machines)
pip install -r requirements.txt
```

Create `backend/.env`:

```ini
MONGO_URL="mongodb://localhost:27017"
DB_NAME="noc_zoom_predictor"
CORS_ORIGINS="*"

# Emergent Universal Key (required for the LLM Explanation column). Ask Emergent for your own,
# or sign up at https://app.emergent.sh and paste the key here.
EMERGENT_LLM_KEY="sk-emergent-XXXXXXXXXXXXXXXXXXXXXXXX"

# --- Optional: OCI live mode. Leave empty to stay on Excel mock. ---
OCI_INCIDENT_API_BASE=""
OCI_TENANCY_OCID=""
OCI_USER_OCID=""
OCI_FINGERPRINT=""
OCI_REGION="us-phoenix-1"
OCI_PRIVATE_KEY_PATH=""           # absolute path to the .pem
OCI_PRIVATE_KEY_PEM=""            # OR paste PEM contents here
```

Start the API:

```bash
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

First-boot bootstrap (automatic): seeds 3000 synthetic incidents → trains ZoomNet → writes predictions. Logs show `PR-AUC ~0.84, F1 ~0.77`. Takes ~30-60 s on a laptop CPU.

Sanity check:
```bash
curl http://localhost:8001/api/health
curl http://localhost:8001/api/kpis
```

---

## 3. Frontend setup

```bash
cd ../frontend
yarn install
```

Create `frontend/.env`:

```ini
REACT_APP_BACKEND_URL=http://localhost:8001
WDS_SOCKET_PORT=0
```

Run dev server:

```bash
yarn start
```

Open **http://localhost:3000**. You should see the Event Manager UI with all 10 tabs populated.

---

## 4. What each tab shows

| Tab | Source | Content |
|---|---|---|
| **Customer View** | `GET /api/views/customer` | Per-reporter / queue aggregate with SEV breakdown + Zoom-Yes count |
| **Instance View** | `GET /api/views/instance` | Per-region aggregate with avg age + Zoom-Yes |
| **All Events** | `GET /api/incidents` | Full 21-column incident table across all statuses |
| **Service Requests** | `GET /api/views/service-requests` | SEV3/SEV4 customer-impact tickets (P3/P4) |
| **Blackouts** | `GET /api/views/blackouts` | Synthetic scheduled maintenance windows |
| **NOC Incidents** | `GET /api/incidents` | Main table with **Zoom Prediction** + **LLM Explanation** columns |
| **Alarm Lens** | `GET /api/views/alarm-lens` | Severity × Region heat-map |
| **Cluster Events** | `GET /api/views/cluster-events` | Multi-region clustered events |
| **Model** | `GET /api/model` | Training metrics + feature list + Retrain button |
| **Docs** | `GET /api/docs/architecture` | Full system architecture markdown |

---

## 5. Useful backend endpoints

```
GET  /api/health                    mongo + model + oci status
GET  /api/kpis                      6 dashboard KPI numbers
GET  /api/incidents?page=1          paged list with predictions + explanation cache
GET  /api/incidents/{alias}         single incident
POST /api/predict                   score one payload
POST /api/predict/batch             score an array
POST /api/seed {n, retrain}         regenerate synthetic dataset + retrain
POST /api/train {epochs}            retrain on stored data
POST /api/explain/{alias}           LLM explanation (cached). ?force=true to regenerate
GET  /api/explain/{alias}           cached LLM explanation (404 if miss)
GET  /api/oci/health                OCI client mode (live / mock)
POST /api/oci/pull?limit=N          paginated live OCI pull (only if creds valid)
GET  /api/docs/architecture         raw ARCHITECTURE.md
```

Full blueprint lives at `/memory/ARCHITECTURE.md` (also rendered under the **Docs** tab).

---

## 6. Running without internet / no keys

* **No Emergent LLM key** → leave `EMERGENT_LLM_KEY` empty. Everything works *except* the ✨ Explain buttons (they'll return 502). Remove the column by editing `HEAD` in `frontend/src/App.js` if desired.
* **No OCI credentials** → leave all `OCI_*` blank. Client falls back to the Excel in `backend/data_store/raw/`.
* **No MongoDB** → you must have one; the app is stateful. Use the Docker one-liner above.

---

## 7. Common issues

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: torch` | `pip install torch` — on Apple Silicon: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Backend 500 on `/api/explain/*` | `EMERGENT_LLM_KEY` not set, or out of credit |
| Frontend can't reach backend | `REACT_APP_BACKEND_URL` in `frontend/.env` must match the backend port; don't forget to restart `yarn start` after editing |
| Model artefacts missing | `POST /api/train` or just restart backend — first-boot bootstrap recreates them |
| MongoDB "Connection refused" | Mongo isn't running. `docker start mongo` or `brew services start mongodb-community` |

---

## 8. Running the test suite

```bash
cd backend
source .venv/bin/activate
pytest tests/ -v
```

Reports land under `/test_reports/`. There are three historical iteration reports (`iteration_1.json` … `iteration_3.json`) from the build runs.

---

## 9. Tear-down

```bash
# stop frontend / backend: Ctrl-C in each terminal
# clear local data:
mongosh noc_zoom_predictor --eval "db.dropDatabase()"
rm -rf backend/data_store/models
```

That's it. Happy NOC-ing.
