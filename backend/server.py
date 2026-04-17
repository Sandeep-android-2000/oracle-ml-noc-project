"""NOC Zoom-Call Prediction API.

Routes are mounted under ``/api``:
  GET  /api/                 service banner
  GET  /api/health           liveness + model + oci status
  POST /api/seed             generate synthetic incidents into Mongo and retrain
  POST /api/train            retrain on whatever is stored (uses synthetic if empty)
  POST /api/predict          score one incident payload on-the-fly
  POST /api/predict/batch    score a list of incidents
  GET  /api/incidents        paged list of incidents enriched with predictions
  GET  /api/incidents/{id}   single incident with full feature + prediction detail
  GET  /api/kpis             dashboard KPI card values
  GET  /api/model            current model metadata (metrics, threshold, ...)
  GET  /api/oci/health       OCI client mode + connectivity
  POST /api/oci/pull         (live-only) pull a page of OCI incidents
  GET  /api/docs/architecture return the ARCHITECTURE.md markdown
"""
from __future__ import annotations
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from ml.inference import ENGINE
from ml.oci_client import CLIENT as OCI_CLIENT
from ml.synthetic import generate_incidents
from ml.train import train as train_model

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
ARCH_DOC_PATH = Path("/app/memory/ARCHITECTURE.md")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("noc.server")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
INCIDENTS = db.incidents
PREDICTIONS = db.predictions
RUNS = db.training_runs

app = FastAPI(title="NOC Zoom-Call Prediction API", version="1.0.0")
api = APIRouter(prefix="/api")


# ------------------------------ Pydantic models ------------------------------

class IncidentIn(BaseModel):
    alias: str
    severity: str = "SEV3"
    status: str = "INVESTIGATING"
    region: str = "us-ashburn-1"
    title: str = ""
    # Everything else is optional; synthetic.generate_incidents fields allowed.
    regions_affected: int = 1
    multi_region: int = 0
    workstream_count: int = 1
    workstream_types_distinct: int = 1
    age_minutes: int = 60
    hour_of_day: int = 12
    dow: int = 1
    is_weekend: int = 0
    is_business_hour: int = 1
    has_jira_link: int = 1
    desc_len: int = 100
    has_customer_impact: int = 0
    has_outage: int = 0
    has_temperature: int = 0
    has_sev1_keyword: int = 0
    assignee_count: int = 1
    unique_authors: int = 1
    note_count_24h: int = 0
    event_count_24h: int = 0
    has_broadcast: int = 0
    page_count: int = 0
    attachment_count: int = 0
    attachment_total_kb: int = 0
    has_image_attachment: int = 0
    has_csv_attachment: int = 0
    has_log_attachment: int = 0
    autocomms_run_count: int = 0
    csv_customer_count: int = 0
    csv_region_count: int = 0
    csv_has_email_list: int = 0
    prior_zoom_rate_reporter_30d: float = 0.1
    prior_zoom_rate_region_30d: float = 0.1
    prior_zoom_rate_severity_30d: float = 0.2


class PredictBatchIn(BaseModel):
    incidents: List[IncidentIn]


class SeedIn(BaseModel):
    n: int = Field(3000, ge=50, le=20000)
    seed: int = 42
    retrain: bool = True


class TrainIn(BaseModel):
    epochs: int = Field(40, ge=1, le=200)
    n_synthetic: int = Field(3000, ge=50, le=20000)
    seed: int = 42


# ------------------------------ helpers ------------------------------

def _strip_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc.pop("_id", None)
    return doc


async def _all_incidents(limit: int = 0) -> List[Dict[str, Any]]:
    cursor = INCIDENTS.find({}, {"_id": 0})
    if limit:
        cursor = cursor.limit(limit)
    return await cursor.to_list(length=None)


async def _persist_incidents(rows: List[Dict[str, Any]]):
    if not rows:
        return
    await INCIDENTS.delete_many({})
    # Mongo mutates dicts on insert_many (adds _id); work on copies.
    await INCIDENTS.insert_many([dict(r) for r in rows])


async def _persist_predictions(preds: List[Dict[str, Any]]):
    if not preds:
        return
    now = datetime.now(timezone.utc).isoformat()
    await PREDICTIONS.delete_many({})
    docs = [{**p, "ts": now} for p in preds]
    await PREDICTIONS.insert_many(docs)


async def _score_all_stored():
    rows = await _all_incidents()
    preds = ENGINE.predict_many(rows)
    await _persist_predictions(preds)
    return preds


# ------------------------------ routes ------------------------------

@api.get("/")
async def root():
    return {
        "service": "noc-zoom-predictor",
        "version": "1.0.0",
        "model_loaded": ENGINE.is_ready(),
        "docs": "/api/docs/architecture",
    }


@api.get("/health")
async def health():
    inc_count = await INCIDENTS.count_documents({})
    pred_count = await PREDICTIONS.count_documents({})
    return {
        "status": "ok",
        "mongo": {"incidents": inc_count, "predictions": pred_count,
                  "db": DB_NAME},
        "model": {
            "loaded": ENGINE.is_ready(),
            "metadata": ENGINE.metadata if ENGINE.is_ready() else None,
        },
        "oci": OCI_CLIENT.health(),
    }


@api.post("/seed")
async def seed(body: SeedIn):
    rows = generate_incidents(n=body.n, seed=body.seed)
    await _persist_incidents(rows)
    result: Dict[str, Any] = {"seeded": len(rows), "retrained": False}
    if body.retrain:
        tr = train_model(rows=rows, epochs=30, seed=body.seed)
        ENGINE.load()
        preds = await _score_all_stored()
        await RUNS.insert_one({**tr.__dict__,
                               "ts": datetime.now(timezone.utc).isoformat()})
        result.update(retrained=True, metrics={
            "pr_auc": tr.best_val_prauc, "roc_auc": tr.val_roc_auc,
            "f1": tr.val_f1, "brier": tr.val_brier,
            "positive_rate": tr.positive_rate, "threshold": tr.threshold,
            "feature_count": tr.feature_count, "epochs": tr.epochs,
        }, predictions_written=len(preds))
    return result


@api.post("/train")
async def train(body: TrainIn):
    stored = await _all_incidents()
    rows = stored if stored else generate_incidents(n=body.n_synthetic, seed=body.seed)
    if not stored:
        await _persist_incidents(rows)
    tr = train_model(rows=rows, epochs=body.epochs, seed=body.seed)
    ENGINE.load()
    preds = await _score_all_stored()
    await RUNS.insert_one({**tr.__dict__, "ts": datetime.now(timezone.utc).isoformat()})
    return {
        "trained_on": len(rows),
        "metrics": {"pr_auc": tr.best_val_prauc, "roc_auc": tr.val_roc_auc,
                    "f1": tr.val_f1, "brier": tr.val_brier,
                    "positive_rate": tr.positive_rate,
                    "threshold": tr.threshold, "feature_count": tr.feature_count,
                    "epochs": tr.epochs},
        "predictions_written": len(preds),
    }


@api.post("/predict")
async def predict(row: IncidentIn):
    preds = ENGINE.predict_many([row.model_dump()])
    return preds[0] if preds else {"error": "no prediction"}


@api.post("/predict/batch")
async def predict_batch(body: PredictBatchIn):
    preds = ENGINE.predict_many([r.model_dump() for r in body.incidents])
    return {"count": len(preds), "predictions": preds}


@api.get("/incidents")
async def list_incidents(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    region: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    active_only: bool = False,
    zoom: Optional[str] = Query(None, regex="^(Yes|No|Review)?$"),
    search: Optional[str] = None,
):
    q: Dict[str, Any] = {}
    if region:
        q["region"] = region
    if severity:
        q["severity"] = severity
    if status:
        q["status"] = status
    if active_only:
        q["active_noc"] = 1
    if search:
        q["$or"] = [
            {"alias": {"$regex": search, "$options": "i"}},
            {"jira_id": {"$regex": search, "$options": "i"}},
            {"title": {"$regex": search, "$options": "i"}},
        ]

    total = await INCIDENTS.count_documents(q)
    rows = await (INCIDENTS.find(q, {"_id": 0})
                  .sort("last_updated_min", 1)
                  .skip((page - 1) * page_size)
                  .limit(page_size)
                  .to_list(length=page_size))
    # Join predictions
    aliases = [r["alias"] for r in rows]
    pred_map: Dict[str, Dict[str, Any]] = {}
    async for p in PREDICTIONS.find({"alias": {"$in": aliases}}, {"_id": 0}):
        pred_map[p["alias"]] = p
    # If a zoom filter is active, apply after join
    enriched = []
    for r in rows:
        p = pred_map.get(r["alias"])
        if p is None:
            p = ENGINE.predict_many([r])[0]
            pred_map[r["alias"]] = p
        r["prediction"] = p
        enriched.append(r)
    if zoom:
        enriched = [e for e in enriched if e["prediction"]["decision"] == zoom]
    return {"total": total, "page": page, "page_size": page_size, "rows": enriched}


@api.get("/incidents/{alias}")
async def get_incident(alias: str):
    row = await INCIDENTS.find_one({"alias": alias}, {"_id": 0})
    if not row:
        raise HTTPException(404, "incident not found")
    pred = await PREDICTIONS.find_one({"alias": alias}, {"_id": 0})
    if not pred:
        pred = ENGINE.predict_many([row])[0]
    row["prediction"] = pred
    return row


# ----------------------------- aggregation views -----------------------------

async def _count(q):
    return await INCIDENTS.count_documents(q)


@api.get("/views/customer")
async def view_customer():
    """Aggregated view by reporter / queue."""
    pipeline = [
        {"$group": {
            "_id": {"reporter": "$reporter_name", "queue": "$queue_name"},
            "total": {"$sum": 1},
            "open": {"$sum": "$active_noc"},
            "sev1": {"$sum": {"$cond": [{"$eq": ["$severity", "SEV1"]}, 1, 0]}},
            "sev2": {"$sum": {"$cond": [{"$eq": ["$severity", "SEV2"]}, 1, 0]}},
            "sev3": {"$sum": {"$cond": [{"$eq": ["$severity", "SEV3"]}, 1, 0]}},
            "sev4": {"$sum": {"$cond": [{"$eq": ["$severity", "SEV4"]}, 1, 0]}},
            "regions": {"$addToSet": "$region"},
        }},
        {"$project": {
            "_id": 0,
            "reporter": "$_id.reporter",
            "queue": "$_id.queue",
            "total": 1, "open": 1,
            "sev1": 1, "sev2": 1, "sev3": 1, "sev4": 1,
            "regions": {"$size": "$regions"},
        }},
        {"$sort": {"open": -1, "total": -1}},
    ]
    rows = await INCIDENTS.aggregate(pipeline).to_list(length=None)
    # Join zoom-yes counts
    zoom_pipeline = [
        {"$lookup": {"from": "predictions", "localField": "alias",
                     "foreignField": "alias", "as": "p"}},
        {"$unwind": "$p"},
        {"$match": {"p.decision": "Yes"}},
        {"$group": {
            "_id": {"reporter": "$reporter_name", "queue": "$queue_name"},
            "zoom_yes": {"$sum": 1},
        }},
    ]
    zoom_map: Dict[tuple, int] = {}
    async for z in INCIDENTS.aggregate(zoom_pipeline):
        zoom_map[(z["_id"]["reporter"], z["_id"]["queue"])] = z["zoom_yes"]
    for r in rows:
        r["zoom_yes"] = zoom_map.get((r["reporter"], r["queue"]), 0)
    return {"total": len(rows), "rows": rows}


@api.get("/views/instance")
async def view_instance():
    """Aggregated view by region (an "instance" ≈ regional plane)."""
    pipeline = [
        {"$group": {
            "_id": "$region",
            "total": {"$sum": 1},
            "open": {"$sum": "$active_noc"},
            "sev1": {"$sum": {"$cond": [{"$eq": ["$severity", "SEV1"]}, 1, 0]}},
            "sev2": {"$sum": {"$cond": [{"$eq": ["$severity", "SEV2"]}, 1, 0]}},
            "sev3": {"$sum": {"$cond": [{"$eq": ["$severity", "SEV3"]}, 1, 0]}},
            "sev4": {"$sum": {"$cond": [{"$eq": ["$severity", "SEV4"]}, 1, 0]}},
            "multi_region": {"$sum": "$multi_region"},
            "avg_age_min": {"$avg": "$age_minutes"},
        }},
        {"$project": {
            "_id": 0, "region": "$_id",
            "total": 1, "open": 1,
            "sev1": 1, "sev2": 1, "sev3": 1, "sev4": 1,
            "multi_region": 1,
            "avg_age_min": {"$round": ["$avg_age_min", 0]},
        }},
        {"$sort": {"open": -1}},
    ]
    rows = await INCIDENTS.aggregate(pipeline).to_list(length=None)
    zoom_pipeline = [
        {"$lookup": {"from": "predictions", "localField": "alias",
                     "foreignField": "alias", "as": "p"}},
        {"$unwind": "$p"},
        {"$match": {"p.decision": "Yes"}},
        {"$group": {"_id": "$region", "zoom_yes": {"$sum": 1}}},
    ]
    zoom_map: Dict[str, int] = {}
    async for z in INCIDENTS.aggregate(zoom_pipeline):
        zoom_map[z["_id"]] = z["zoom_yes"]
    for r in rows:
        r["zoom_yes"] = zoom_map.get(r["region"], 0)
    return {"total": len(rows), "rows": rows}


@api.get("/views/alarm-lens")
async def view_alarm_lens():
    """Severity × Region heat-map matrix for the alarm lens tab."""
    pipeline = [
        {"$group": {
            "_id": {"sev": "$severity", "region": "$region"},
            "count": {"$sum": 1},
            "open": {"$sum": "$active_noc"},
        }},
    ]
    matrix: Dict[str, Dict[str, Dict[str, int]]] = {}
    regions_set: set = set()
    async for r in INCIDENTS.aggregate(pipeline):
        sev = r["_id"]["sev"]
        reg = r["_id"]["region"]
        regions_set.add(reg)
        matrix.setdefault(sev, {})[reg] = {"count": r["count"], "open": r["open"]}
    regions = sorted(regions_set)
    severities = ["SEV1", "SEV2", "SEV3", "SEV4"]
    return {
        "regions": regions,
        "severities": severities,
        "matrix": matrix,
    }


@api.get("/views/cluster-events")
async def view_cluster_events(page: int = Query(1, ge=1),
                              page_size: int = Query(25, ge=1, le=200)):
    """Multi-region clustered NOC events (multi_region==1)."""
    q = {"multi_region": 1}
    total = await INCIDENTS.count_documents(q)
    rows = await (INCIDENTS.find(q, {"_id": 0})
                  .sort("regions_affected", -1)
                  .skip((page - 1) * page_size)
                  .limit(page_size)
                  .to_list(length=page_size))
    aliases = [r["alias"] for r in rows]
    pred_map: Dict[str, Dict] = {}
    async for p in PREDICTIONS.find({"alias": {"$in": aliases}}, {"_id": 0}):
        pred_map[p["alias"]] = p
    for r in rows:
        r["prediction"] = pred_map.get(r["alias"])
    return {"total": total, "page": page, "page_size": page_size, "rows": rows}


@api.get("/views/service-requests")
async def view_service_requests(page: int = Query(1, ge=1),
                                page_size: int = Query(25, ge=1, le=200)):
    """Lower-severity, customer-impact incidents treated as Service Requests."""
    q = {"severity": {"$in": ["SEV3", "SEV4"]},
         "$or": [{"has_customer_impact": 1}, {"queue_name": "DBOps"}]}
    total = await INCIDENTS.count_documents(q)
    rows = await (INCIDENTS.find(q, {"_id": 0})
                  .sort("open_since_min", 1)
                  .skip((page - 1) * page_size)
                  .limit(page_size)
                  .to_list(length=page_size))
    # Service-request-shaped projection
    for r in rows:
        r["sr_id"] = f"SR-{3000000 + abs(hash(r['alias'])) % 999999:06d}"
        r["priority"] = "P3" if r["severity"] == "SEV3" else "P4"
    return {"total": total, "page": page, "page_size": page_size, "rows": rows}


@api.get("/views/blackouts")
async def view_blackouts():
    """Synthetic scheduled maintenance blackouts (until a real feed is wired)."""
    import hashlib
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    regions = await INCIDENTS.distinct("region")
    rows: List[Dict[str, Any]] = []
    for idx, region in enumerate(sorted(regions)[:12]):
        seed = int(hashlib.md5(region.encode()).hexdigest(), 16)
        start_offset_h = (seed % 48) + 2
        duration_h = 2 + (seed % 6)
        start = now + timedelta(hours=start_offset_h - 24 * (idx % 3))
        end = start + timedelta(hours=duration_h)
        rows.append({
            "id": f"BO-{idx + 1:04d}",
            "region": region,
            "description": f"Planned maintenance window for {region}",
            "status": "ACTIVE" if start <= now <= end else
                      ("UPCOMING" if start > now else "COMPLETED"),
            "window_start": start.strftime("%Y-%m-%d %H:%M UTC"),
            "window_end": end.strftime("%Y-%m-%d %H:%M UTC"),
            "duration_hours": duration_h,
            "owner": ["NetworkOps", "DBOps", "SRE", "COE"][seed % 4],
            "impact": ["Low", "Medium", "High"][seed % 3],
        })
    rows.sort(key=lambda r: r["window_start"])
    return {"total": len(rows), "rows": rows}


# -----------------------------------------------------------------------------


@api.get("/kpis")
async def kpis():
    total_open = await INCIDENTS.count_documents({"active_noc": 1})
    total_incidents = await INCIDENTS.count_documents({})
    regions = await INCIDENTS.distinct("region", {"active_noc": 1})
    multi_region = await INCIDENTS.count_documents({"active_noc": 1, "multi_region": 1})
    zoom_yes = await PREDICTIONS.count_documents({"decision": "Yes"})
    zoom_review = await PREDICTIONS.count_documents({"decision": "Review"})
    return {
        "total_open_events": total_open,
        "total_open_nocs": total_open,
        "impacted_regions": len(regions),
        "multi_region_nocs": multi_region,
        "impacted_customers": 0,
        "total_incidents": total_incidents,
        "zoom_calls_predicted": zoom_yes,
        "zoom_calls_review": zoom_review,
    }


@api.get("/model")
async def model_info():
    if not ENGINE.is_ready():
        ENGINE.load()
    if not ENGINE.is_ready():
        return {"loaded": False}
    return {"loaded": True, "metadata": ENGINE.metadata}


@api.get("/oci/health")
async def oci_health():
    return OCI_CLIENT.health()


@api.post("/oci/pull")
async def oci_pull(limit: int = 100):
    try:
        items = OCI_CLIENT.list_incidents(limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"OCI pull failed: {exc}")
    return {"mode": OCI_CLIENT.mode, "count": len(items), "sample": items[:5]}


@api.get("/docs/architecture", response_class=PlainTextResponse)
async def get_architecture():
    if not ARCH_DOC_PATH.exists():
        raise HTTPException(404, "architecture doc not found")
    return ARCH_DOC_PATH.read_text()


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    # Attempt to load model; if nothing persisted yet, seed+train so the UI
    # is immediately useful on first boot.
    ENGINE.load()
    inc_count = await INCIDENTS.count_documents({})
    if inc_count == 0:
        logger.info("First-boot bootstrap: seeding + training")
        rows = generate_incidents(n=3000, seed=42)
        await _persist_incidents(rows)
        tr = train_model(rows=rows, epochs=25, seed=42)
        ENGINE.load()
        preds = ENGINE.predict_many(rows)
        await _persist_predictions(preds)
        await RUNS.insert_one({**tr.__dict__,
                               "ts": datetime.now(timezone.utc).isoformat()})
        logger.info("Bootstrap complete: metrics=%s", tr.__dict__)
    elif not ENGINE.is_ready():
        logger.info("Incidents present but no model -- training")
        rows = await _all_incidents()
        tr = train_model(rows=rows, epochs=25, seed=42)
        ENGINE.load()
        preds = ENGINE.predict_many(rows)
        await _persist_predictions(preds)
        await RUNS.insert_one({**tr.__dict__,
                               "ts": datetime.now(timezone.utc).isoformat()})


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
