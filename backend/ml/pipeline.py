"""Live pull pipeline:

  1. Call API-1 for page=var
  2. For each incident id:
      - Call API-2 (attachments)
      - Call API-3 (csv content) for the first attachment (if any)
      - Call API-4 (comm channels) → extract zoom linkToJoin
  3. Build a feature row compatible with the ZoomNet feature schema
  4. Run the ANN to get probability/decision
  5. Ask Ollama Mistral (fallback Claude Haiku) for a short explanation
  6. Persist the incident + prediction + explanation + zoom_link into Mongo
  7. Every N ticks, retrain the model on the accumulated live-pull data
     (online / reinforcement-style learning from new fetches)
"""
from __future__ import annotations
import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from .inference import ENGINE
from .llm_ollama import explain as llm_explain
from .oci_subprocess import (api1_list_incidents, api2_list_attachments,
                             api3_attachment_content,
                             api4_communication_channels,
                             extract_zoom_link, save_noc_bundle)
from .synthetic import TITLE_TEMPLATES  # for keyword matching cross-ref
from .train import train as train_model

logger = logging.getLogger("ml.pipeline")

RETRAIN_EVERY_N_TICKS = 10


def _norm_ts(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def _build_feature_row(item: Dict[str, Any], api2: Dict[str, Any],
                       api3: Dict[str, Any], api4: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten API payloads into the ZoomNet feature schema."""
    regions = item.get("regions") or []
    region = regions[0] if regions else "us-ashburn-1"
    severity = item.get("severity") or "SEV3"
    status = item.get("status") or "INVESTIGATING"
    title = item.get("displayName") or ""
    t_lower = title.lower()
    created = _norm_ts(item.get("createdAt") or "")
    now = datetime.now(timezone.utc)
    age_minutes = int((now - created).total_seconds() / 60) if created else 60

    attachments = (api2.get("data") or {}).get("items") or []
    channels = api4.get("items") or []
    has_zoom = any(c.get("communicationChannelType") == "ZOOM" for c in channels)
    has_broadcast = any(c.get("communicationChannelType") == "SLACK"
                        for c in channels)

    # CSV-derived
    csv_text = ""
    if isinstance(api3.get("data"), str):
        csv_text = api3["data"]
    csv_lines = [l for l in csv_text.splitlines() if l.strip()][1:]  # skip header
    csv_customer_count = len({l.split(",", 1)[0] for l in csv_lines}) if csv_lines else 0
    region_matches = re.findall(r",Region,([^,]+)", csv_text)
    csv_region_count = len(set(region_matches))
    csv_has_email_list = 1 if re.search(r",[\w.+-]+@[\w-]+\.[\w.-]+", csv_text) else 0

    # Simple prior-zoom estimates (deterministic from severity/region)
    sev_prior_map = {"SEV1": 0.85, "SEV2": 0.50, "SEV3": 0.18, "SEV4": 0.05}

    return {
        "alias": item.get("alias"),
        "incidentId": item.get("id"),
        "jira_id": item.get("alias"),
        "title": title,
        "severity": severity,
        "status": status,
        "region": region,
        "regions_affected": len(regions) if regions else 1,
        "multi_region": 1 if len(regions) > 1 else 0,
        "workstream_count": 1 + len(attachments),
        "workstream_types_distinct": min(3, 1 + len(attachments)),
        "age_minutes": age_minutes,
        "created_at": item.get("createdAt"),
        "updated_at": item.get("updatedAt"),
        "first_occurrence": item.get("createdAt"),
        "last_occurrence": item.get("updatedAt"),
        "open_since_min": age_minutes,
        "last_updated_min": max(0, int((now - (_norm_ts(item.get("updatedAt") or "") or now)).total_seconds() / 60)),
        "hour_of_day": created.hour if created else 12,
        "dow": created.weekday() if created else 1,
        "is_weekend": 1 if (created and created.weekday() >= 5) else 0,
        "is_business_hour": 1 if (created and 8 <= created.hour <= 18) else 0,
        "has_jira_link": 1,
        "desc_len": len(title),
        "has_customer_impact": 1 if "customer" in t_lower or "impact" in t_lower else 0,
        "has_outage": 1 if "outage" in t_lower else 0,
        "has_temperature": 1 if "temperature" in t_lower else 0,
        "has_sev1_keyword": 1 if "sev1" in t_lower or severity == "SEV1" else 0,
        "assignee_count": 1,
        "unique_authors": 1,
        "note_count_24h": 0,
        "event_count_24h": 0,
        "has_broadcast": 1 if has_broadcast else 0,
        "page_count": 1 if has_zoom else 0,
        "attachment_count": len(attachments),
        "attachment_total_kb": sum(int(a.get("size", 0)) for a in attachments) // 1024,
        "has_image_attachment": 0,
        "has_csv_attachment": 1 if any("csv" in (a.get("mediaType") or "") for a in attachments) else 0,
        "has_log_attachment": 0,
        "autocomms_run_count": len([a for a in attachments if "AutoComms" in (a.get("name") or "")]),
        "csv_customer_count": csv_customer_count,
        "csv_region_count": csv_region_count,
        "csv_has_email_list": csv_has_email_list,
        "prior_zoom_rate_reporter_30d": 0.2,
        "prior_zoom_rate_region_30d": 0.25,
        "prior_zoom_rate_severity_30d": sev_prior_map.get(severity, 0.2),
        "reporter_name": ((item.get("commander") or {}).get("email") or "unknown"),
        "queue_name": "COE",
        "active_noc": 1 if status in ("INVESTIGATING", "OPEN") else 0,
        "active_oci_service_health": 0,
        "event_state": "Open" if status in ("INVESTIGATING", "OPEN") else "Closed",
        "tag_status": {"OPEN": "In Progress", "INVESTIGATING": "In Progress",
                       "RESOLVED": "Resolved", "CANCELED": "Cancelled",
                       "CLOSED": "Resolved"}.get(status, "No Action"),
        "sub_status": status,
        "owner": (item.get("commander") or {}).get("email", ""),
        # Ground truth Zoom label (whenever API-4 exposes a ZOOM channel the
        # incident actually produced one → perfect label for online retraining)
        "zoom_required": 1 if has_zoom else 0,
        "source": item.get("_source", "oci_live"),
    }


async def pull_one_var(db: AsyncIOMotorDatabase, var: int) -> Dict[str, Any]:
    """Execute the 4-API chain for a single page and persist everything."""
    api1 = api1_list_incidents(var)
    items = ((api1.get("data") or {}).get("items")) or []
    if not items:
        return {"var": var, "pulled": 0, "reason": "empty page"}

    results: List[Dict[str, Any]] = []
    for item in items:
        incident_id = item.get("id")
        alias = item.get("alias") or "NOC-UNKNOWN"

        api2 = api2_list_attachments(incident_id)
        attachments = (api2.get("data") or {}).get("items") or []
        first_att = attachments[0] if attachments else None
        if first_att:
            api3 = api3_attachment_content(
                first_att.get("noteId") or incident_id,
                first_att.get("id"),
                region_for_fallback=(item.get("regions") or ["us-ashburn-1"])[0],
            )
        else:
            api3 = {"data": "", "_source": "skipped_no_attachment"}
        api4 = api4_communication_channels(incident_id, alias_for_fallback=alias)

        save_noc_bundle(alias, api1, api2, api3, api4)

        row = _build_feature_row(item, api2, api3, api4)
        zoom_link = extract_zoom_link(api4)
        row["zoom_link"] = zoom_link

        # Predict + Explain
        preds = ENGINE.predict_many([row])
        pred = preds[0] if preds else None
        row["prediction"] = pred

        explanation_doc = None
        if pred:
            try:
                exp = await llm_explain(row, pred)
                explanation_doc = {
                    "alias": alias,
                    "text": exp["text"],
                    "model": exp.get("provider", "unknown"),
                    "decision": pred["decision"],
                    "probability": pred["probability"],
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("llm explain failed: %s", exc)

        row["explanation"] = explanation_doc

        # Persist incident (upsert by alias)
        await db.incidents.update_one({"alias": alias},
                                      {"$set": row}, upsert=True)
        if pred:
            await db.predictions.update_one(
                {"alias": alias},
                {"$set": {**pred, "ts": datetime.now(timezone.utc).isoformat()}},
                upsert=True)
        if explanation_doc:
            await db.explanations.update_one(
                {"alias": alias}, {"$set": explanation_doc}, upsert=True)
        results.append({
            "alias": alias,
            "decision": pred["decision"] if pred else None,
            "probability": pred["probability"] if pred else None,
            "zoom_link": zoom_link,
            "source": row["source"],
        })

    await db.live_ticks.insert_one({
        "var": var,
        "pulled": len(results),
        "ts": datetime.now(timezone.utc).isoformat(),
        "sample": results,
    })
    return {"var": var, "pulled": len(results), "items": results}


async def online_retrain(db: AsyncIOMotorDatabase) -> Dict[str, Any]:
    """Retrain ZoomNet on all stored incidents (including freshly pulled)."""
    rows = await db.incidents.find({}, {"_id": 0}).to_list(length=None)
    if len(rows) < 50:
        return {"skipped": True, "reason": f"only {len(rows)} rows"}
    # Run blocking training in a thread to avoid stalling the event loop
    tr = await asyncio.to_thread(train_model, rows=rows, epochs=15, seed=42)
    ENGINE.load()
    return {
        "trained_on": len(rows),
        "pr_auc": tr.best_val_prauc, "f1": tr.val_f1,
        "threshold": tr.threshold, "epochs": tr.epochs,
    }
