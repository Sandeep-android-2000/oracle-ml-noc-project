"""4-API OCI raw-request chain invoked via `subprocess`.

Chain:
  API-1  list incidents         (page={var})
  API-2  list attachments       (for each incident id)
  API-3  attachment CSV content (for each (noteId, attachmentId))
  API-4  communication channels (pull Zoom linkToJoin)

On a real OCI compute instance the `oci` CLI is available and the calls
run with `--auth instance_principal`. In any other environment the CLI
will fail; the caller falls back to deterministic synthetic data that
matches the exact JSON shapes shown in the spec so the end-to-end pipeline
still demonstrates correctly.

Every NOC incident pulled gets its own folder under
`backend/data_store/noc_pulls/{alias}/` with api1.json, api2.json,
api3.csv, api4.json.
"""
from __future__ import annotations
import hashlib
import json
import logging
import random
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ml.oci_subprocess")

BASE_URL = "https://api-iris-staging.us-phoenix-1.oci.oraclecloud.com"
ROOT = Path(__file__).resolve().parents[1]
NOC_PULLS_DIR = ROOT / "data_store" / "noc_pulls"
NOC_PULLS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- CLI wrapper -----------------------------

def _oci_raw_request(target_uri: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """Invoke `oci raw-request --auth instance_principal` and parse JSON.

    Returns None if the CLI is missing or the call fails.
    """
    cmd = [
        "oci", "raw-request",
        "--auth", "instance_principal",
        "--http-method", "GET",
        "--target-uri", target_uri,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        logger.info("oci CLI not found -- using synthetic fallback")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("oci CLI timeout calling %s", target_uri)
        return None

    if result.returncode != 0:
        logger.warning("oci CLI returned %d for %s: %s",
                       result.returncode, target_uri, result.stderr[:200])
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("oci CLI produced non-JSON stdout for %s", target_uri)
        return None


# ----------------------------- synthetic fallback -----------------------------

REGIONS = ["ap-melbourne-1", "eu-frankfurt-1", "us-ashburn-1", "us-phoenix-1",
           "ap-sydney-1", "uk-london-1", "ap-tokyo-1", "ca-toronto-1",
           "ap-mumbai-1", "ap-singapore-1"]
SEVERITIES = ["SEV1", "SEV2", "SEV3", "SEV4"]
STATUSES = ["INVESTIGATING", "OPEN", "RESOLVED", "CANCELED", "CLOSED"]
REPORTERS = [
    ("James George", "james.george@oracle.com"),
    ("Nikhil Jaladi", "nikhil.jaladi@oracle.com"),
    ("Murat Mukhtarov", "murat.mukhtarov@oracle.com"),
    ("Frank Jantunen", "frank.jantunen@oracle.com"),
    ("Allwyn Alexander", "allwyn.alexander@oracle.com"),
]


def _uuid_from(seed: str) -> str:
    h = hashlib.md5(seed.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _synth_api1(var: int) -> Dict[str, Any]:
    """One-item page matching the spec shape."""
    rng = random.Random(9001 + var)
    now = datetime.now(timezone.utc)
    alias_num = 1_600_000 + var
    alias = f"NOC-{alias_num}"
    incident_id = _uuid_from(f"incident:{alias}")
    region = rng.choice(REGIONS)
    severity = rng.choices(SEVERITIES, weights=[0.08, 0.25, 0.47, 0.20])[0]
    status = rng.choices(STATUSES, weights=[0.30, 0.20, 0.28, 0.12, 0.10])[0]
    commander_name, commander_email = rng.choice(REPORTERS)
    created = now - timedelta(minutes=rng.randint(10, 4000))

    return {
        "data": {
            "items": [{
                "alias": alias,
                "commander": {
                    "displayName": commander_name,
                    "email": commander_email,
                    "id": _uuid_from("user:" + commander_email),
                    "name": commander_email.split("@")[0],
                },
                "createdAt": created.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                "definedTags": None,
                "displayName": (
                    f"TESTING - A large number of PACT probes are failing in {region}"
                    if severity != "SEV1" else
                    f"Outage -- customer impact in {region} ({alias})"
                ),
                "freeformTags": None,
                "id": incident_id,
                "labels": {"items": [
                    "tosim-apac-created", "sync_noc_ticket",
                    f"pact_created_{severity.lower()}_event",
                ]},
                "projectKey": "NOC",
                "regions": [region],
                "severity": severity,
                "status": status,
                "systemTags": None,
                "ticket": None,
                "updatedAt": now.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            }]
        }
    }


def _synth_api2(incident_id: str, var: int) -> Dict[str, Any]:
    rng = random.Random(int(hashlib.md5(incident_id.encode()).hexdigest()[:8], 16))
    has_attachment = rng.random() < 0.75
    items: List[Dict[str, Any]] = []
    if has_attachment:
        items.append({
            "definedTags": None,
            "etag": hashlib.sha256(incident_id.encode()).hexdigest(),
            "freeformTags": None,
            "id": _uuid_from(f"att:{incident_id}"),
            "mediaType": "text/csv",
            "name": f"noc-pull-{var}_AutomatedAutoComms_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
            "noteId": _uuid_from(f"note:{incident_id}"),
            "otsAttachmentId": f"9{rng.randint(10000, 99999)}",
            "otsTicketId": f"IRIS-21-{rng.randint(5000000, 5999999)}",
            "size": rng.randint(500, 5000),
            "systemTags": None,
            "uploadedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "uploadedBy": {
                "displayName": "AUTOCOMMS",
                "email": "jichouha_directs_ww@oracle.com",
                "id": "ocid1.tenancy.oc1..aaaaaaaade4catkgnaurcikdecs6ng53x5jgzdpv2yalhh4fza5uxz4645xa",
                "name": None,
            },
            "workStreamId": "Incident",
        })
    return {"data": {"items": items},
            "headers": {"opc-total-items": str(len(items))},
            "status": "200 OK"}


def _synth_api3(region: str) -> str:
    """Synthetic CSV body matching the spec's shape."""
    header = ("tenantId,tenantName,email_address,Table_Header_1,Table_Content_1,"
              "Custom_Header_1,Custom_Name_1,Custom_Value_1,Table_Header_2,"
              "Table_Content_2,Custom_Header_2,Custom_Name_2,Custom_Value_2,"
              "Table_Header_3,Table_Content_3,Additional_Emails")
    rows = [header]
    for i in range(random.randint(3, 9)):
        tid = f"ocid1.tenancy.oc1..aaaaaaaa{hashlib.md5(f'{region}{i}'.encode()).hexdigest()[:40]}"
        rows.append(f"{tid},,,Region,{region},,,,,,,,,,,")
    return "\r\n".join(rows) + "\r\n"


def _synth_api4(incident_id: str, alias: str) -> Dict[str, Any]:
    """Zoom link always generated (matches real behaviour where Zoom is the
    default for incidents that produced a meeting)."""
    zoom_id = int(hashlib.md5(incident_id.encode()).hexdigest()[:10], 16) % (10**11)
    pwd = hashlib.md5(alias.encode()).hexdigest()[:22]
    zoom_link = f"https://oracle.zoom.us/j/{zoom_id}?pwd={pwd}"
    slack_channel = alias.lower().replace("-", "")
    return {
        "items": [
            {
                "id": "slackMain",
                "displayName": "Slack",
                "description": f"Incident slack channel for {alias}",
                "incidentId": incident_id,
                "workStreamId": "Incident",
                "communicationChannelType": "SLACK",
                "communicationChannelDetails": {
                    "linkToJoin": f"https://oracle.enterprise.slack.com/archives/{slack_channel}",
                    "data": json.dumps({"type": "COMMUNICATION_SLACK"}),
                },
            },
            {
                "id": "zoomMain",
                "displayName": "Zoom",
                "description": "Zoom bridge for those Paged to respond to this Incident.",
                "incidentId": incident_id,
                "workStreamId": "Incident",
                "communicationChannelType": "ZOOM",
                "communicationChannelDetails": {
                    "linkToJoin": zoom_link,
                    "data": json.dumps({
                        "type": "COMMUNICATION_ZOOM",
                        "linkToJoin": zoom_link,
                        "meetingId": str(zoom_id),
                    }),
                },
            },
        ]
    }


# ----------------------------- public pipeline entry-points -----------------------------

def api1_list_incidents(var: int, project_key: str = "NOC",
                        created_after: str = "2025-01-01T03:21:14.412Z",
                        created_before: str = "2026-04-02T03:21:14.412Z") -> Dict[str, Any]:
    uri = (f"{BASE_URL}/20241201/incidents?"
           f"createdAfter={created_after}&createdBefore={created_before}"
           f"&limit=1&page={var}&projectKey={project_key}")
    body = _oci_raw_request(uri)
    if body is None:
        body = _synth_api1(var)
        body["_source"] = "synthetic"
    else:
        body["_source"] = "oci_live"
    return body


def api2_list_attachments(incident_id: str) -> Dict[str, Any]:
    uri = f"{BASE_URL}/20241201/incidents/{incident_id}/workStreams/all/attachments"
    body = _oci_raw_request(uri)
    if body is None:
        body = _synth_api2(incident_id, var=0)
        body["_source"] = "synthetic"
    else:
        body["_source"] = "oci_live"
    return body


def api3_attachment_content(note_id: str, attachment_id: str,
                            region_for_fallback: str = "us-ashburn-1") -> Dict[str, Any]:
    uri = (f"{BASE_URL}/20241201/incidents/{note_id}/workStreams/"
           f"Incident/attachments/{attachment_id}/content")
    body = _oci_raw_request(uri)
    if body is None:
        body = {"data": _synth_api3(region_for_fallback), "_source": "synthetic"}
    else:
        body["_source"] = "oci_live"
    return body


def api4_communication_channels(incident_id: str,
                                alias_for_fallback: str = "NOC-FAKE") -> Dict[str, Any]:
    uri = (f"{BASE_URL}/20241201/incidents/{incident_id}/workStreams/"
           f"Incident/communicationChannels")
    body = _oci_raw_request(uri)
    if body is None:
        body = _synth_api4(incident_id, alias_for_fallback)
        body["_source"] = "synthetic"
    else:
        body["_source"] = "oci_live"
    return body


def save_noc_bundle(alias: str, api1: Dict[str, Any], api2: Dict[str, Any],
                    api3: Dict[str, Any], api4: Dict[str, Any]) -> Path:
    """Persist all 4 API responses under data_store/noc_pulls/{alias}/."""
    folder = NOC_PULLS_DIR / alias
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "api1_incident.json").write_text(json.dumps(api1, indent=2))
    (folder / "api2_attachments.json").write_text(json.dumps(api2, indent=2))
    # api3 stores CSV content as both wrapped JSON and raw CSV
    (folder / "api3_content.json").write_text(json.dumps(api3, indent=2))
    if isinstance(api3, dict) and isinstance(api3.get("data"), str):
        (folder / "api3_content.csv").write_text(api3["data"])
    (folder / "api4_channels.json").write_text(json.dumps(api4, indent=2))
    return folder


def extract_zoom_link(api4_body: Dict[str, Any]) -> Optional[str]:
    items = api4_body.get("items") or api4_body.get("data", {}).get("items") or []
    for it in items:
        if it.get("communicationChannelType") == "ZOOM":
            details = it.get("communicationChannelDetails") or {}
            link = details.get("linkToJoin")
            if link:
                return link
    return None
