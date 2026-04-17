"""
Synthetic NOC incident generator that mirrors the real Excel schema
(NOC_Incidents.xlsx) so the PyTorch ANN has enough labelled data to train
convincingly in the prototype. The real Excel workbook is mostly schema
samples (most sheets have 2 rows); production will swap this module out
for the OCI ingest in `ingest.py`.

Label logic is a hidden non-linear function of the same features the model
will see, plus ~8% Bernoulli noise, so the ANN must actually learn.
"""
from __future__ import annotations
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import numpy as np

REGIONS = [
    "eu-frankfurt-1", "us-ashburn-1", "ap-sydney-1", "ap-melbourne-1",
    "us-phoenix-1", "uk-london-1", "ap-tokyo-1", "ca-toronto-1",
    "ap-mumbai-1", "ap-singapore-1", "sa-saopaulo-1", "me-jeddah-1",
    "ap-batam-1", "us-sanjose-1", "ap-seoul-1",
]
SEVERITIES = ["SEV1", "SEV2", "SEV3", "SEV4"]
STATUSES = ["INVESTIGATING", "OPEN", "RESOLVED", "CANCELED", "CLOSED"]
WS_TYPES = ["OTHER", "TRIAGE", "INVESTIGATION", "COMMS", "PHYSICAL_NW",
            "VIRTUAL_NW", "DB_RECOVERY", "CUSTOMER_COMMS"]
TITLE_TEMPLATES = [
    "OC1 | {r} | High Temperature Detected -- (Multiple Alarms)",
    "[PACT Alert] Netprobe latency spike in {r}",
    "Customer impact -- {r} compute plane degraded",
    "Power event suspected in {r} physical room",
    "Storage IO saturation across {r}",
    "TESTING - new Response plan validation",
    "Control-plane throttling on {r}",
    "TLS cert renewal failure in {r} -- partial outage",
    "DNS resolution errors for {r}",
    "OKE node pool eviction storm {r}",
    "[NOC-{n}] Rack top-of-switch failure in {r}",
    "Database failover latency spike -- {r}",
]
SEV1_KEYWORDS = ["outage", "customer impact", "critical", "multi-region", "sev1"]


def _ts(now: datetime, minutes_ago: int) -> str:
    return (now - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_incidents(n: int = 3000, seed: int = 42) -> List[Dict[str, Any]]:
    """Return N synthetic incidents, each a flat dict with all feature fields
    + the ground-truth ``zoom_required`` label (0/1)."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    rows: List[Dict[str, Any]] = []

    for i in range(n):
        alias = f"NOC-{5000000 + i}"
        severity = rng.choices(SEVERITIES, weights=[0.06, 0.22, 0.45, 0.27])[0]
        status = rng.choices(STATUSES, weights=[0.28, 0.18, 0.34, 0.10, 0.10])[0]
        region = rng.choice(REGIONS)
        multi_region = int(rng.random() < 0.18)
        regions_affected = 1 + (np_rng.poisson(1.3) if multi_region else 0)
        workstream_count = max(1, int(np_rng.normal(3.5, 1.8)))
        workstream_types_distinct = min(workstream_count, rng.randint(1, 5))
        age_minutes = int(abs(np_rng.normal(480, 600)))
        created = now - timedelta(minutes=age_minutes)

        title = rng.choice(TITLE_TEMPLATES).format(r=region, n=190000 + i)
        desc_len = int(abs(np_rng.normal(180, 90)))
        has_customer_impact = int("customer" in title.lower() or rng.random() < 0.25)
        has_outage = int("outage" in title.lower() or rng.random() < 0.12)
        has_temperature = int("temperature" in title.lower())
        has_sev1_keyword = int(any(k in title.lower() for k in SEV1_KEYWORDS))

        assignee_count = max(1, int(np_rng.normal(2.2, 1.2)))
        unique_authors = max(1, int(np_rng.normal(3.4, 1.8)))
        note_count_24h = int(abs(np_rng.normal(6, 5)))
        event_count_24h = int(abs(np_rng.normal(12, 9)))
        has_broadcast = int(rng.random() < 0.22)
        page_count = int(abs(np_rng.normal(1.4, 1.6)))

        attachment_count = int(abs(np_rng.normal(1.5, 2)))
        attachment_total_kb = int(abs(np_rng.normal(500, 900))) if attachment_count else 0
        has_image = int(attachment_count > 0 and rng.random() < 0.55)
        has_csv = int(attachment_count > 0 and rng.random() < 0.5)
        has_log = int(attachment_count > 0 and rng.random() < 0.25)
        autocomms_run_count = int(has_csv) * rng.randint(0, 5)
        csv_customer_count = autocomms_run_count * rng.randint(1, 25) if has_csv else 0
        csv_region_count = 1 + (rng.randint(0, 4) if multi_region else 0) if has_csv else 0
        csv_has_email_list = int(has_csv and rng.random() < 0.4)

        prior_zoom_rate_reporter_30d = round(float(np_rng.beta(2, 6)), 3)
        prior_zoom_rate_region_30d = round(float(np_rng.beta(2, 5)), 3)
        prior_zoom_rate_severity_30d = {
            "SEV1": 0.88, "SEV2": 0.52, "SEV3": 0.18, "SEV4": 0.05
        }[severity] + float(np_rng.normal(0, 0.05))
        prior_zoom_rate_severity_30d = float(np.clip(prior_zoom_rate_severity_30d, 0, 1))

        # ---- Ground-truth label: hidden non-linear decision function ----
        sev_w = {"SEV1": 3.8, "SEV2": 1.8, "SEV3": 0.3, "SEV4": -1.0}[severity]
        score = (
            sev_w
            + 1.4 * multi_region
            + 1.1 * has_customer_impact
            + 1.6 * has_outage
            + 0.6 * has_broadcast
            + 0.15 * page_count
            + 0.22 * (workstream_count - 3)
            + 0.9 * (csv_customer_count >= 5)
            + 0.8 * (attachment_count >= 3)
            + 0.5 * has_image
            + 3.0 * prior_zoom_rate_reporter_30d
            + 2.2 * prior_zoom_rate_region_30d
            + 1.7 * prior_zoom_rate_severity_30d
            - 0.002 * age_minutes
            - 1.2 * (status == "CANCELED")
            - 0.6 * (status == "RESOLVED")
        )
        prob = 1.0 / (1.0 + np.exp(-(score - 2.2)))
        label = int(rng.random() < prob)
        # label noise
        if rng.random() < 0.08:
            label = 1 - label

        rows.append(dict(
            alias=alias,
            incidentId=f"{alias}-{i:06d}",
            title=title,
            jira_id=f"NOC-{190000 + i}",
            severity=severity,
            status=status,
            region=region,
            regions_affected=int(regions_affected),
            multi_region=multi_region,
            workstream_count=int(workstream_count),
            workstream_types_distinct=int(workstream_types_distinct),
            age_minutes=int(age_minutes),
            created_at=_ts(now, age_minutes),
            updated_at=_ts(now, max(0, age_minutes - rng.randint(0, 60))),
            first_occurrence=_ts(now, age_minutes),
            last_occurrence=_ts(now, max(0, age_minutes - rng.randint(0, 120))),
            open_since_min=int(age_minutes),
            last_updated_min=int(max(0, age_minutes - rng.randint(0, 60))),
            hour_of_day=created.hour,
            dow=created.weekday(),
            is_weekend=int(created.weekday() >= 5),
            is_business_hour=int(8 <= created.hour <= 18),
            has_jira_link=1,
            desc_len=desc_len,
            has_customer_impact=has_customer_impact,
            has_outage=has_outage,
            has_temperature=has_temperature,
            has_sev1_keyword=has_sev1_keyword,
            assignee_count=int(assignee_count),
            unique_authors=int(unique_authors),
            note_count_24h=int(note_count_24h),
            event_count_24h=int(event_count_24h),
            has_broadcast=has_broadcast,
            page_count=int(page_count),
            attachment_count=int(attachment_count),
            attachment_total_kb=int(attachment_total_kb),
            has_image_attachment=has_image,
            has_csv_attachment=has_csv,
            has_log_attachment=has_log,
            autocomms_run_count=int(autocomms_run_count),
            csv_customer_count=int(csv_customer_count),
            csv_region_count=int(csv_region_count),
            csv_has_email_list=csv_has_email_list,
            prior_zoom_rate_reporter_30d=float(prior_zoom_rate_reporter_30d),
            prior_zoom_rate_region_30d=float(prior_zoom_rate_region_30d),
            prior_zoom_rate_severity_30d=float(prior_zoom_rate_severity_30d),
            reporter_name=rng.choice([
                "jirasd-incident-response-us-ashburn-1",
                "jirasd-telemetry-us-ashburn-1",
                "james.george@oracle.com",
                "ocean-auto-page@oracle.com",
                "air-test-user@oracle.com",
            ]),
            queue_name=rng.choice(["COE", "SRE", "NetworkOps", "DBOps"]),
            active_noc=int(status in ("INVESTIGATING", "OPEN")),
            active_oci_service_health=int(rng.random() < 0.04),
            event_state="Open" if status in ("INVESTIGATING", "OPEN") else "Closed",
            tag_status=rng.choice(["In Progress", "Resolved", "Cancelled", "No Action"]),
            sub_status=status,
            owner=rng.choice(["", "", "Unknown",
                              "frank.jantunen@oracle.com",
                              "murat.mukhtarov@oracle.com"]),
            zoom_required=int(label),
        ))
    return rows
