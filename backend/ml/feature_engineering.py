"""Feature engineering: turn incident dicts into a dense numpy matrix ready
for the PyTorch ANN. Encoders/scalers are persisted so inference matches
training exactly."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Top-N categories we will one-hot; anything else → __other__
TOP_REGIONS = [
    "eu-frankfurt-1", "us-ashburn-1", "ap-sydney-1", "ap-melbourne-1",
    "us-phoenix-1", "uk-london-1", "ap-tokyo-1", "ca-toronto-1",
    "ap-mumbai-1", "ap-singapore-1", "sa-saopaulo-1", "me-jeddah-1",
    "ap-batam-1", "us-sanjose-1", "ap-seoul-1",
]
STATUSES = ["INVESTIGATING", "OPEN", "RESOLVED", "CANCELED", "CLOSED"]
SEV_MAP = {"SEV1": 4, "SEV2": 3, "SEV3": 2, "SEV4": 1}

NUMERIC_COLS = [
    "regions_affected", "multi_region",
    "workstream_count", "workstream_types_distinct",
    "age_minutes", "hour_of_day", "dow", "is_weekend", "is_business_hour",
    "has_jira_link",
    "desc_len",
    "has_customer_impact", "has_outage", "has_temperature", "has_sev1_keyword",
    "assignee_count", "unique_authors", "note_count_24h", "event_count_24h",
    "has_broadcast", "page_count",
    "attachment_count", "attachment_total_kb",
    "has_image_attachment", "has_csv_attachment", "has_log_attachment",
    "autocomms_run_count", "csv_customer_count", "csv_region_count",
    "csv_has_email_list",
    "prior_zoom_rate_reporter_30d", "prior_zoom_rate_region_30d",
    "prior_zoom_rate_severity_30d",
]


def _one_hot(value: str, vocab: List[str]) -> List[int]:
    v = [0] * (len(vocab) + 1)  # +1 for __other__
    if value in vocab:
        v[vocab.index(value)] = 1
    else:
        v[-1] = 1
    return v


def incidents_to_frame(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["severity_ord"] = df["severity"].map(SEV_MAP).fillna(0).astype(int)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """Return (X, fitted_scaler, feature_names)."""
    feats: List[np.ndarray] = []
    names: List[str] = []

    # 1. severity ordinal
    feats.append(df["severity_ord"].to_numpy().reshape(-1, 1))
    names.append("severity_ord")

    # 2. region one-hot
    region_oh = np.array([_one_hot(r, TOP_REGIONS) for r in df["region"].fillna("")])
    feats.append(region_oh)
    names += [f"region__{r}" for r in TOP_REGIONS] + ["region__other"]

    # 3. status one-hot
    status_oh = np.array([_one_hot(s, STATUSES) for s in df["status"].fillna("")])
    feats.append(status_oh)
    names += [f"status__{s}" for s in STATUSES] + ["status__other"]

    # 4. numeric block
    num = df[NUMERIC_COLS].fillna(0).to_numpy(dtype=float)
    feats.append(num)
    names += NUMERIC_COLS

    X = np.concatenate(feats, axis=1).astype(np.float32)

    # Standard-scale only the numeric block to avoid disturbing OH bits.
    num_start = 1 + len(TOP_REGIONS) + 1 + len(STATUSES) + 1
    if scaler is None:
        scaler = StandardScaler()
        X[:, num_start:] = scaler.fit_transform(X[:, num_start:])
    else:
        X[:, num_start:] = scaler.transform(X[:, num_start:])

    return X, scaler, names


def save_encoders(path: Path, scaler: StandardScaler, feature_names: List[str]):
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path / "scaler.joblib")
    (path / "feature_names.json").write_text(json.dumps(feature_names))


def load_encoders(path: Path) -> Tuple[StandardScaler, List[str]]:
    scaler = joblib.load(path / "scaler.joblib")
    feature_names = json.loads((path / "feature_names.json").read_text())
    return scaler, feature_names
