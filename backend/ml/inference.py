"""Load the trained ZoomNet artefact and score incidents."""
from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .feature_engineering import build_feature_matrix, incidents_to_frame, load_encoders
from .model import ZoomNet

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data_store" / "models" / "zoom_net" / "latest"


class InferenceEngine:
    _lock = threading.Lock()

    def __init__(self):
        self.model: ZoomNet | None = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.threshold: float = 0.5
        self.metadata: Dict = {}
        self.loaded = False

    def load(self):
        with self._lock:
            meta_path = ARTIFACT_DIR / "metadata.json"
            if not meta_path.exists():
                self.loaded = False
                return
            self.metadata = json.loads(meta_path.read_text())
            self.scaler, self.feature_names = load_encoders(ARTIFACT_DIR)
            self.threshold = float(self.metadata["threshold"])
            model = ZoomNet(input_dim=int(self.metadata["input_dim"]))
            state = torch.load(ARTIFACT_DIR / "model.pt", map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self.loaded = True

    def is_ready(self) -> bool:
        return self.loaded and self.model is not None

    def predict_many(self, rows: List[Dict]) -> List[Dict]:
        if not rows:
            return []
        if not self.is_ready():
            self.load()
            if not self.is_ready():
                # Fallback rule: SEV1/SEV2 → Zoom
                return [{
                    "alias": r.get("alias"),
                    "probability": 0.9 if r.get("severity") in ("SEV1", "SEV2") else 0.1,
                    "decision": "Yes" if r.get("severity") in ("SEV1", "SEV2") else "No",
                    "confidence": 0.55,
                    "reason": "fallback_rule_no_model",
                    "model_version": "rule-v0",
                } for r in rows]

        df = incidents_to_frame(rows)
        X, _, _ = build_feature_matrix(df, scaler=self.scaler)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

        T_low = max(0.0, self.threshold - 0.15)
        T_high = min(1.0, self.threshold + 0.08)

        out: List[Dict] = []
        for r, p in zip(rows, probs):
            # Business override: SEV1 always Yes
            if r.get("severity") == "SEV1":
                decision = "Yes"
                reason = "sev1_hard_rule"
            elif p >= T_high:
                decision = "Yes"
                reason = "model_high_conf"
            elif p <= T_low:
                decision = "No"
                reason = "model_low_conf"
            else:
                decision = "Review"
                reason = "model_uncertain"
            if decision == "Yes":
                conf = float(p)
            elif decision == "No":
                conf = float(1 - p)
            else:
                conf = float(1 - 2 * abs(p - 0.5))  # 1.0 = most uncertain
            out.append({
                "alias": r.get("alias"),
                "probability": round(float(p), 4),
                "decision": decision,
                "confidence": round(conf, 4),
                "reason": reason,
                "model_version": self.metadata.get("model_version", "zoom_net-v1"),
            })
        return out


ENGINE = InferenceEngine()
