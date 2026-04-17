"""Train ZoomNet on synthetic (or real) incident data and persist artefacts."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .feature_engineering import build_feature_matrix, incidents_to_frame, save_encoders
from .model import ZoomNet
from .synthetic import generate_incidents

logger = logging.getLogger("ml.train")

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data_store" / "models" / "zoom_net" / "latest"


@dataclass
class TrainResult:
    n_samples: int
    input_dim: int
    epochs: int
    best_val_prauc: float
    val_roc_auc: float
    val_f1: float
    val_brier: float
    positive_rate: float
    threshold: float
    artifact_dir: str
    feature_count: int


def _pick_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                    fn_cost: float = 5.0, fp_cost: float = 1.0) -> tuple[float, float]:
    best_t, best_score = 0.5, 1e18
    for t in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_score:
            best_score, best_t = cost, float(t)
    return best_t, best_score


def train(
    rows: List[Dict] | None = None,
    n_synthetic: int = 3000,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if rows is None:
        rows = generate_incidents(n=n_synthetic, seed=seed)
    logger.info("Training on %d incidents", len(rows))

    df = incidents_to_frame(rows)
    y = df["zoom_required"].astype(int).to_numpy()
    X, scaler, feature_names = build_feature_matrix(df)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ZoomNet(input_dim=X.shape[1]).to(device)

    pos_weight_val = float((y_tr == 0).sum()) / max(1, (y_tr == 1).sum())
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device).unsqueeze(1)
    X_va_t = torch.from_numpy(X_va).to(device)

    n = X_tr_t.shape[0]
    best_prauc, best_state, stale = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimiser.zero_grad()
            out = model(X_tr_t[idx])
            loss = criterion(out, y_tr_t[idx])
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(X_va_t)).cpu().numpy().ravel()
        prauc = float(average_precision_score(y_va, val_prob))
        logger.info("epoch %02d loss=%.4f val_prauc=%.4f", epoch, float(np.mean(losses)), prauc)
        if prauc > best_prauc:
            best_prauc = prauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(X_va_t)).cpu().numpy().ravel()

    threshold, _ = _pick_threshold(y_va, val_prob)
    val_pred = (val_prob >= threshold).astype(int)
    roc = float(roc_auc_score(y_va, val_prob))
    f1 = float(f1_score(y_va, val_pred))
    brier = float(brier_score_loss(y_va, val_prob))

    # Persist artefacts
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR / "model.pt")
    save_encoders(ARTIFACT_DIR, scaler, feature_names)
    metadata = {
        "input_dim": int(X.shape[1]),
        "feature_count": int(X.shape[1]),
        "feature_names": feature_names,
        "threshold": float(threshold),
        "metrics": {
            "pr_auc": best_prauc, "roc_auc": roc, "f1": f1, "brier": brier,
        },
        "training": {
            "n_samples": int(len(rows)),
            "positive_rate": float(y.mean()),
            "epochs_run": epoch + 1,
            "pos_weight": pos_weight_val,
            "seed": seed,
        },
        "model_version": "zoom_net-v1",
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return TrainResult(
        n_samples=len(rows),
        input_dim=int(X.shape[1]),
        epochs=epoch + 1,
        best_val_prauc=best_prauc,
        val_roc_auc=roc,
        val_f1=f1,
        val_brier=brier,
        positive_rate=float(y.mean()),
        threshold=float(threshold),
        artifact_dir=str(ARTIFACT_DIR),
        feature_count=int(X.shape[1]),
    )
