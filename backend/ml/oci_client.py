"""
OCI Iris IncidentAPI client.

When real OCI credentials are provided via backend/.env
  OCI_USER_OCID, OCI_TENANCY_OCID, OCI_FINGERPRINT,
  OCI_PRIVATE_KEY_PEM (or OCI_PRIVATE_KEY_PATH), OCI_REGION,
  OCI_INCIDENT_API_BASE
this client issues authenticated requests with pagination and retry.

When credentials are NOT set, the client returns the uploaded Excel workbook
as a stand-in so the rest of the pipeline keeps working end-to-end.
"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

logger = logging.getLogger("ml.oci_client")

ROOT = Path(__file__).resolve().parents[1]
EXCEL_FALLBACK = ROOT / "data_store" / "raw" / "noc_incidents.xlsx"


class OCIAuthError(RuntimeError):
    ...


class OCIClient:
    def __init__(self) -> None:
        self.base = os.environ.get("OCI_INCIDENT_API_BASE", "").rstrip("/")
        self.region = os.environ.get("OCI_REGION", "us-phoenix-1")
        self.tenancy = os.environ.get("OCI_TENANCY_OCID", "")
        self.user = os.environ.get("OCI_USER_OCID", "")
        self.fingerprint = os.environ.get("OCI_FINGERPRINT", "")
        self.private_key_pem = os.environ.get("OCI_PRIVATE_KEY_PEM", "")
        self.private_key_path = os.environ.get("OCI_PRIVATE_KEY_PATH", "")
        self.session = requests.Session()
        self.signer = None
        self.mode = "mock"
        self._try_enable_live()

    # ---- auth ----
    def _try_enable_live(self) -> None:
        required = [self.base, self.tenancy, self.user, self.fingerprint]
        has_key = bool(self.private_key_pem) or bool(self.private_key_path)
        if not all(required) or not has_key:
            logger.info("OCI live mode disabled (missing env). Using Excel fallback.")
            return
        try:
            import oci  # type: ignore
            from oci.signer import Signer  # type: ignore
            if self.private_key_pem:
                key_bytes = self.private_key_pem.encode()
                signer = Signer(
                    tenancy=self.tenancy, user=self.user,
                    fingerprint=self.fingerprint,
                    private_key_content=key_bytes,
                )
            else:
                signer = Signer(
                    tenancy=self.tenancy, user=self.user,
                    fingerprint=self.fingerprint,
                    private_key_file_location=self.private_key_path,
                )
            self.signer = signer
            self.mode = "live"
            logger.info("OCI live mode enabled (region=%s)", self.region)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCI auth init failed: %s -- falling back to mock", exc)
            self.mode = "mock"

    # ---- live request with retry & pagination ----
    def _request(self, method: str, path: str, params: Optional[Dict] = None,
                 max_retries: int = 5) -> requests.Response:
        url = f"{self.base}{path}"
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = self.session.request(
                    method, url, params=params, auth=self.signer, timeout=15,
                )
                if resp.status_code in (429,) or resp.status_code >= 500:
                    ra = float(resp.headers.get("Retry-After", delay))
                    logger.warning("OCI %s retryable %s (attempt %d/%d) -> sleep %.1fs",
                                   method, resp.status_code, attempt + 1, max_retries, ra)
                    time.sleep(ra)
                    delay = min(delay * 2, 16)
                    continue
                if resp.status_code in (401, 403):
                    raise OCIAuthError(f"OCI auth failed: {resp.status_code} {resp.text[:200]}")
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("OCI network error %s (attempt %d/%d)", exc, attempt + 1, max_retries)
                time.sleep(delay)
                delay = min(delay * 2, 16)
        raise RuntimeError(f"OCI request exhausted retries: {last_exc}")

    def paginate(self, path: str, params: Optional[Dict] = None) -> Iterable[Dict]:
        params = dict(params or {})
        params.setdefault("limit", 100)
        while True:
            resp = self._request("GET", path, params=params)
            body = resp.json()
            items = body.get("items") or body.get("data") or []
            for it in items:
                yield it
            next_page = resp.headers.get("opc-next-page")
            if not next_page:
                return
            params["page"] = next_page

    # ---- public API ----
    def list_incidents(self, limit: int = 200) -> List[Dict[str, Any]]:
        if self.mode == "live":
            out: List[Dict[str, Any]] = []
            for it in self.paginate("/20241201/incidents", {"limit": min(100, limit)}):
                out.append(it)
                if len(out) >= limit:
                    break
            return out
        return self._from_excel(limit=limit)

    def health(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "base": self.base if self.mode == "live" else "(excel fallback)",
            "region": self.region,
            "has_signer": self.signer is not None,
            "excel_fallback_exists": EXCEL_FALLBACK.exists(),
        }

    # ---- Excel mock ----
    def _from_excel(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not EXCEL_FALLBACK.exists():
            return []
        try:
            df = pd.read_excel(EXCEL_FALLBACK, sheet_name="db_NOC_Alias")
            aliases = df["Alias"].dropna().astype(str).tolist()[:limit]
            return [{"alias": a, "source": "excel"} for a in aliases]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Excel fallback read failed: %s", exc)
            return []


CLIENT = OCIClient()
