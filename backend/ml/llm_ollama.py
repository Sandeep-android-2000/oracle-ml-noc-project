"""Ollama Mistral client with Claude-Haiku fallback.

Calls Ollama on http://localhost:11434 (override via OLLAMA_HOST env).
If the daemon is not reachable or the `mistral` model isn't pulled yet,
falls back to Claude Haiku 4.5 so the UI column always populates.
The returned dict carries a `provider` label so the frontend can show
which LLM produced the text.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict, Tuple

import httpx

from .llm_explain import explain_prediction as _claude_explain

logger = logging.getLogger("ml.llm_ollama")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "30"))

SYSTEM_PROMPT = (
    "You are a senior NOC incident triage assistant at Oracle Cloud. "
    "Given a single NOC incident's key facts and an ML model's Zoom-call "
    "probability, write ONE or TWO concise English sentences (max 35 words) "
    "explaining WHY the engineer should or should not join a Zoom bridge. "
    "Reference concrete features (severity, region, customer impact, "
    "workstreams, prior zoom rate). No bullets, no disclaimers, no "
    "repeating the probability verbatim."
)


def _compact(inc: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "alias", "severity", "status", "region", "title",
        "multi_region", "regions_affected",
        "workstream_count", "attachment_count", "page_count",
        "has_customer_impact", "has_outage", "has_broadcast",
        "prior_zoom_rate_reporter_30d", "prior_zoom_rate_region_30d",
        "prior_zoom_rate_severity_30d", "open_since_min",
    ]
    return {k: inc[k] for k in keep if inc.get(k) is not None}


async def _call_ollama(incident: Dict[str, Any],
                       prediction: Dict[str, Any]) -> Tuple[str, str]:
    user_text = (
        f"Incident facts: {_compact(incident)}\n"
        f"Model prediction: "
        f"{{decision: {prediction.get('decision')}, "
        f"probability: {prediction.get('probability')}}}\n"
        "Write the explanation now."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_text,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 120},
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as cli:
        resp = await cli.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        resp.raise_for_status()
        body = resp.json()
    return str(body.get("response", "")).strip(), f"ollama/{OLLAMA_MODEL}"


async def explain(incident: Dict[str, Any],
                  prediction: Dict[str, Any]) -> Dict[str, str]:
    """Ollama Mistral first, Claude Haiku fallback."""
    try:
        text, provider = await _call_ollama(incident, prediction)
        if text:
            return {"text": text, "provider": provider}
        raise RuntimeError("empty response from ollama")
    except Exception as exc:  # noqa: BLE001
        logger.info("Ollama unavailable (%s) -- falling back to Claude Haiku", exc)
        text = await _claude_explain(incident, prediction)
        return {"text": text, "provider": "claude-haiku-4-5-fallback"}
