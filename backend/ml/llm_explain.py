"""LLM-generated natural-language explanation of a Zoom-call prediction.

Uses the Emergent LLM key (EMERGENT_LLM_KEY) via `emergentintegrations`
and Claude Haiku 4.5 (claude-haiku-4-5-20251001).
"""
from __future__ import annotations
import logging
import os
import uuid
from typing import Any, Dict

from emergentintegrations.llm.chat import LlmChat, UserMessage  # type: ignore

logger = logging.getLogger("ml.llm_explain")

MODEL = "claude-haiku-4-5-20251001"
PROVIDER = "anthropic"
SYSTEM_PROMPT = (
    "You are a senior NOC incident triage assistant at Oracle Cloud. "
    "Given a single NOC incident's key facts and the ML model's Zoom-call "
    "prediction, write ONE or TWO concise English sentences (max 35 words) "
    "that explain WHY the model reached that decision. Reference the most "
    "important concrete features (severity, region, customer impact, "
    "workstreams, prior zoom rate, etc). Do NOT add disclaimers, do NOT use "
    "bullet points, do NOT repeat the probability verbatim if it's in the "
    "input — paraphrase it (e.g. 'high', 'moderate', 'low')."
)


def _compact_incident(inc: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "alias", "severity", "status", "region", "title",
        "multi_region", "regions_affected",
        "workstream_count", "attachment_count", "page_count",
        "has_customer_impact", "has_outage", "has_broadcast",
        "prior_zoom_rate_reporter_30d", "prior_zoom_rate_region_30d",
        "prior_zoom_rate_severity_30d",
        "open_since_min",
    ]
    return {k: inc.get(k) for k in keep if inc.get(k) is not None}


async def explain_prediction(incident: Dict[str, Any],
                             prediction: Dict[str, Any]) -> str:
    """Return a 1-2 sentence English explanation of the Zoom-call prediction."""
    key = os.environ.get("EMERGENT_LLM_KEY")
    if not key:
        raise RuntimeError("EMERGENT_LLM_KEY missing from environment")

    compact = _compact_incident(incident)
    pred = {
        "decision": prediction.get("decision"),
        "probability": prediction.get("probability"),
        "reason": prediction.get("reason"),
    }
    user_text = (
        f"Incident facts: {compact}\n"
        f"Model prediction: {pred}\n"
        "Write the explanation now."
    )

    chat = LlmChat(
        api_key=key,
        session_id=f"zoom-explain-{uuid.uuid4()}",
        system_message=SYSTEM_PROMPT,
    ).with_model(PROVIDER, MODEL)

    response = await chat.send_message(UserMessage(text=user_text))
    # LlmChat returns either str or object; normalise
    text = response if isinstance(response, str) else getattr(response, "text", str(response))
    return text.strip()
