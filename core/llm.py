"""Gemini LLM orchestration for conversational sales turns and memory summarization."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import (
    COMPANY_NAME,
    COMPANY_PRODUCTS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

logger = logging.getLogger(__name__)

VALID_SENTIMENTS: frozenset[str] = frozenset({"positive", "neutral", "negative"})
VALID_INTENTS: frozenset[str] = frozenset(
    {"inquiry", "pricing", "objection", "booking", "complaint", "other"}
)
EXTRACTED_KEYS: tuple[str, ...] = (
    "name",
    "email",
    "phone",
    "company",
    "budget",
    "timeline",
    "need",
)


def _default_turn_response(
    reply: str = "I am sorry, I had trouble processing that. Could you please repeat?",
    language: str = "en",
) -> dict[str, Any]:
    """Return a safe fallback turn payload matching the expected JSON schema."""
    return {
        "reply": reply,
        "language": language or "en",
        "sentiment": "neutral",
        "sentiment_score": 0.0,
        "intent": "other",
        "extracted": {key: None for key in EXTRACTED_KEYS},
        "wants_booking": False,
        "preferred_time": None,
    }


def _strip_code_fences(text: str) -> str:
    """Remove optional Markdown code fences around a JSON response."""
    cleaned = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def _normalize_turn_payload(data: dict[str, Any], default_lang: str) -> dict[str, Any]:
    """Validate and coerce parsed JSON fields into the strict turn schema."""
    fallback = _default_turn_response(language=default_lang)

    reply = str(data.get("reply") or "").strip() or fallback["reply"]
    language = str(data.get("language") or default_lang or "en").strip().lower()

    sentiment = str(data.get("sentiment") or "neutral").strip().lower()
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"

    try:
        sentiment_score = float(data.get("sentiment_score", 0.0))
    except (TypeError, ValueError):
        sentiment_score = 0.0
    # Clamp score into [-1.0, 1.0] so downstream analytics stay bounded.
    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    intent = str(data.get("intent") or "other").strip().lower()
    if intent not in VALID_INTENTS:
        intent = "other"

    raw_extracted = data.get("extracted")
    extracted: dict[str, Any] = {}
    for key in EXTRACTED_KEYS:
        val = raw_extracted.get(key) if isinstance(raw_extracted, dict) else None
        if isinstance(val, str):
            val = val.strip() or None
        extracted[key] = val

    wants_booking = bool(data.get("wants_booking", False))
    preferred_time_raw = data.get("preferred_time")
    preferred_time = (
        preferred_time_raw.strip() or None
        if isinstance(preferred_time_raw, str)
        else None
    )

    return {
        "reply": reply,
        "language": language,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "intent": intent,
        "extracted": extracted,
        "wants_booking": wants_booking,
        "preferred_time": preferred_time,
    }


def _parse_turn_json(raw_text: str, default_lang: str) -> dict[str, Any]:
    """Parse Gemini output into the turn dict, falling back to plain reply on failure."""
    cleaned = _strip_code_fences(raw_text)
    if not cleaned:
        return _default_turn_response(language=default_lang)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return _normalize_turn_payload(parsed, default_lang=default_lang)
        logger.warning("LLM returned non-object JSON (%s); using plain fallback.", type(parsed))
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON output; falling back to plain text reply.")

    # Strip accidental quotes/braces only if needed; otherwise speak the plain text directly.
    return _default_turn_response(reply=cleaned, language=default_lang)


def _build_system_prompt(
    rag_context: str,
    lead_state: dict[str, Any],
    detected_lang: str,
    memory_summary: str,
) -> str:
    """Build the structured sales agent system prompt with strict grounding and JSON rules."""
    context_block = rag_context.strip() or "No product knowledge context provided."
    summary_block = memory_summary.strip() or "None yet."
    lead_block = json.dumps(lead_state, ensure_ascii=False)

    return (
        f"You are a friendly, helpful AI sales representative for {COMPANY_NAME}. "
        f"Our products/services are: {COMPANY_PRODUCTS}.\n\n"
        "CONVERSATION RULES:\n"
        f"1. Reply in the user's language (detected language: '{detected_lang}').\n"
        "2. Keep 'reply' under 3 sentences. Make it natural and voice-friendly: "
        "do NOT use markdown, bullet points, asterisks, or emojis.\n"
        "3. Answer factual, pricing, or product questions ONLY using the PROVIDED CONTEXT below. "
        "If the context does not contain the answer, politely say you will connect a human specialist to help with that.\n"
        "4. Naturally collect lead details (name, need, budget, timeline) one question at a time. "
        "Do not re-ask for fields already present in LEAD STATE.\n"
        "5. Analyze the user's latest message for sentiment, intent, newly shared lead fields, and booking interest.\n\n"
        f"EARLIER CONVERSATION SUMMARY:\n{summary_block}\n\n"
        f"LEAD STATE COLLECTED SO FAR:\n{lead_block}\n\n"
        f"PROVIDED CONTEXT:\n{context_block}\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY valid JSON with these exact keys:\n"
        "{\n"
        '  "reply": string,\n'
        '  "language": string,\n'
        '  "sentiment": "positive" | "neutral" | "negative",\n'
        '  "sentiment_score": float between -1.0 and 1.0,\n'
        '  "intent": "inquiry" | "pricing" | "objection" | "booking" | "complaint" | "other",\n'
        '  "extracted": {\n'
        '    "name": string or null,\n'
        '    "email": string or null,\n'
        '    "phone": string or null,\n'
        '    "company": string or null,\n'
        '    "budget": string or null,\n'
        '    "timeline": string or null,\n'
        '    "need": string or null\n'
        "  },\n"
        '  "wants_booking": boolean,\n'
        '  "preferred_time": string or null\n'
        "}"
    )


def _create_chat_client(json_mode: bool = True, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini LangChain chat model with optional JSON enforcement."""
    kwargs: dict[str, Any] = {
        "model": GEMINI_MODEL,
        "google_api_key": GEMINI_API_KEY,
        "temperature": temperature,
    }
    if json_mode:
        # Enforce structured JSON output directly at the Gemini API layer.
        kwargs["response_mime_type"] = "application/json"
    return ChatGoogleGenerativeAI(**kwargs)


def _extract_response_text(response: Any) -> str:
    """Normalize LangChain AIMessage content (str or list of content parts) to text."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        return "".join(parts)
    return str(content or "")


def generate_turn(
    user_text: str,
    history: list[dict[str, Any]] | None = None,
    rag_context: str = "",
    lead_state: dict[str, Any] | None = None,
    detected_lang: str = "en",
    memory_summary: str = "",
) -> dict[str, Any]:
    """Generate a structured sales agent turn using Gemini and return a normalized dict."""
    cleaned_input = (user_text or "").strip()
    if not cleaned_input:
        logger.warning("generate_turn called with empty user_text.")
        return _default_turn_response(
            reply="I did not catch that. Could you please say that again?",
            language=detected_lang or "en",
        )

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is missing; returning fallback response.")
        return _default_turn_response(
            reply="Our AI assistant is temporarily unavailable. Let me connect you with a human specialist.",
            language=detected_lang or "en",
        )

    system_prompt = _build_system_prompt(
        rag_context=rag_context,
        lead_state=lead_state or {},
        detected_lang=detected_lang or "en",
        memory_summary=memory_summary,
    )

    messages: list[Any] = [SystemMessage(content=system_prompt)]
    for item in history or []:
        role = item.get("role", "user")
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if role == "assistant":
            messages.append(AIMessage(content=text))
        else:
            messages.append(HumanMessage(content=text))
    messages.append(HumanMessage(content=cleaned_input))

    try:
        client = _create_chat_client(json_mode=True, temperature=0.3)
        response = client.invoke(messages)
        raw_text = _extract_response_text(response)
        return _parse_turn_json(raw_text, default_lang=detected_lang or "en")
    except Exception:
        logger.exception("Gemini generate_turn failed.")
        return _default_turn_response(
            reply="Sorry, I ran into a brief technical issue. Let me connect you with a human team member.",
            language=detected_lang or "en",
        )


def summarize_messages(
    messages_to_summarize: list[dict[str, Any]],
    existing_summary: str = "",
) -> str:
    """Summarize older conversation turns into a concise running memory string."""
    if not messages_to_summarize:
        return existing_summary

    transcript_lines = [
        f"{msg.get('role', 'user').upper()}: {msg.get('text', '')}"
        for msg in messages_to_summarize
        if str(msg.get("text") or "").strip()
    ]
    if not transcript_lines:
        return existing_summary

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY missing during summarization; using plain text fallback.")
        combined = (existing_summary + "\n" if existing_summary else "") + " | ".join(transcript_lines)
        return combined[-1000:]

    prompt = (
        "Update the running sales call summary using the earlier summary and new messages. "
        "Keep key facts, user needs, objections, budget, timeline, and contact details in 3-5 concise sentences.\n\n"
        f"EXISTING SUMMARY:\n{existing_summary or 'None'}\n\n"
        "OLDER MESSAGES TO MERGE:\n"
        + "\n".join(transcript_lines)
    )

    try:
        client = _create_chat_client(json_mode=False, temperature=0.1)
        response = client.invoke([HumanMessage(content=prompt)])
        summary = _extract_response_text(response).strip()
        return summary or existing_summary
    except Exception:
        logger.exception("Gemini summarize_messages failed; preserving previous summary.")
        fallback = (existing_summary + " " if existing_summary else "") + " ".join(transcript_lines)
        return fallback[:1000]
