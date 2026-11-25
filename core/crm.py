"""Pluggable CRM synchronization (local SQLite/CSV export and webhook POST with retries)."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import requests

from core.config import CRM_EXPORT_PATH, CRM_MODE, CRM_WEBHOOK_URL
from core.db import get_lead, list_appointments, list_calls, log_crm_sync

logger = logging.getLogger(__name__)

CSV_COLUMNS: tuple[str, ...] = (
    "lead_id",
    "name",
    "phone",
    "email",
    "company",
    "language",
    "score",
    "status",
    "appointment_start",
    "appointment_end",
    "summary",
    "sync_status",
)


def build_crm_payload(
    lead: dict[str, Any],
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Assemble a normalized CRM JSON payload containing lead, score, appointment, and summary."""
    lead_id = lead.get("id") or lead.get("lead_id")
    db_lead = get_lead(int(lead_id), db_path=db_path) if lead_id is not None else None
    merged_lead = {**(db_lead or {}), **lead}

    appointment = merged_lead.get("appointment")
    if appointment is None and lead_id is not None:
        appts = [
            appt
            for appt in list_appointments(lead_id=int(lead_id), db_path=db_path)
            if appt.get("status") == "booked"
        ]
        appointment = appts[-1] if appts else None

    summary = merged_lead.get("summary")
    if not summary and lead_id is not None:
        calls = list_calls(lead_id=int(lead_id), db_path=db_path)
        for call in calls:
            if call.get("summary"):
                raw_summary = str(call["summary"]).strip()
                if raw_summary.startswith("{"):
                    try:
                        summary = json.loads(raw_summary).get("summary") or raw_summary
                    except json.JSONDecodeError:
                        summary = raw_summary
                else:
                    summary = raw_summary
                break

    return {
        "lead": {
            "id": lead_id,
            "name": merged_lead.get("name"),
            "phone": merged_lead.get("phone"),
            "email": merged_lead.get("email"),
            "company": merged_lead.get("company"),
            "language": merged_lead.get("language", "en"),
        },
        "score": merged_lead.get("score", 0),
        "status": merged_lead.get("status", "Cold"),
        "appointment": appointment,
        "summary": summary or "",
    }


def _append_local_csv(
    payload: dict[str, Any],
    sync_status: str,
    csv_path: Path,
) -> None:
    """Append a flattened CRM sync row to the local CSV export file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    lead_info = payload.get("lead") or {}
    appt = payload.get("appointment") or {}

    row = {
        "lead_id": lead_info.get("id"),
        "name": lead_info.get("name"),
        "phone": lead_info.get("phone"),
        "email": lead_info.get("email"),
        "company": lead_info.get("company"),
        "language": lead_info.get("language"),
        "score": payload.get("score"),
        "status": payload.get("status"),
        "appointment_start": appt.get("slot_start"),
        "appointment_end": appt.get("slot_end"),
        "summary": payload.get("summary"),
        "sync_status": sync_status,
    }
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _post_webhook_with_retries(
    url: str,
    payload: dict[str, Any],
    max_retries: int = 3,
) -> tuple[bool, str]:
    """POST CRM JSON payload to the webhook URL with simple retry attempts."""
    last_error = ""
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if 200 <= response.status_code < 300:
                return True, f"webhook_http_{response.status_code}"
            last_error = f"HTTP {response.status_code}"
            logger.warning(
                "CRM webhook attempt %d/%d returned status %d.",
                attempt,
                max_retries,
                response.status_code,
            )
        except Exception as exc:
            last_error = str(exc)
            logger.exception(
                "CRM webhook attempt %d/%d raised an error.",
                attempt,
                max_retries,
            )
    return False, f"webhook_failed: {last_error}"


def sync_lead(
    lead: dict[str, Any],
    mode: str | None = None,
    db_path: Path | str | None = None,
    csv_path: Path | str | None = None,
    webhook_url: str | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Push lead + score + appointment + summary JSON via 'local' or 'webhook' CRM mode."""
    selected_mode = (mode or CRM_MODE or "local").strip().lower()
    payload = build_crm_payload(lead, db_path=db_path)
    lead_id = payload["lead"].get("id")
    if lead_id is None:
        logger.warning("Skipping CRM sync because lead ID is missing.")
        return {"ok": False, "status": "skipped_no_lead_id", "payload": payload}

    payload_json = json.dumps(payload, ensure_ascii=False)
    target_csv = Path(csv_path) if csv_path is not None else CRM_EXPORT_PATH

    if selected_mode == "webhook":
        target_url = (webhook_url if webhook_url is not None else CRM_WEBHOOK_URL).strip()
        if not target_url:
            status = "failed_missing_webhook_url"
            log_crm_sync(int(lead_id), payload_json, status, db_path=db_path)
            logger.error("CRM webhook mode selected, but CRM_WEBHOOK_URL is not configured.")
            return {"ok": False, "status": status, "payload": payload}

        ok, detail = _post_webhook_with_retries(
            target_url,
            payload,
            max_retries=max_retries,
        )
        status = "synced_webhook" if ok else detail
        log_crm_sync(int(lead_id), payload_json, status, db_path=db_path)
        return {"ok": ok, "status": status, "payload": payload}

    status = "synced_local"
    try:
        _append_local_csv(payload, sync_status=status, csv_path=target_csv)
    except Exception:
        logger.exception("Failed writing CRM export CSV at %s.", target_csv)
        status = "failed_local_csv"
        log_crm_sync(int(lead_id), payload_json, status, db_path=db_path)
        return {"ok": False, "status": status, "payload": payload}

    log_crm_sync(int(lead_id), payload_json, status, db_path=db_path)
    logger.info("Synced lead_id=%s locally to CRM log and %s.", lead_id, target_csv)
    return {"ok": True, "status": status, "payload": payload}
