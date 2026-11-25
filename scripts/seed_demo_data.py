"""Seed ~30 realistic leads, calls, messages, appointments, and CRM logs over 14 days."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path
import random
import sys

# Ensure project root is importable when running `python scripts/seed_demo_data.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.booking import IST_ZONE  # noqa: E402
from core.db import get_connection, init_db  # noqa: E402

SAMPLE_PROSPECTS: list[tuple[str, str, str, str, str]] = [
    ("Aarav Sharma", "+91-9810011223", "aarav@sharmatech.in", "Sharma Tech", "en"),
    ("Simran Kaur", "+91-9876543210", "simran@khalsalogistics.in", "Khalsa Logistics", "pa"),
    ("Rohan Verma", "+91-9822033445", "rohan@vermaclinic.in", "Verma Health Clinic", "hi"),
    ("Ananya Iyer", "+91-9845099887", "ananya@iyercloud.io", "Iyer Cloud Solutions", "en"),
    ("Vikramjit Gill", "+91-9815544332", "vikram@gillmotors.in", "Gill Motors", "pa"),
    ("Neha Gupta", "+91-9899012345", "neha@guptaretail.com", "Gupta Retail Group", "hi"),
    ("Carlos Mendez", "+1-415-555-0142", "carlos@andesfreight.com", "Andes Freight", "es"),
    ("Claire Dubois", "+33-6-12-34-56-78", "claire@ateliercloud.fr", "Atelier Cloud", "fr"),
    ("Kabir Malhotra", "+91-9811223344", "kabir@malhotrafintech.in", "Malhotra FinTech", "en"),
    ("Pooja Nair", "+91-9847011223", "pooja@nairhomes.in", "Nair Realty", "en"),
    ("Harpreet Sandhu", "+91-9872099111", "harpreet@sandhuagro.in", "Sandhu Agro", "pa"),
    ("Siddharth Joshi", "+91-9823055667", "sid@joshiedtech.in", "Joshi EdTech", "hi"),
    ("Meera Krishnan", "+91-9840077889", "meera@krishnancare.in", "Krishnan Diagnostics", "en"),
    ("Mateo Silva", "+1-305-555-0198", "mateo@silvasolar.io", "Silva Solar", "es"),
    ("Amelie Laurent", "+33-6-98-76-54-32", "amelie@laurentstudio.fr", "Laurent Studio", "fr"),
    ("Devendra Kulkarni", "+91-9820066554", "dev@kulkarnimfg.in", "Kulkarni Manufacturing", "hi"),
    ("Gurpreet Dhillon", "+91-9814088990", "gurpreet@dhillonexports.in", "Dhillon Exports", "pa"),
    ("Ishita Banerjee", "+91-9830011992", "ishita@banerjeelabs.in", "Banerjee Labs", "en"),
    ("Arjun Reddy", "+91-9849033221", "arjun@reddymed.in", "Reddy MedTech", "en"),
    ("Sunita Deshmukh", "+91-9822144556", "sunita@deshmukhinsure.in", "Deshmukh Insurance", "hi"),
    ("Manpreet Bains", "+91-9878011229", "manpreet@bainstravel.in", "Bains Travels", "pa"),
    ("Elena Morales", "+1-512-555-0177", "elena@moraleslegal.com", "Morales Legal", "es"),
    ("Lucas Moreau", "+33-6-45-67-89-01", "lucas@moreaudata.fr", "Moreau Data", "fr"),
    ("Tanvi Kapoor", "+91-9818099443", "tanvi@kapoorwellness.in", "Kapoor Wellness", "en"),
    ("Nikhil Bansal", "+91-9810345678", "nikhil@bansalinfra.in", "Bansal Infra", "hi"),
    ("Jasleen Grewal", "+91-9815098765", "jasleen@grewalfoods.in", "Grewal Foods", "pa"),
    ("Pranav Menon", "+91-9895067890", "pranav@menonanalytics.io", "Menon Analytics", "en"),
    ("Sofia Navarro", "+1-646-555-0134", "sofia@navarrodental.com", "Navarro Dental", "es"),
    ("Karan Oberoi", "+91-9811988776", "karan@oberoiestates.in", "Oberoi Estates", "en"),
    ("Divya Saxena", "+91-9839011225", "divya@saxenacommerce.in", "Saxena Commerce", "hi"),
]

OBJECTION_POOL: list[str] = [
    "Monthly pricing concern",
    "CRM integration effort",
    "Team onboarding time",
    "Regional language accuracy",
    "Need internal leadership approval",
]

PAIN_POINT_POOL: list[str] = [
    "Missed after-hours inbound calls",
    "Slow manual lead follow-up",
    "Unqualified leads wasting rep time",
    "No visibility into multilingual callers",
    "Manual calendar double-booking",
]

LANGUAGE_UTTERANCES: dict[str, tuple[str, str]] = {
    "en": (
        "Hi, we miss too many inbound sales calls after business hours. What plans do you offer?",
        "Our Starter plan is $49/month and Growth is $149/month with 24/7 AI voice coverage. What timeline are you targeting?",
    ),
    "hi": (
        "Namaste, hume apni sales team ke liye 24/7 AI voice receptionist chahiye. Pricing kya hai?",
        "Namaste! Hamara Starter plan $49 per month aur Growth plan $149 per month se shuru hota hai. Kya main aapka demo schedule karun?",
    ),
    "pa": (
        "Sat Sri Akal, saanu apne customers layi Punjabi te English voice agent chahida hai. Demo mil sakda hai?",
        "Sat Sri Akal! Bilkul, sada Growth plan multi-language support dinda hai. Tusi demo layi kehda din pasand karoge?",
    ),
    "es": (
        "Hola, necesitamos calificar leads automaticamente por telefono. Cual es el precio?",
        "Hola! Nuestro plan Starter cuesta $49 al mes y el plan Growth $149 al mes con sincronizacion CRM. Cuando le gustaria una demo?",
    ),
    "fr": (
        "Bonjour, nous perdons des appels entrants le soir. Proposez-vous un essai gratuit?",
        "Bonjour! Oui, nous offrons un essai gratuit de 14 jours avec 100 minutes incluses. Souhaitez-vous planifier une demonstration?",
    ),
}


def _clear_tables(db_path: Path | str | None = None) -> None:
    """Remove all existing rows in dependency order before re-seeding."""
    with closing(get_connection(db_path)) as conn:
        with conn:
            for table in ("crm_sync_log", "appointments", "messages", "calls", "leads"):
                conn.execute(f"DELETE FROM {table}")


def seed_demo_data(
    reset: bool = False,
    seed: int = 42,
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Populate SQLite with ~30 leads, calls, messages, and appointments over 14 days."""
    init_db(db_path=db_path)
    rng = random.Random(seed)

    with closing(get_connection(db_path)) as conn:
        existing_count = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        if existing_count >= len(SAMPLE_PROSPECTS) and not reset:
            return {
                "leads": 0,
                "calls": 0,
                "messages": 0,
                "appointments": 0,
                "crm_sync_log": 0,
            }

    if reset:
        _clear_tables(db_path=db_path)

    now_utc = datetime.now(timezone.utc)
    now_ist = datetime.now(IST_ZONE)
    # Build upcoming Mon-Fri slots so seeded appointments never clash with each other.
    future_slots: list[tuple[str, str]] = []
    day_cursor = now_ist.date() + timedelta(days=1)
    while len(future_slots) < 16:
        if day_cursor.weekday() < 5:
            for hour in (10, 11, 14, 16):
                slot_start = datetime.combine(day_cursor, time(hour, 0), tzinfo=IST_ZONE)
                slot_end = slot_start + timedelta(minutes=30)
                future_slots.append((slot_start.isoformat(), slot_end.isoformat()))
        day_cursor += timedelta(days=1)

    counts = {"leads": 0, "calls": 0, "messages": 0, "appointments": 0, "crm_sync_log": 0}
    slot_idx = 0

    with closing(get_connection(db_path)) as conn:
        with conn:
            for idx, (name, phone, email, company, lang) in enumerate(SAMPLE_PROSPECTS):
                days_ago = idx % 14
                call_start = now_utc - timedelta(days=days_ago, hours=rng.randint(1, 8))
                call_end = call_start + timedelta(minutes=rng.randint(3, 12))
                created_iso = call_start.isoformat()

                # Mix Hot (~35%), Warm (~40%), and Cold (~25%) leads cleanly.
                bucket = idx % 10
                if bucket in (0, 1, 2, 3):
                    status = "Hot"
                    score = rng.randint(72, 95)
                    sentiment_avg = round(rng.uniform(0.35, 0.85), 2)
                    outcome = "booked" if slot_idx < 10 else "qualified"
                elif bucket in (4, 5, 6, 7):
                    status = "Warm"
                    score = rng.randint(42, 68)
                    sentiment_avg = round(rng.uniform(-0.05, 0.45), 2)
                    outcome = "follow_up"
                else:
                    status = "Cold"
                    score = rng.randint(15, 38)
                    sentiment_avg = round(rng.uniform(-0.65, 0.05), 2)
                    outcome = "not_interested" if sentiment_avg < -0.2 else "other"

                lead_cursor = conn.execute(
                    """
                    INSERT INTO leads (name, phone, email, company, language, score, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (name, phone, email, company, lang, score, status, created_iso),
                )
                lead_id = int(lead_cursor.lastrowid)
                counts["leads"] += 1

                pain = rng.choice(PAIN_POINT_POOL)
                objection = rng.choice(OBJECTION_POOL)
                overall_sent = (
                    "positive"
                    if sentiment_avg > 0.1
                    else ("negative" if sentiment_avg < -0.1 else "neutral")
                )
                summary_payload = {
                    "summary": f"{name} from {company} inquired about AI voice coverage ({status} lead).",
                    "key_points": [f"Company: {company}", f"Primary language: {lang.upper()}"],
                    "customer_pain_points": [pain],
                    "objections": [objection],
                    "next_steps": ["Send pricing breakdown" if outcome != "booked" else "Conduct scheduled product demo"],
                    "outcome": outcome,
                    "overall_sentiment": overall_sent,
                    "recommended_action": "Follow up within 24 hours." if status != "Cold" else "Add to nurture list.",
                }

                call_cursor = conn.execute(
                    """
                    INSERT INTO calls (lead_id, started_at, ended_at, sentiment_avg, summary, outcome)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lead_id,
                        call_start.isoformat(),
                        call_end.isoformat(),
                        sentiment_avg,
                        json.dumps(summary_payload, ensure_ascii=False),
                        outcome,
                    ),
                )
                call_id = int(call_cursor.lastrowid)
                counts["calls"] += 1

                user_open, assistant_open = LANGUAGE_UTTERANCES[lang]
                turns = [
                    ("user", user_open, sentiment_avg),
                    ("assistant", assistant_open, None),
                    ("user", f"Our concern is {objection.lower()}, and we want to solve {pain.lower()}.", round(max(-1.0, sentiment_avg - 0.1), 2)),
                    ("assistant", "Understood. I have noted your details and our team can walk you through a tailored setup.", None),
                ]
                for turn_idx, (role, text, msg_sent) in enumerate(turns):
                    msg_ts = (call_start + timedelta(seconds=30 * (turn_idx + 1))).isoformat()
                    conn.execute(
                        """
                        INSERT INTO messages (call_id, role, text, language, sentiment, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (call_id, role, text, lang, msg_sent, msg_ts),
                    )
                    counts["messages"] += 1

                appt_obj = None
                if outcome == "booked" and slot_idx < len(future_slots):
                    slot_start, slot_end = future_slots[slot_idx]
                    slot_idx += 1
                    conn.execute(
                        """
                        INSERT INTO appointments (lead_id, call_id, slot_start, slot_end, status, created_at)
                        VALUES (?, ?, ?, ?, 'booked', ?)
                        """,
                        (lead_id, call_id, slot_start, slot_end, call_end.isoformat()),
                    )
                    counts["appointments"] += 1
                    appt_obj = {"slot_start": slot_start, "slot_end": slot_end, "status": "booked"}

                if status in ("Hot", "Warm"):
                    crm_payload = {
                        "lead": {"id": lead_id, "name": name, "phone": phone, "email": email, "company": company, "language": lang},
                        "score": score,
                        "status": status,
                        "appointment": appt_obj,
                        "summary": summary_payload["summary"],
                    }
                    conn.execute(
                        """
                        INSERT INTO crm_sync_log (lead_id, payload, status, created_at)
                        VALUES (?, ?, 'synced_local', ?)
                        """,
                        (lead_id, json.dumps(crm_payload, ensure_ascii=False), call_end.isoformat()),
                    )
                    counts["crm_sync_log"] += 1

    return counts


def main() -> None:
    """CLI entrypoint for seeding demo data into SQLite."""
    parser = argparse.ArgumentParser(description="Seed demo data for the AI Sales Voice Agent.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing SQLite tables before inserting demo records.",
    )
    args = parser.parse_args()

    result = seed_demo_data(reset=args.reset)
    print(f"Seed complete: {result}")


if __name__ == "__main__":
    main()

