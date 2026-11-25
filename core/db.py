"""SQLite database initialization and parameterized CRUD helpers."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any

from core.config import DB_PATH

logger = logging.getLogger(__name__)

# Explicit allowlist prevents dynamic column injection when building UPDATE statements.
ALLOWED_LEAD_FIELDS: frozenset[str] = frozenset(
    {"name", "phone", "email", "company", "language", "score", "status"}
)


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve and ensure the parent directory exists for the target SQLite file."""
    target = Path(db_path) if db_path is not None else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with Row factory and foreign key enforcement enabled."""
    target = _resolve_db_path(db_path)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    # SQLite disables foreign key constraints per connection unless explicitly enabled.
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """Create all required SQLite tables if they do not already exist."""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT,
        email TEXT,
        company TEXT,
        language TEXT NOT NULL DEFAULT 'en',
        score INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'new',
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id INTEGER,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        sentiment_avg REAL,
        summary TEXT,
        outcome TEXT,
        FOREIGN KEY (lead_id) REFERENCES leads (id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        text TEXT NOT NULL,
        language TEXT NOT NULL DEFAULT 'en',
        sentiment REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (call_id) REFERENCES calls (id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id INTEGER NOT NULL,
        call_id INTEGER,
        slot_start TEXT NOT NULL,
        slot_end TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'booked',
        created_at TEXT NOT NULL,
        FOREIGN KEY (lead_id) REFERENCES leads (id) ON DELETE CASCADE,
        FOREIGN KEY (call_id) REFERENCES calls (id) ON DELETE SET NULL
    );

    CREATE TABLE IF NOT EXISTS crm_sync_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id INTEGER NOT NULL,
        payload TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (lead_id) REFERENCES leads (id) ON DELETE CASCADE
    );
    """
    # sqlite3's connection context manager commits/rolls back transactions but does not
    # close the underlying file handle, so closing() prevents file-lock leaks on Windows.
    with closing(get_connection(db_path)) as conn:
        with conn:
            conn.executescript(schema_sql)
    logger.info("Initialized SQLite database at %s", _resolve_db_path(db_path))


# --- Leads CRUD ---


def create_lead(
    name: str,
    phone: str | None = None,
    email: str | None = None,
    company: str | None = None,
    language: str = "en",
    score: int = 0,
    status: str = "new",
    db_path: Path | str | None = None,
) -> int:
    """Insert a new lead record and return its generated primary key ID."""
    created_at = _utc_now()
    query = """
        INSERT INTO leads (name, phone, email, company, language, score, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with closing(get_connection(db_path)) as conn:
        with conn:
            cursor = conn.execute(
                query,
                (name, phone, email, company, language, score, status, created_at),
            )
            return int(cursor.lastrowid)


def get_lead(lead_id: int, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Fetch a single lead by ID, or return None if not found."""
    query = "SELECT * FROM leads WHERE id = ?"
    with closing(get_connection(db_path)) as conn:
        row = conn.execute(query, (lead_id,)).fetchone()
        return dict(row) if row is not None else None


def list_leads(
    status: str | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List leads ordered by newest first, optionally filtered by status."""
    with closing(get_connection(db_path)) as conn:
        if status is not None:
            rows = conn.execute(
                "SELECT * FROM leads WHERE status = ? ORDER BY id DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM leads ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


def update_lead(
    lead_id: int,
    db_path: Path | str | None = None,
    **fields: Any,
) -> bool:
    """Update allowed fields on a lead record and return True if a row was modified."""
    if not fields:
        return False

    invalid_fields = set(fields.keys()) - ALLOWED_LEAD_FIELDS
    if invalid_fields:
        raise ValueError(f"Unsupported lead fields: {sorted(invalid_fields)}")

    assignments = ", ".join(f"{column} = ?" for column in fields)
    values = list(fields.values()) + [lead_id]
    query = f"UPDATE leads SET {assignments} WHERE id = ?"

    with closing(get_connection(db_path)) as conn:
        with conn:
            cursor = conn.execute(query, values)
            return cursor.rowcount > 0


# --- Calls CRUD ---


def create_call(
    lead_id: int | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Start a new call record and return its generated primary key ID."""
    started_at = _utc_now()
    query = "INSERT INTO calls (lead_id, started_at) VALUES (?, ?)"
    with closing(get_connection(db_path)) as conn:
        with conn:
            cursor = conn.execute(query, (lead_id, started_at))
            return int(cursor.lastrowid)


def get_call(call_id: int, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Fetch a single call record by ID, or return None if not found."""
    query = "SELECT * FROM calls WHERE id = ?"
    with closing(get_connection(db_path)) as conn:
        row = conn.execute(query, (call_id,)).fetchone()
        return dict(row) if row is not None else None


def list_calls(
    lead_id: int | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List calls ordered by newest first, optionally filtered by lead_id."""
    with closing(get_connection(db_path)) as conn:
        if lead_id is not None:
            rows = conn.execute(
                "SELECT * FROM calls WHERE lead_id = ? ORDER BY id DESC",
                (lead_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM calls ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


def end_call(
    call_id: int,
    sentiment_avg: float | None = None,
    summary: str | None = None,
    outcome: str | None = None,
    db_path: Path | str | None = None,
) -> bool:
    """Mark a call as ended and store its sentiment average, summary, and outcome."""
    ended_at = _utc_now()
    query = """
        UPDATE calls
        SET ended_at = ?, sentiment_avg = ?, summary = ?, outcome = ?
        WHERE id = ?
    """
    with closing(get_connection(db_path)) as conn:
        with conn:
            cursor = conn.execute(
                query,
                (ended_at, sentiment_avg, summary, outcome, call_id),
            )
            return cursor.rowcount > 0


def assign_call_lead(
    call_id: int,
    lead_id: int,
    db_path: Path | str | None = None,
) -> bool:
    """Link an existing call record to a newly created or matched lead ID."""
    query = "UPDATE calls SET lead_id = ? WHERE id = ?"
    with closing(get_connection(db_path)) as conn:
        with conn:
            cursor = conn.execute(query, (lead_id, call_id))
            return cursor.rowcount > 0


# --- Messages CRUD ---


def add_message(
    call_id: int,
    role: str,
    text: str,
    language: str = "en",
    sentiment: float | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Persist a single conversation message for a call and return its ID."""
    created_at = _utc_now()
    query = """
        INSERT INTO messages (call_id, role, text, language, sentiment, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with closing(get_connection(db_path)) as conn:
        with conn:
            cursor = conn.execute(
                query,
                (call_id, role, text, language, sentiment, created_at),
            )
            return int(cursor.lastrowid)


def get_call_messages(
    call_id: int,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return all messages for a call in chronological order."""
    query = "SELECT * FROM messages WHERE call_id = ? ORDER BY id ASC"
    with closing(get_connection(db_path)) as conn:
        rows = conn.execute(query, (call_id,)).fetchall()
        return [dict(row) for row in rows]


# --- Appointments CRUD ---


def create_appointment(
    lead_id: int,
    slot_start: str,
    slot_end: str,
    call_id: int | None = None,
    status: str = "booked",
    db_path: Path | str | None = None,
) -> int:
    """Create an appointment slot for a lead and return its generated ID."""
    created_at = _utc_now()
    query = """
        INSERT INTO appointments (lead_id, call_id, slot_start, slot_end, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with closing(get_connection(db_path)) as conn:
        with conn:
            cursor = conn.execute(
                query,
                (lead_id, call_id, slot_start, slot_end, status, created_at),
            )
            return int(cursor.lastrowid)


def get_booked_slots(
    status: str = "booked",
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return appointment slots matching the given status, ordered by slot_start."""
    query = """
        SELECT id, lead_id, call_id, slot_start, slot_end, status, created_at
        FROM appointments
        WHERE status = ?
        ORDER BY slot_start ASC
    """
    with closing(get_connection(db_path)) as conn:
        rows = conn.execute(query, (status,)).fetchall()
        return [dict(row) for row in rows]


def list_appointments(
    lead_id: int | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List appointments ordered by slot_start, optionally filtered by lead_id."""
    with closing(get_connection(db_path)) as conn:
        if lead_id is not None:
            rows = conn.execute(
                "SELECT * FROM appointments WHERE lead_id = ? ORDER BY slot_start ASC",
                (lead_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM appointments ORDER BY slot_start ASC"
            ).fetchall()
        return [dict(row) for row in rows]


def update_appointment_status(
    appointment_id: int,
    status: str,
    db_path: Path | str | None = None,
) -> bool:
    """Update the status of an appointment and return True if a row was modified."""
    query = "UPDATE appointments SET status = ? WHERE id = ?"
    with closing(get_connection(db_path)) as conn:
        with conn:
            cursor = conn.execute(query, (status, appointment_id))
            return cursor.rowcount > 0


# --- CRM Sync Log CRUD ---


def log_crm_sync(
    lead_id: int,
    payload: str,
    status: str,
    db_path: Path | str | None = None,
) -> int:
    """Record a CRM sync attempt payload and status, returning its generated ID."""
    created_at = _utc_now()
    query = """
        INSERT INTO crm_sync_log (lead_id, payload, status, created_at)
        VALUES (?, ?, ?, ?)
    """
    with closing(get_connection(db_path)) as conn:
        with conn:
            cursor = conn.execute(query, (lead_id, payload, status, created_at))
            return int(cursor.lastrowid)


def get_crm_sync_logs(
    lead_id: int | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List CRM sync log entries ordered by newest first, optionally filtered by lead_id."""
    with closing(get_connection(db_path)) as conn:
        if lead_id is not None:
            rows = conn.execute(
                "SELECT * FROM crm_sync_log WHERE lead_id = ? ORDER BY id DESC",
                (lead_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM crm_sync_log ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

