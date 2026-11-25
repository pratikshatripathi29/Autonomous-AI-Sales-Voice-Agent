# Security & Privacy Review

An honest assessment of security controls currently implemented in this codebase and the hardening required before production deployment.

## Controls Implemented in the Current Codebase
1. **Secret Management**:
   - API keys (`GEMINI_API_KEY`, `GROQ_API_KEY`, `CRM_WEBHOOK_URL`) are loaded strictly from `.env` via `python-dotenv` (`core/config.py`) and never hardcoded.
   - `.env` is excluded in `.gitignore`.
   - `app.py` (`get_config_status()`) displays only boolean `OK` / `Missing` indicators in the sidebar, never secret values.
2. **SQL Injection Prevention**:
   - All SQL queries in `core/db.py` and `core/booking.py` use parameterized `?` placeholders.
   - Dynamic updates in `update_lead()` validate keys against an explicit `ALLOWED_LEAD_FIELDS` frozenset before constructing SQL `SET` clauses.
3. **File Upload Safety**:
   - Sidebar knowledge uploads (`_handle_knowledge_sidebar()` in `app.py`) restrict extensions to `.md` and `.txt` and sanitize filenames with `Path(uploaded.name).name` to prevent path-traversal writes outside `data/knowledge/`.
4. **FAISS Deserialization Scope**:
   - `core/rag.py` passes `allow_dangerous_deserialization=True` to `FAISS.load_local()` only for the locally built `data/faiss_index/index.pkl` generated from `data/knowledge/`. Untrusted `.pkl` uploads are never accepted.
5. **Graceful Error Handling**:
   - All external calls (`core/stt.py`, `core/tts.py`, `core/llm.py`, `core/rag.py`, `core/crm.py`, `core/summary.py`, `core/analytics.py`) are wrapped in `try/except` with Python `logging`, returning safe fallbacks without exposing raw stack traces or credentials to the UI.

## Known MVP Gaps (Required for Production)
- **Authentication & Authorization**: The Streamlit UI currently has no user login, session cookie signing, or role-based access control (RBAC). Anyone with network access to the port can view `Leads & Appointments`, `Analytics`, and upload knowledge docs.
- **PII Encryption at Rest**: `data/db/sales_agent.db` and `data/crm_export.csv` store customer names, phone numbers, emails, and full call transcripts in plaintext. Production requires disk/column encryption, PII redaction in logs, and adding `data/db/*.db`, `data/crm_export.csv`, and `data/faiss_index/` to `.gitignore`.
- **Prompt Injection Defense**: Both spoken user turns and uploaded `.md`/`.txt` documents enter the Gemini prompt context. Production should separate system instructions from untrusted content, enforce strict JSON schema validation, and scan uploaded docs.
- **Webhook Security (SSRF & Integrity)**: `CRM_WEBHOOK_URL` should be restricted to HTTPS allowlisted domains (blocking private/metadata IPs like `169.254.169.254`) and requests should include an HMAC-SHA256 signature header so receivers can verify authenticity.
- **Audio & Transcript Retention**: Configure explicit retention/deletion policies for stored `messages` and temporary audio buffers to comply with GDPR / local privacy regulations.

