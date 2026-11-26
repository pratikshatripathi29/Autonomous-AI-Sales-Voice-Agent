# Autonomous AI Sales Voice Agent
https://autonomous-ai-sales-voice-agent-ptt.streamlit.app/
An end-to-end, modular Python MVP for an autonomous multilingual AI sales voice agent built with **Streamlit**, **Groq Whisper (STT)**, **Google Gemini (LLM & Embeddings)**, **LangChain + FAISS (RAG)**, **gTTS (TTS)**, **SQLite**, and **Plotly**.

## Architecture & Documentation
- [Architecture & Sequence Diagrams](docs/architecture.md)
- [Scalability Strategy (100,000 Users)](docs/scalability.md)
- [Security & Privacy Review](docs/security.md)
- [Cost Model & Optimization](docs/costs.md)

## Project Structure
- `app.py` — Streamlit UI only (3 pages: **Voice Agent**, **Leads & Appointments**, **Analytics**).
- `core/` — Single-concern modules:
  - `config.py`: `.env` configuration & logging
  - `db.py`: Pure `sqlite3` schema & parameterized CRUD (`leads`, `calls`, `messages`, `appointments`, `crm_sync_log`)
  - `stt.py`: Groq Whisper auto-detect speech-to-text (`en`, `hi`, `pa`, etc.)
  - `tts.py`: `gTTS` in-memory MP3 synthesis with language mapping
  - `rag.py`: LangChain + FAISS local index build/load and top-k retrieval
  - `llm.py`: Gemini JSON-mode turn generation, contextual query rewriting, and summarization
  - `memory.py`: Rolling `N=10` window + memory summary compression + `lead_state` merge
  - `leads.py`: Transparent rule-based 0–100 lead scoring (`Hot`/`Warm`/`Cold`) & SQLite sync
  - `booking.py`: `Asia/Kolkata` slot parser (`dateparser`), `BEGIN IMMEDIATE` double-booking guard, `.ics` export
  - `crm.py`: Pluggable CRM sync (`local` SQLite/CSV export or `webhook` POST with retries)
  - `summary.py`: Structured post-call Gemini JSON summary saved to `calls`
  - `analytics.py`: KPIs, 6 Plotly dashboard charts, live gauge/trend widgets, and AI business insights
- `data/knowledge/` — Starter `.md` product & FAQ docs for RAG.
- `scripts/seed_demo_data.py` — Seeds 30 realistic leads/calls/messages/appointments across 14 days.
- `tests/` — Unit and `pytest` tests (`test_core.py`, `test_db.py`, `test_voice_io.py`, `test_brain.py`, `test_rag.py`, `test_leads.py`, `test_booking_crm.py`, `test_summary_analytics.py`).

## Setup & Installation

1. Create and activate a Python virtual environment (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install pinned dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Configure `.env` in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   GEMINI_MODEL=gemini-2.5-flash
   EMBEDDING_MODEL=gemini-embedding-001
   # Optional overrides:
   # COMPANY_NAME=NovaCloud AI
   # CRM_MODE=local
   # CRM_WEBHOOK_URL=https://example.com/webhook
   ```

## Running the App & Tests

1. **Seed 14 days of realistic demo data** (optional, makes dashboards immediately rich):
   ```powershell
   python scripts/seed_demo_data.py --reset
   ```
2. **Run the full test suite**:
   ```powershell
   pytest -v
   ```
3. **Launch Streamlit**:
   ```powershell
   streamlit run app.py
   ```

## Live Interview Demo Script (3–5 Minutes)

1. **Pre-demo check (30s)**:
   - Run `python scripts/seed_demo_data.py --reset` and `streamlit run app.py`.
   - Confirm sidebar **Environment Status** shows `Gemini Key: OK` and `Groq Key: OK`.
   - Click **Rebuild knowledge base** in the sidebar to verify `data/faiss_index` builds from `data/knowledge/`.
2. **Live Multilingual Voice + RAG Turn (90s)**:
   - On **Voice Agent**, record: *"Hi, I'm Aarav from Acme Tech. Can you tell me about your AI Voice Receptionist?"*
   - Show the transcript, sentiment badge, intent tag, auto-played MP3 reply, and expand **Sources used** to show retrieved FAISS chunks.
   - Follow up with a short query: *"How much does it cost?"* — point out contextual query rewriting retrieving `$49/month` / `$149/month` accurately.
   - (Optional) Speak a turn in Hindi or Punjabi to demonstrate Whisper auto-detection + matching `gTTS` reply.
3. **Lead Scoring & Appointment Booking (60s)**:
   - Provide urgency/budget/contact: *"Our budget is $200 a month and we want to launch next month. Book a demo tomorrow at 3 PM, my email is aarav@acme.io."*
   - Highlight the right-hand **Live Lead Intelligence** panel jumping to **Hot Lead**, the Plotly gauge, the transparent **Score Breakdown**, the confirmed slot, and the **.ics** calendar download button.
4. **End Call & CRM Sync (30s)**:
   - Click **End Call** to generate the structured **Post-Call Summary Card** (pain points, objections, next steps, recommended action) and auto-sync to `crm_sync_log` + `data/crm_export.csv`.
5. **Leads & Analytics Dashboards (45s)**:
   - Open **Leads & Appointments** to filter Hot leads, download CSVs, and inspect the CRM Sync Log.
   - Open **Analytics** to show the 6 KPI cards, conversion funnel, language distribution, top objections/pain points, and click **AI business insights** for live Gemini recommendations.
