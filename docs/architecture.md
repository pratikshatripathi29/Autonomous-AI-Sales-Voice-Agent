# System Architecture

The Autonomous AI Sales Voice Agent separates presentation (`app.py`) from single-concern domain modules inside `core/`.

## Module Responsibilities

| Module | Concern | Key External / Local Dependency |
| :--- | :--- | :--- |
| `app.py` | Streamlit UI, sidebar KB controls, audio deduplication | `streamlit` |
| `core/config.py` | Environment loading (`.env`), constants, logging setup | `python-dotenv`, `logging` |
| `core/db.py` | Schema creation (`init_db`) and parameterized CRUD | `sqlite3` (`data/db/sales_agent.db`) |
| `core/stt.py` | Audio transcription & auto language detection | Groq Whisper (`whisper-large-v3-turbo`) |
| `core/tts.py` | Language mapping (`en`, `hi`, `pa`, `es`, `fr`) & MP3 synthesis | `gTTS` (in-memory `BytesIO`) |
| `core/rag.py` | Document loading, chunking, FAISS index build/load, top-k retrieval | LangChain, FAISS, `gemini-embedding-001` |
| `core/llm.py` | Contextual query rewriting, JSON-mode turn generation, summarization | `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) |
| `core/memory.py` | Rolling `N=10` window + summary compression, `lead_state` merge | `core/db.py`, `core/llm.py` |
| `core/leads.py` | Rule-based 0–100 scoring (`Hot`/`Warm`/`Cold`) and SQLite lead sync | `core/db.py` |
| `core/booking.py` | `Asia/Kolkata` slot parsing, `BEGIN IMMEDIATE` conflict check, `.ics` export | `dateparser`, `zoneinfo`, `core/db.py` |
| `core/crm.py` | Pluggable CRM sync (`local` CSV/log or `webhook` POST with retries) | `requests`, `csv`, `core/db.py` |
| `core/summary.py` | Post-call structured JSON summary persisted to `calls` | `core/llm.py`, `core/db.py` |
| `core/analytics.py` | KPI calculation, summary aggregation, Plotly charts, AI insights | `plotly`, `core/db.py`, `core/llm.py` |

## High-Level System Architecture

```mermaid
flowchart TB
    User["User (Browser Mic & Speaker)"] --> UI["app.py (Streamlit UI)"]

    subgraph VoicePipeline["Voice & Conversational Pipeline"]
        UI -->|"audio bytes"| STT["core/stt.py (Groq Whisper STT)"]
        UI -->|"user_text + history"| LLM["core/llm.py (Gemini LLM)"]
        LLM <-->|"rewrite query / retrieve k=3"| RAG["core/rag.py (LangChain + FAISS)"]
        RAG <-->|"chunks & embeddings"| KB["data/knowledge/ + data/faiss_index/"]
        UI <-->|"rolling N=10 window + lead_state"| MEM["core/memory.py"]
        UI -->|"reply_text + lang"| TTS["core/tts.py (gTTS MP3)"]
        TTS -->|"MP3 audio"| UI
    end

    subgraph BusinessAndStorage["Qualification, Scheduling & Persistence"]
        UI -->|"score_lead & sync"| LEADS["core/leads.py"]
        UI -->|"wants_booking + preferred_time"| BOOK["core/booking.py (dateparser + ICS)"]
        UI -->|"End Call"| SUM["core/summary.py (Gemini JSON Summary)"]
        MEM --> DB[("core/db.py -> SQLite")]
        LEADS --> DB
        BOOK --> DB
        SUM --> DB
        BOOK -->|"post-booking sync"| CRM["core/crm.py (Local CSV / Webhook)"]
        SUM -->|"end-of-call sync"| CRM
        CRM --> DB
        CRM -->|"local mode"| CSV["data/crm_export.csv"]
        CRM -->|"webhook mode"| HOOK["External CRM Webhook"]
    end

    subgraph Dashboard["Reporting & Analytics"]
        DB --> ANALYTICS["core/analytics.py (KPIs + Plotly + AI Insights)"]
        ANALYTICS --> UI
    end
```

## Single Voice Turn Data Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as app.py (Streamlit)
    participant STT as core/stt.py (Groq)
    participant LLM as core/llm.py (Gemini)
    participant RAG as core/rag.py (FAISS)
    participant MEM as core/memory.py
    participant DB as core/db.py (SQLite)
    participant LEAD as core/leads.py
    participant BOOK as core/booking.py
    participant CRM as core/crm.py
    participant TTS as core/tts.py (gTTS)

    User->>UI: Speak into st.audio_input
    UI->>UI: Check SHA-256 last_audio_hash (skip if rerun)
    UI->>STT: transcribe(audio_bytes)
    STT-->>UI: {text, language}
    UI->>LLM: rewrite_search_query(text, history, last_intent)
    LLM-->>UI: standalone search_query
    UI->>RAG: retrieve(search_query, k=3)
    RAG-->>UI: top-k (text, source, score)
    UI->>LLM: generate_turn(text, history, rag_context, lead_state, lang, summary)
    LLM-->>UI: JSON {reply, sentiment, score, intent, extracted, wants_booking, preferred_time}
    UI->>MEM: record_message(user) & update_session_lead_state(extracted)
    MEM->>DB: INSERT INTO messages (and summarize overflow if > 10)
    UI->>LEAD: score_lead() & sync_lead_to_db()
    LEAD->>DB: INSERT or UPDATE leads + link calls.lead_id
    opt wants_booking is true
        UI->>BOOK: process_booking_request(lead_id, call_id, lead_state, preferred_time)
        BOOK->>DB: BEGIN IMMEDIATE -> check conflict -> INSERT appointments
        BOOK-->>UI: confirmation or 3 nearest alternatives (+ .ics if booked)
        opt slot booked
            UI->>CRM: sync_lead(lead + score + appointment)
            CRM->>DB: INSERT INTO crm_sync_log (+ data/crm_export.csv or POST webhook)
        end
    end
    UI->>MEM: record_message(assistant, sources)
    MEM->>DB: INSERT INTO messages
    UI->>TTS: speak(reply_text, reply_lang)
    TTS-->>UI: MP3 bytes (BytesIO)
    UI-->>User: Render transcript, sentiment badge, sources, live gauge & autoplay MP3
```

