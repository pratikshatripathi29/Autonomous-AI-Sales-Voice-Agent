# Autonomous AI Sales Voice Agent

Modular Python MVP for an autonomous AI sales voice agent built with Streamlit, Groq Whisper (STT), Google Gemini (LLM), gTTS (TTS), LangChain + FAISS (RAG), SQLite, and Plotly.

## Stage 1: Project Scaffold & SQLite Layer

### Structure
- `app.py`: Streamlit UI shell with sidebar navigation (`Voice Agent`, `Leads & Appointments`, `Analytics`) and startup database initialization.
- `core/config.py`: Loads `.env` via `python-dotenv`, exposes configuration constants, and configures logging.
- `core/db.py`: Pure `sqlite3` persistence layer (no ORM) managing `leads`, `calls`, `messages`, `appointments`, and `crm_sync_log` with parameterized queries.
- `core/*.py`: Module stubs reserved for upcoming stages (`stt.py`, `tts.py`, `llm.py`, `rag.py`, `memory.py`, `leads.py`, `booking.py`, `crm.py`, `summary.py`, `analytics.py`).
- `data/knowledge/`, `data/db/`: Local storage directories for knowledge documents and the SQLite database.
- `tests/test_db.py`: Unit tests for database schema and CRUD helpers.

### Setup & Verification
1. Install dependencies (ideally in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
2. Run database tests:
   ```bash
   python -m unittest discover -s tests -v
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Stage 2: Voice I/O (STT & TTS Echo Loop)

### Modules Added / Updated
- `core/stt.py`: `transcribe(audio_bytes)` calls Groq Whisper (`whisper-large-v3-turbo`) with `response_format="verbose_json"` and automatic language detection (no forced language) for English, Hindi, Punjabi, and more, with empty-audio and error fallbacks.
- `core/tts.py`: `speak(text, lang_code)` synthesizes MP3 bytes into an in-memory `BytesIO` buffer using `gTTS`, mapping Whisper language names/codes (`english` -> `en`, `hindi` -> `hi`, `punjabi` -> `pa`) with fallback to `en`.
- `app.py`: Voice Agent page records microphone audio via `st.audio_input`, caches the SHA-256 hash in `st.session_state` to prevent duplicate API calls across reruns, displays the transcript and detected language, and echoes it back via `st.audio(..., autoplay=True)`.
- `tests/test_voice_io.py`: Mocked unit tests for STT auto-detection, error fallbacks, language mapping, and in-memory TTS generation.

## Stage 3: Conversational Brain & Rolling Memory

### Modules Added / Updated
- `core/config.py`: Added `COMPANY_NAME`, `COMPANY_PRODUCTS`, and `MEMORY_WINDOW_SIZE` (default `10`).
- `core/llm.py`: Uses `langchain_google_genai.ChatGoogleGenerativeAI` (`GEMINI_MODEL`) to implement `generate_turn(...)` returning structured JSON (`reply`, `language`, `sentiment`, `sentiment_score`, `intent`, `extracted`, `wants_booking`, `preferred_time`) with code-fence stripping and plain-text fallback, plus `summarize_messages(...)` for memory compression.
- `core/memory.py`: Persists every message to SQLite `messages`, maintains a rolling 10-message LLM context window with automatic summarization of older turns into `memory_summary`, and merges newly extracted lead fields into `st.session_state["lead_state"]`.
- `app.py`: Wired full `record -> STT -> generate_turn -> memory/SQLite -> TTS` loop on the `Voice Agent` page, displaying chat messages with sentiment badges, intent tags, captured lead state, and rolling memory summary.
- `tests/test_brain.py`: Unit tests for JSON parsing/clamping, fallback behavior, rolling memory summarization, lead state merging, and SQLite message persistence.

## Stage 4: RAG Knowledge Base (LangChain + FAISS)

### Modules Added / Updated
- `core/config.py`: Added `FAISS_INDEX_DIR` (`data/faiss_index`).
- `core/rag.py`: Implements `build_index()` (loads `.md`/`.txt` from `data/knowledge/` via `DirectoryLoader`/`TextLoader`, chunks via `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)`, embeds with `GoogleGenerativeAIEmbeddings`, saves FAISS index locally), `load_index()` (loads existing index or builds on demand), `retrieve(query, k=3)` returning `(text, source, score)` tuples, and `format_context()`.
- `data/knowledge/`: Added starter docs `products_and_pricing.md` and `faq_and_support.md`.
- `core/llm.py`: Added `rewrite_search_query(user_text, history, last_intent)` so short follow-ups like `"how much?"` expand with the previous user turn and intent before RAG retrieval, and injected retrieved context into `generate_turn()`.
- `core/memory.py`: Tracks `last_intent` and stores retrieved `sources` on assistant transcript entries.
- `app.py`: Added sidebar knowledge-doc file uploader (`.md`/`.txt`) + **Rebuild knowledge base** button, wired query rewriting + RAG retrieval into each voice turn, and rendered a **Sources used** expander under each AI reply.
- `tests/test_rag.py`: Unit tests for FAISS index build/load/retrieval, context formatting, and query rewriting.

## Stage 5: Lead Scoring, Sentiment Tracking & Live Panel

### Modules Added / Updated
- `core/leads.py`: Transparent rule-based `score_lead(lead_state, sentiment_history, intents) -> (score, status)` (`Hot >= 70`, `Warm 40-69`, `Cold < 40`), `get_score_breakdown(...)` returning human-readable scoring reasons, and `sync_lead_to_db(...)` creating a SQLite lead on first captured name/phone/email and updating it on subsequent turns.
- `core/db.py`: Added `assign_call_lead(call_id, lead_id)` to link active calls when a lead is created mid-call.
- `core/memory.py`: Tracks `sentiment_history`, `intents`, `wants_booking`, `lead_id`, `lead_score`, `lead_status`, and `lead_breakdown` per call session.
- `core/analytics.py`: Added Plotly `build_score_gauge(score, status)` and `build_sentiment_trend_figure(sentiment_history)` builders.
- `app.py`: Added a right-hand **Live Lead Intelligence** panel on the `Voice Agent` page displaying the Hot/Warm/Cold badge, Plotly score gauge, score breakdown, collected lead card, and turn-by-turn Plotly sentiment trend line.
- `tests/test_leads.py`: Unit tests covering scoring rules, negative sentiment penalty, 0-100 clamping, SQLite lead create/update sync, and Plotly chart generation.


