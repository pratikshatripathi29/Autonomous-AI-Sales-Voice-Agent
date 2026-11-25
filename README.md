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


