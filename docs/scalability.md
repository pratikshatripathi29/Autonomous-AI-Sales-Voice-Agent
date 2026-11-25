# Scalability Strategy (MVP to 100,000 Users)

This codebase is intentionally built as a single-node interview MVP. Scaling it to 100,000 users (and thousands of concurrent voice sessions) requires decoupling UI, state, database writes, vector search, and media streaming.

## Current Bottlenecks vs. 100,000-User Target

| Layer | Current MVP Implementation | Bottleneck at Scale | 100,000-User Architecture |
| :--- | :--- | :--- | :--- |
| **Frontend & Audio** | `app.py` (Streamlit `st.audio_input`, full-file upload per turn) | Streamlit reruns hold state in Python process memory; non-streaming audio adds round-trip latency | Replace UI with React/Next.js + WebRTC (`LiveKit` or raw WebRTC) for low-latency bidirectional audio streaming |
| **Backend API** | Monolithic Streamlit script execution | Cannot scale horizontally across workers cleanly | Stateless FastAPI / ASGI service behind an L7 load balancer with horizontal autoscaling |
| **Session & Memory** | `st.session_state` (`llm_history`, `lead_state`, `last_audio_hash`) | Lost on container restart; tied to single server sticky session | Store ephemeral call state in **Redis** keyed by `call_id` with TTL |
| **Relational DB** | SQLite (`data/db/sales_agent.db`) with `BEGIN IMMEDIATE` | Single-writer lock limits concurrent write throughput and prevents multi-node writes | Migrate `core/db.py` to **PostgreSQL** (`psycopg` + PgBouncer pool), read replicas for `core/analytics.py`, and row-level locking (`SELECT ... FOR UPDATE`) or exclusion constraints on `appointments` |
| **Vector Index (RAG)** | Local `FAISS` (`data/faiss_index`) loaded in process | Not shared across replicas; single global namespace without tenant isolation | Move to managed vector DB (**pgvector**, **Qdrant**, or **Pinecone**) with `tenant_id` metadata filters and async indexing workers |
| **Voice I/O (STT/TTS)** | Synchronous Groq Whisper REST + synchronous `gTTS` HTTP | `gTTS` has strict rate limits and no SLA; turn-by-turn HTTP buffering adds ~1.5–3s latency | Stream audio to Deepgram/Whisper streaming STT; stream Gemini tokens sentence-by-sentence into low-latency TTS (Cartesia, ElevenLabs, or Google Cloud TTS) |
| **Post-Call & CRM** | Synchronous calls on **End Call** (`summarize_call`, `sync_lead`) | Slows user response and risks lost webhooks on worker crash | Push post-call summarization and CRM webhook retries onto an async task queue (**Celery / Redis Streams / Cloud Tasks**) with dead-letter queues |

## Concurrency & Multi-Tenancy Plan
1. **Tenant Isolation**: Add `tenant_id` to `leads`, `calls`, `messages`, `appointments`, `crm_sync_log`, and vector metadata filters so one deployment serves multiple sales organizations safely.
2. **Idempotent Booking Locks**: In PostgreSQL, enforce slot uniqueness at the DB level (`EXCLUDE USING gist` or unique `(tenant_id, slot_start)` partial index `WHERE status = 'booked'`) so concurrent booking requests never double-book across API pods.
3. **Analytics Offloading**: Run `core/analytics.py` queries against a PostgreSQL read replica (or pre-aggregated materialized views refreshed every few minutes) so heavy dashboard queries never contend with live call writes.
4. **Provider Resilience**: Add per-tenant token-bucket rate limiters and circuit breakers around Groq and Gemini APIs, with automatic fallback models when a provider hits rate limits.

