# Cost Model & Optimization

This document breaks down the real cost drivers of the stack implemented in this repository and how to control spend as usage grows.

## Per-Component Cost Drivers in This Codebase

| Component | Implementation | Pricing Unit | Where Called |
| :--- | :--- | :--- | :--- |
| **STT** | Groq Whisper (`whisper-large-v3-turbo`) in `core/stt.py` | Per audio second / minute (minimum billing window per request) | Once per unique voice turn (`last_audio_hash` prevents duplicate calls on Streamlit reruns) |
| **LLM (Turn + Rewrite + Summary)** | Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai` | Input tokens + Output tokens | `rewrite_search_query()` (skipped on turn 1), `generate_turn()`, `summarize_messages()` (when > 10 turns), `summarize_call()`, `generate_business_insights()` |
| **Embeddings** | Google Gemini (`gemini-embedding-001`) in `core/rag.py` | Input tokens | `build_index()` (on KB rebuild/upload) and `retrieve()` (1 short query per turn) |
| **TTS** | `gTTS` in `core/tts.py` | Free (unofficial Google Translate endpoint; no SLA) | Once per assistant reply (`speak()`) |
| **Storage & UI** | Local SQLite + Local FAISS + Streamlit | Compute / disk only (no per-query DB fee) | Persistent across calls |

## Call Lifecycle API Footprint (Typical 6-Turn Call)
For a 6-turn sales call (6 user turns + 6 assistant replies + End Call):
- **Groq STT**: 6 short audio clips (typically 5–15s each = ~1 minute total audio).
- **Gemini Embeddings**: 6 short query embeddings (`~10–25` tokens each); documents are embedded **once** at index build time and cached locally in `data/faiss_index/`.
- **Gemini LLM (`gemini-2.5-flash`)**:
  - `rewrite_search_query()`: 5 calls (skipped on turn 1).
  - `generate_turn()`: 6 calls (bounded by `MEMORY_WINDOW_SIZE=10` + 3 retrieved chunks of `chunk_size=500` chars).
  - `summarize_messages()`: 0 calls until conversation exceeds 10 messages (2 turns per round = after turn 5).
  - `summarize_call()`: 1 call on **End Call**.
- **gTTS**: 6 in-memory MP3 syntheses (`< 3` sentences each).

## Cost-Optimization Levers
1. **Eliminate Extra Query-Rewrite LLM Calls**: Currently `rewrite_search_query()` calls Gemini whenever `history` has a prior user turn. Adding a word-count/pronoun heuristic (only rewrite when `len(user_text.split()) <= 6` or when pronouns like `"it"`, `"that"`, `"much"` appear) can cut rewrite LLM calls by ~50–70%.
2. **Cache Common TTS Clips**: Standard greetings, booking prompts, and fallback lines (*"Let me connect you with a human specialist"*) can be cached by `(text, lang)` hash in memory or object storage.
3. **Keep Context Bounded**: `MEMORY_WINDOW_SIZE=10` and `chunk_size=500` (`k=3`) already prevent prompt bloat on long calls.
4. **Production TTS Upgrade**: Replacing `gTTS` with a commercial streaming TTS (Google Cloud TTS, Cartesia, or ElevenLabs) adds per-character cost, which should be budgeted alongside telephony/WebRTC minutes in production.

