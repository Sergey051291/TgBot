# NewsDigestBot — Telegram RAG + Digests

**Telegram bot that collects posts from monitored channels, indexes them in a vector database, and provides RAG-based Q&A plus automated digest generation.**

The bot monitors selected Telegram chats/channels, stores content locally, answers user questions using retrieval-augmented generation (RAG), and publishes structured digests via Telegraph.

---

## Screenshots

> Add screenshots to `docs/screenshots/` and uncomment the lines below.

<!--
![Bot menu](docs/screenshots/01-bot-menu.png)
![RAG answer](docs/screenshots/02-rag-answer.png)
![Digest link](docs/screenshots/03-digest.png)
-->

---

## Features

| Feature | Description |
|---------|-------------|
| **Channel monitoring** | Track posts from configured Telegram channels/chats |
| **RAG Q&A** | Ask questions about indexed content using local LLM |
| **Digests** | Generate 7-day and 30-day digest links via Telegraph |
| **Dual auth mode** | Bot token (aiogram) or user session (Telethon) |
| **Vector backends** | ChromaDB (local) or Astra DB (cloud) |
| **Scheduled digests** | Optional cron-based auto-digest posting |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Bot framework** | aiogram 3.x |
| **Telegram client** | Telethon |
| **LLM / Embeddings** | Ollama (llama3, mxbai-embed-large) |
| **Vector DB** | ChromaDB / Astra DB |
| **RAG** | LangChain + custom pipeline |
| **Publishing** | Telegraph API |
| **Scheduler** | APScheduler |

---

## Architecture

```
Telegram channels/chats
        │
        ▼
┌───────────────────┐
│  bot.py           │  aiogram handlers + Telethon watcher
│  (aiogram/Telethon)│
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐     ┌──────────────────┐
│  rag_system.py    │────▶│  Chroma / Astra  │
│  database.py      │     │  vector store    │
└─────────┬─────────┘     └──────────────────┘
          │
          ▼
┌───────────────────┐     ┌──────────────────┐
│  Ollama (local)   │     │  digest_generator│
│  LLM + embeddings │     │  → Telegraph     │
└───────────────────┘     └──────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (`ollama serve`)
- Models: `llama3`, `mxbai-embed-large:latest`
- Telegram Bot Token + API ID/HASH from [my.telegram.org](https://my.telegram.org)

### Installation

```powershell
git clone https://github.com/Sergey051291/TgBot.git
cd TgBot

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```powershell
copy .env.example .env
# Edit .env with your tokens, API credentials, and monitored channels
```

Key variables in `.env`:
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`
- `AUTH_MODE=bot` or `user`
- `VECTOR_BACKEND=chroma` or `astra`
- `MONITORED_CHANNELS` — JSON map of channel IDs to categories
- `OLLAMA_MODEL`, `OLLAMA_EMBEDDINGS`

### Run

```powershell
python bot.py
```

---

## Project Structure

```
TgBot/
├── bot.py                 # Main bot entry point
├── config.py              # Configuration loader
├── database.py            # Data persistence layer
├── rag_system.py          # RAG pipeline (search + generation)
├── digest_generator.py    # Telegraph digest builder
├── connect_astra.py       # Astra DB connection helper
├── script/                # Utility scripts (channel IDs, checks)
├── .env.example           # Environment template
├── requirements.txt
└── docs/screenshots/
```

---

## Utility Scripts

Located in `script/`:
- `get_channel_ids.py` — resolve channel IDs
- `check_all.py` — verify configuration
- `whoami.py` — check Telethon session
- `quick_fetch.py` — test message fetching

---

## Notes

- **Never commit** `.env`, `*.session`, or `chroma_storage/` — they contain secrets and local data.
- If you change the embedding model, re-index: clear `chroma_storage/` or run `/reindex`.
- Switch to Astra DB by setting `VECTOR_BACKEND=astra` in `.env`.

---

## Author

Personal project — Telegram news aggregation with local LLM and RAG.

**Stack:** Python · aiogram · Telethon · Ollama · ChromaDB · LangChain
