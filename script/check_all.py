# check_all.py
import os
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ---------------- 1) Проверка .env ----------------
print("=== 1) Проверка .env ===")
print("OK .env загружен.")
print("TELEGRAM_API_ID:", Config.TELEGRAM_API_ID)
print("ASTRA_DB_ENDPOINT:", bool(Config.ASTRA_DB_ENDPOINT))
print("OLLAMA_MODEL:", Config.OLLAMA_MODEL)
print("OLLAMA_EMBEDDINGS:", Config.OLLAMA_EMBEDDINGS)
print("MONITORED_CHANNELS:", Config.MONITORED_CHANNELS)

# ---------------- 2) Проверка AstraDB + эмбеддинги ----------------
print("\n=== 2) Проверка AstraDB + эмбеддинги ===")
use_faiss = os.getenv("USE_FAISS_FALLBACK", "false").lower() == "true"

if use_faiss:
    print("FAISS fallback включён → пропускаю проверку AstraDB (это нормально).")
else:
    # Проверяем Ollama + AstraDB через наш векторстор
    try:
        from database import AstraDBVectorStore
        store = AstraDBVectorStore()
        print("Коллекция OK.")
        dim = len(store.embeddings.embed_query("probe"))
        print("Размерность эмбеддинга:", dim)
        ok = store.add_texts(
            ["тестовая запись для проверки индекса"],
            [{"category": "test", "channel": "probe"}],
        )
        print("Insert:", ok)
        res = store.similarity_search("проверка", k=1)
        print("Search sample:", res[:1])
    except Exception as e:
        print("Ошибка AstraDB:", e)

# ---------------- 3) Проверка Telethon доступа к каналам ----------------
print("\n=== 3) Проверка Telethon доступа к каналам ===")
try:
    import asyncio
    from telethon import TelegramClient

    async def check_channels():
        channels = Config.MONITORED_CHANNELS or {}
        if not channels:
            print("MONITORED_CHANNELS пуст — пропускаю проверку каналов.")
            return
        # создаём отдельную сессию 'parser' (файл parser.session появится в папке)
        async with TelegramClient('parser', Config.TELEGRAM_API_ID, Config.TELEGRAM_API_HASH) as client:
            # Telethon сам спросит phone/bot token в консоли (как раньше)
            for chat_id, cat in channels.items():
                try:
                    # chat_id в .env хранится как строка; приводим к int, если это id канала
                    ent = await client.get_entity(int(chat_id))
                    print(f"OK read: {abs(ent.id)} cat: {cat}")
                except ValueError:
                    # если это username/ссылка — пробуем как есть
                    try:
                        ent = await client.get_entity(chat_id)
                        print(f"OK read: {abs(ent.id)} cat: {cat}")
                    except Exception as ee:
                        print(f"FAIL read: {chat_id} → {ee}")
                except Exception as e:
                    print(f"FAIL read: {chat_id} → {e}")

    asyncio.run(check_channels())
except Exception as e:
    print("Ошибка Telethon-проверки:", e)

# ---------------- 4) Проверка часового пояса ----------------
print("\n=== 4) Проверка часового пояса ===")
try:
    # локальная зона задаётся в bot.py как LOCAL_TZ, но для проверки выведем Europe/Warsaw
    tz = ZoneInfo("Europe/Moscow")
    now = datetime.now(tz)
    print("Now local:", now)
except Exception as e:
    print("Ошибка при выводе времени:", e)
