# quick_fetch.py
import asyncio
from datetime import datetime, timedelta
from telethon import TelegramClient
from config import Config

async def main():
    channels = Config.MONITORED_CHANNELS
    async with TelegramClient('parser', Config.TELEGRAM_API_ID, Config.TELEGRAM_API_HASH) as client:
        for chat_id, cat in channels.items():
            print(f"\n=== Канал {chat_id} [{cat}] ===")
            try:
                entity = await client.get_entity(int(chat_id))
            except Exception:
                entity = await client.get_entity(chat_id)  # username/ссылка

            cnt = 0
            async for m in client.iter_messages(entity, limit=10):
                # Телеграм даёт дату в UTC; печатаем кратко
                text = (m.message or "").strip().replace("\n", " ")
                print(f"{m.id} | {m.date} | {text[:80]}")
                cnt += 1
            print(f"Всего прочитано: {cnt}")

asyncio.run(main())
