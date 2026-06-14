import asyncio
from telethon import TelegramClient
from config import Config


async def main():
    async with TelegramClient('id_checker', Config.TELEGRAM_API_ID, Config.TELEGRAM_API_HASH) as client:
        channels = Config.MONITORED_CHANNELS or {}
        if not channels:
            print("❌ MONITORED_CHANNELS пуст. Проверь .env")
            return

        print("🔎 Проверяем каналы из MONITORED_CHANNELS:")
        for url, category in channels.items():
            try:
                entity = await client.get_entity(url)
                print(f"✅ {url} ({category}) → ID: {entity.id}")
            except Exception as e:
                print(f"⚠️ Ошибка при получении ID {url}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
