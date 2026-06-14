from telethon import TelegramClient

API_ID = 27553059   # твой api_id из .env
API_HASH = "8b0e8cb44be370a2d0b485b2e78ab693"  # из .env

# имя должно совпадать с parser.session, чтобы Telethon его нашёл
client = TelegramClient("parser", API_ID, API_HASH)

async def main():
    me = await client.get_me()
    print("=== Данные parser.session ===")
    print("ID:", me.id)
    print("Username:", me.username)
    print("Имя:", me.first_name, me.last_name)
    print("Телефон:", me.phone)

with client:
    client.loop.run_until_complete(main())
