"""Quick connectivity check for Astra DB. Requires .env with ASTRA_DB_TOKEN and ASTRA_DB_ENDPOINT."""

from astrapy import DataAPIClient

from config import Config

if not Config.ASTRA_DB_TOKEN or not Config.ASTRA_DB_ENDPOINT:
    raise SystemExit("Set ASTRA_DB_TOKEN and ASTRA_DB_ENDPOINT in .env")

client = DataAPIClient(Config.ASTRA_DB_TOKEN)
db = client.get_database_by_api_endpoint(Config.ASTRA_DB_ENDPOINT)

print(f"Connected to Astra DB: {db.list_collection_names()}")
