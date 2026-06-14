# check_token_raw.py
import os, requests
from config import Config

endpoint = Config.ASTRA_DB_ENDPOINT.rstrip("/")
token = Config.ASTRA_DB_TOKEN

print("Endpoint:", endpoint)
print("Token starts with 'AstraCS:' ->", token.startswith("AstraCS:"))
print("Token length:", len(token))

headers = {
    "X-Cassandra-Token": token,
    "Content-Type": "application/json",
}
# просто запрос списка коллекций в default_keyspace (должен дать 200, даже если пусто)
resp = requests.post(f"{endpoint}/api/json/v1/default_keyspace",
                     headers=headers, json={"findCollections": {}}, timeout=20)

print("Status:", resp.status_code)
print("Body head:", resp.text[:300])
