from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

try:
    pc.delete_index("portfolio")
    print("✅ Deleted old index 'portfolio'")
except Exception as e:
    print(f"⚠️ No existing index or delete failed: {e}")

print("Remaining indexes:", [index.name for index in pc.list_indexes()])

