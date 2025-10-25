from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Environment variables
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

print(f"[OK] Connected to Pinecone")
print(f"[INFO] Existing indexes: {[index.name for index in pc.list_indexes()]}")

# STEP 1 — Delete old index if it exists
try:
    pc.delete_index("portfolio")
    print("[DELETE] Deleted old index 'portfolio'")
    time.sleep(5)  # wait a bit for propagation
except Exception as e:
    print(f"[INFO] No existing index to delete: {e}")

# STEP 2 — Create vector-enabled index
print("[CREATE] Creating vector-enabled index...")
pc.create_index(
    name="portfolio",
    dimension=768,
    metric="cosine",
    spec={
        "serverless": {
            "cloud": "aws",
            "region": pinecone_environment
        }
    }
)
print(f"[OK] Created vector-enabled index: portfolio")

# STEP 3 — Verify
print("[INFO] Indexes now:", [index.name for index in pc.list_indexes()])

# Get the index
index = pc.Index("portfolio")

# STEP 4 — Load JSON data
with open("./utils/data.json", "r", encoding="utf-8") as file:
    text_list = json.load(file)

# Text splitter
r_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],
    chunk_size=200,
    chunk_overlap=0,
    length_function=len
)

# Encoder
encoder = SentenceTransformer("all-mpnet-base-v2")

# STEP 5 — Insert vectorized chunks
vectors_to_upsert = []

for item in text_list:
    id = item["id"]
    title = item["title"]
    description = item["description"]

    chunks = r_splitter.split_text(description)

    for idx, chunk in enumerate(chunks):
        vector = encoder.encode(chunk).tolist()
        
        vectors_to_upsert.append({
            "id": f"{id}-{idx}",
            "values": vector,
            "metadata": {
                "title": title,
                "description": chunk
            }
        })

# Batch upsert to Pinecone
if vectors_to_upsert:
    try:
        index.upsert(vectors=vectors_to_upsert)
        print(f"[OK] Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone")
    except Exception as e:
        print(f"[ERROR] Error upserting vectors: {e}")
else:
    print("[INFO] No vectors to upsert")
