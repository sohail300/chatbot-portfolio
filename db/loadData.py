from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from astrapy import DataAPIClient
import os
from dotenv import load_dotenv

load_dotenv()

# Environment variables
db_token = os.getenv('DB_TOKEN')
db_endpoint = os.getenv('DB_ENDPOINT')

# Initialize the DB client
client = DataAPIClient(db_token)
db = client.get_database_by_api_endpoint(
  db_endpoint
)
print(f"Connected to Astra DB: {db.info().name}")
print(f"Connected to Astra DB: {db.list_collection_names()}")

# Create a collection.
collection = db.create_collection(
    "portfolio",
    dimension=768,
    metric= "cosine",  # or simply "cosine"
    check_exists=False,
)
print(f"* Collection: {collection.full_name}\n")

# Read and load the JSON data from the file
with open('./utils/data.json', 'r') as file:
    text_list = json.load(file)

# Spiltter
r_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " "],
    chunk_size = 200,
    chunk_overlap  = 0,
    length_function = len
)

# Encoder
encoder = SentenceTransformer("all-mpnet-base-v2")

for item in text_list:
        id = item['id']
        title = item['title']
        description = item['description']
        
        chunks = r_splitter.split_text(description)
        
        for chunk in chunks:
            vector = encoder.encode(chunk)
            dim=vector.shape
            print(dim)

            try:
                insertion_result = collection.insert_one({
                    "_id": id,
                    "title": title,
                    "$vector": vector,
                    "description": chunk
                })
                print(f"* Inserted {len(insertion_result.inserted_ids)} items.\n")
            except:
                print("* Documents found on DB already. Let's move on.\n")
