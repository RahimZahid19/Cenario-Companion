import os
from dotenv import load_dotenv
from cohere import Client as CohereClient
from pinecone import Pinecone
from typing import List
import textwrap




# Load .env
load_dotenv()


CHUNK_SIZE = 1000

# Initialize Cohere
co = CohereClient(os.getenv("COHERE_API_KEY"))

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Access the index
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Read transcript from file
def read_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# Get embeddings from Cohere
def get_embedding(text: str) -> list:
    return co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_document"  # This is REQUIRED for this model
    ).embeddings[0]

# Upsert to Pinecone
def upsert_to_pinecone(embedding: list, metadata: dict):
    vector_id = f"{metadata['project_id']}_chunk_{metadata['chunk_index']}"
    pinecone_index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }
        ],
        namespace=metadata["project_id"]
    )

def upsert_question_answer(embedding: list, metadata: dict, vector_id: str, namespace: str):
    pinecone_index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }
        ],
        namespace=namespace
    )

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    return textwrap.wrap(text, width=chunk_size, break_long_words=False, replace_whitespace=False)