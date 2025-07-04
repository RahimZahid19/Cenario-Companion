import os
from dotenv import load_dotenv

def setup_env():
    load_dotenv()

    # === GROQ Key Check ===
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set. Make sure your .env file includes it.")

    # === Pinecone API Key Check ===
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY is not set. Add it to your .env file.")

    # === Pinecone Environment Check ===
    if not os.getenv("PINECONE_ENVIRONMENT"):
        raise ValueError("PINECONE_ENVIRONMENT is not set. Add it to your .env file.")
