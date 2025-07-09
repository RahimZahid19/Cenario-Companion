import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Validate environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME environment variable is not set.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Vector ID and namespace
vector_id = "85368f09-0769-484e-a8f9-1630e00cc646_chunk_0"
namespace = "85368f09-0769-484e-a8f9-1630e00cc646"

def get_vector_by_id(vector_id: str, namespace: str):
    try:
        response = index.fetch(ids=[vector_id], namespace=namespace)
        vector_data = response.vectors.get(vector_id)
        if vector_data:
            return vector_data.values
        else:
            print(f"⚠️ Vector ID '{vector_id}' not found in namespace '{namespace}'.")
            return None
    except Exception as e:
        print(f"❌ Error fetching vector: {e}")
        return None

def search_by_vector(vector_values, namespace: str, top_k=5):
    try:
        response = index.query(
            namespace=namespace,
            vector=vector_values,
            top_k=top_k,
            include_metadata=True
        )
        if hasattr(response, 'matches') and response.matches:
            print(f"\n🔍 Found {len(response.matches)} similar texts:")
            for i, match in enumerate(response.matches, 1):
                print(f"\n--- Match {i} ---")
                metadata = getattr(match, 'metadata', {})
                if isinstance(metadata, dict):
                    print(f"📝 Text: {metadata.get('text', '[No text found]')}")
                    print(f"🏷️ Meeting Title: {metadata.get('meeting_title', '[No meeting title]')}")
                    print(f"👥 Client: {metadata.get('client_name', '[No client name]')}")
                    print(f"📊 Score: {getattr(match, 'score', 'N/A')}")
                else:
                    print(f"📝 Text: {metadata}")
        else:
            print("⚠️ No matches found.")
    except Exception as e:
        print(f"❌ Error during vector search: {e}")

def display_all_texts_in_namespace(namespace: str, limit=10):
    """Display all texts from a specific namespace"""
    try:
        print(f"\n📚 Displaying all texts from namespace: {namespace}")
        print("=" * 60)
        
        # Query with dummy vector to get all documents in namespace
        response = index.query(
            namespace=namespace,
            vector=[0.0] * 1024,  # Dummy vector
            top_k=limit,
            include_metadata=True
        )
        
        if hasattr(response, 'matches') and response.matches:
            for i, match in enumerate(response.matches, 1):
                print(f"\n--- Document {i} ---")
                metadata = getattr(match, 'metadata', {})
                if isinstance(metadata, dict):
                    print(f"📝 Text: {metadata.get('text', '[No text found]')[:200]}...")
                    print(f"🏷️ Meeting Title: {metadata.get('meeting_title', '[No meeting title]')}")
                    print(f"👥 Client: {metadata.get('client_name', '[No client name]')}")
                    print(f"📅 Timestamp: {metadata.get('timestamp', '[No timestamp]')}")
                else:
                    print(f"📝 Text: {metadata}")
        else:
            print("⚠️ No documents found in this namespace.")
            
    except Exception as e:
        print(f"❌ Error fetching documents: {e}")

if __name__ == "__main__":
    print("🚀 Pinecone Text Display Tool")
    print("=" * 40)
    
    # Option 1: Search by vector similarity
    print("\n1️⃣ Searching by vector similarity...")
    vector_values = get_vector_by_id(vector_id, namespace)
    if vector_values:
        search_by_vector(vector_values, namespace)
    
    # Option 2: Display all texts in namespace
    print("\n2️⃣ Displaying all texts in namespace...")
    display_all_texts_in_namespace(namespace)
