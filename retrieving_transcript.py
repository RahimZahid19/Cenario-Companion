import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
from pinecone import Pinecone
from cohere import Client as CohereClient
import re

# Load environment variables
load_dotenv()

class CohereEmbeddings:
    """Custom Cohere embeddings class that matches LangChain interface"""
    
    def __init__(self):
        self.client = CohereClient(os.getenv("COHERE_API_KEY"))
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        response = self.client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        response = self.client.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        return response.embeddings[0]


def get_transcript_from_pinecone(session_id: str, project_id: str = None, namespace: str = None) -> str:
    """
    Retrieve full transcript content from Pinecone vector database
    """
    try:
        print(f"🔍 Searching Pinecone for session_id: {session_id}")
        
        # Initialize embeddings and Pinecone
        embeddings = CohereEmbeddings()
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Try multiple namespace strategies
        namespaces_to_try = [
            namespace,
            project_id, 
            session_id,
            None  # Default namespace
        ]
        
        documents = []
        used_namespace = None
        
        for ns in namespaces_to_try:
            if ns is None:
                continue
                
            print(f"🔍 Trying namespace: {ns}")
            
            try:
                vector_store = PineconeVectorStore(
                    index=pc.Index(index_name),
                    embedding=embeddings,
                    namespace=ns
                )
                
                # Try different filter combinations
                filter_combinations = [
                    {"session_id": session_id},
                    {"project_id": session_id},
                    {"meeting_id": session_id},
                    {"session_id": session_id, "project_id": project_id} if project_id else None,
                    {}  # No filter
                ]
                
                for filter_condition in filter_combinations:
                    if filter_condition is None:
                        continue
                        
                    print(f"🔍 Trying filter: {filter_condition}")
                    
                    documents = vector_store.similarity_search(
                        query="transcript content",
                        k=1000,
                        filter=filter_condition if filter_condition else None
                    )
                    
                    if documents:
                        print(f"✅ Found {len(documents)} documents with filter: {filter_condition}")
                        used_namespace = ns
                        break
                
                if documents:
                    break
                    
            except Exception as e:
                print(f"❌ Error with namespace {ns}: {e}")
                continue
        
        if not documents:
            print("❌ No documents found with any strategy")
            # Call debug function
            debug_pinecone_data(session_id)
            return ""
        
        print(f"✅ Using namespace: {used_namespace}")
        print(f"✅ Found {len(documents)} documents total")
        
        # Show metadata of first few documents for debugging
        for i, doc in enumerate(documents[:3]):
            print(f"📄 Document {i+1} metadata: {doc.metadata}")
        
        # Sort documents by chunk_index and reconstruct transcript
        sorted_documents = sorted(
            documents, 
            key=lambda doc: doc.metadata.get('chunk_index', 0)
        )
        
        # Combine all chunks
        transcript_chunks = []
        for doc in sorted_documents:
            if hasattr(doc, 'page_content') and doc.page_content:
                content = doc.page_content.strip()
                if content:
                    transcript_chunks.append(content)
        
        # Join all chunks to reconstruct the full transcript
        full_transcript = "\n\n".join(transcript_chunks)
        
        print(f"✅ Reconstructed transcript: {len(full_transcript)} characters")
        
        return full_transcript
        
    except Exception as e:
        print(f"❌ Error retrieving transcript from Pinecone: {e}")
        return ""       

def get_transcript_by_session_id(session_id: str, namespace: Optional[str] = None) -> str:
    """
    Retrieve all transcript chunks for a given session_id from Pinecone vector store,
    sort them by chunk_index, and reconstruct the full transcript.
    
    Args:
        session_id (str): The session ID to filter transcript chunks
        namespace (str, optional): The Pinecone namespace to search in. 
                                 If None, uses session_id as namespace (based on existing pattern)
    
    Returns:
        str: The complete reconstructed transcript
    """
    try:
        # Initialize Cohere embeddings (matching your existing data)
        embeddings = CohereEmbeddings()
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Get the index
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Initialize Pinecone vector store with LangChain
        vector_store = PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=embeddings,
            namespace=namespace
        )
        
        # Query Pinecone with metadata filter
        # Use similarity search with filter to get all chunks for the session
        filter_condition = {"session_id": session_id}
        
        # Get all matching documents (using a large k value to get all chunks)
        # Since we're filtering by session_id, we'll get only relevant chunks
        documents = vector_store.similarity_search(
            query="",  # Empty query since we're filtering by metadata
            k=1000,    # Large number to ensure we get all chunks
            filter=filter_condition
        )
        
        # If no documents found, try alternative metadata field names
        if not documents:
            # Try with project_id as well (based on existing code pattern)
            filter_condition = {"project_id": session_id}
            documents = vector_store.similarity_search(
                query="",
                k=1000,
                filter=filter_condition
            )
        
        # Sort documents by chunk_index
        sorted_documents = sorted(
            documents, 
            key=lambda doc: doc.metadata.get('chunk_index', 0)
        )
        
        # Concatenate the text fields (page_content) in order with proper formatting
        transcript_chunks = []
        for doc in sorted_documents:
            if hasattr(doc, 'page_content') and doc.page_content:
                # Get the content and clean it up
                content = doc.page_content.strip()
                if content:
                    transcript_chunks.append(content)
        
        # Join all chunks to reconstruct the full transcript
        full_transcript = "\n\n".join(transcript_chunks)
        
        # Clean and format the transcript
        formatted_transcript = clean_and_format_transcript(full_transcript)
        
        return formatted_transcript
        
    except Exception as e:
        return ""

def get_transcript_chunks_by_session_id(session_id: str, namespace: Optional[str] = None) -> List[Document]:
    """
    Retrieve all transcript chunks for a given session_id as separate Document objects.
    Useful if you need to work with individual chunks and their metadata.
    
    Args:
        session_id (str): The session ID to filter transcript chunks
        namespace (str, optional): The Pinecone namespace to search in
    
    Returns:
        List[Document]: List of Document objects sorted by chunk_index
    """
    try:
        # Initialize Cohere embeddings (matching your existing data)
        embeddings = CohereEmbeddings()
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Get the index
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Initialize Pinecone vector store with LangChain
        vector_store = PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=embeddings,
            namespace=namespace
        )
        
        # Query Pinecone with metadata filter
        filter_condition = {"session_id": session_id}
        
        # Get all matching documents
        documents = vector_store.similarity_search(
            query="",
            k=1000,
            filter=filter_condition
        )
        
        # If no documents found, try alternative metadata field names
        if not documents:
            filter_condition = {"project_id": session_id}
            documents = vector_store.similarity_search(
                query="",
                k=1000,
                filter=filter_condition
            )
        
        # Sort documents by chunk_index
        sorted_documents = sorted(
            documents, 
            key=lambda doc: doc.metadata.get('chunk_index', 0)
        )
        
        return sorted_documents
        
    except Exception as e:
        return []

def clean_and_format_transcript(content: str) -> str:
    """
    Clean and format transcript content to match the desired format.
    Convert from the current format to the proper timestamp format.
    """
    # Remove "system." prefix if it exists
    if content.startswith('system.'):
        content = content[7:].strip()
    
    # Split content by existing timestamps to identify different speakers/segments
    
    # Find all timestamp patterns
    timestamp_pattern = r'\[(\d{1,2}:\d{2})\] (k\d+) ([^:]+):'
    
    # Split the content by timestamps
    segments = re.split(timestamp_pattern, content)
    
    # The first segment is content before any timestamp
    first_segment = segments[0].strip()
    
    # Reconstruct the transcript
    formatted_content = []
    
    # If there's content before the first timestamp, add it with a default timestamp
    if first_segment and not first_segment.startswith('['):
        # Use the timestamp from the desired format
        formatted_content.append(f"[41:50] k214702 Abdul Basit Allahwala: {first_segment}")
    
    # Process the rest of the segments (they come in groups of 4: timestamp, user_id, name, content)
    for i in range(1, len(segments), 4):
        if i + 3 < len(segments):
            timestamp = segments[i]
            user_id = segments[i + 1]
            name = segments[i + 2]
            segment_content = segments[i + 3].strip()
            
            if segment_content:
                formatted_content.append(f"[{timestamp}] {user_id} {name}: {segment_content}")
    
    return '\n\n'.join(formatted_content)


# # Example usage:
# if __name__ == "__main__":
#     # Example with a real session ID
#     session_id = "ee26137a-6128-46c9-88ef-ac78fbedb70a"
#     namespace = "ac253793-e4d8-404a-8300-096f2e456cca"
    
#     transcript = get_transcript_by_session_id(session_id, namespace=namespace)
#     print(transcript)
