import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
from pinecone import Pinecone

# Load environment variables
load_dotenv()

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
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Get the index
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Initialize Pinecone vector store with LangChain
        vector_store = PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=embeddings,
            namespace=namespace or session_id
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
        
        # Concatenate the text fields (page_content) in order
        transcript_chunks = []
        for doc in sorted_documents:
            if hasattr(doc, 'page_content') and doc.page_content:
                transcript_chunks.append(doc.page_content)
        
        # Join all chunks to reconstruct the full transcript
        full_transcript = " ".join(transcript_chunks)
        
        return full_transcript
        
    except Exception as e:
        print(f"Error retrieving transcript for session_id {session_id}: {str(e)}")
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
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Get the index
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Initialize Pinecone vector store with LangChain
        vector_store = PineconeVectorStore(
            index=pc.Index(index_name),
            embedding=embeddings,
            namespace=namespace or session_id
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
        print(f"Error retrieving transcript chunks for session_id {session_id}: {str(e)}")
        return []

# Example usage (commented out):
# if __name__ == "__main__":
#     # Example usage
#     session_id = "your_session_id"
#     transcript = get_transcript_by_session_id(session_id)
#     print(f"Retrieved transcript length: {len(transcript)} characters")
#     
#     # Or get individual chunks
#     chunks = get_transcript_chunks_by_session_id(session_id)
#     print(f"Found {len(chunks)} chunks")
#     for i, chunk in enumerate(chunks):
#         print(f"Chunk {i}: {chunk.metadata.get('chunk_index', 'unknown')}")
