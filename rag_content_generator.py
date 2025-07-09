import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from typing import List, Dict, Optional
from pinecone import Pinecone
from cohere import Client as CohereClient
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
import json
from datetime import datetime

# Load environment variables
from load_env import setup_env
setup_env()

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

class RAGContentGenerator:
    """
    RAG-based content generator that retrieves relevant vectors and generates 
    key insights, action items, and meeting summaries
    """
    
    def __init__(self):
        # Initialize embeddings
        self.embeddings = CohereEmbeddings()
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Initialize LLM (using the same setup as the project)
        self.llm = init_chat_model("llama3-8b-8192", model_provider="groq")
        
    def retrieve_relevant_documents(self, query: str, session_id: str, project_id: str = None, 
                                  namespace: str = None, top_k: int = 10) -> List[Document]:
        """
        Retrieve relevant documents from Pinecone based on query and metadata filters
        """
        try:
            # Initialize vector store
            vector_store = PineconeVectorStore(
                index=self.pc.Index(self.index_name),
                embedding=self.embeddings,
                namespace=namespace or project_id
            )
            
            # Create metadata filter
            filter_condition = {"session_id": session_id}
            if project_id:
                filter_condition["project_id"] = project_id
            
            # Retrieve relevant documents
            documents = vector_store.similarity_search(
                query=query,
                k=top_k,
                filter=filter_condition
            )
            
            # If no documents found, try without project_id filter
            if not documents:
                filter_condition = {"session_id": session_id}
                documents = vector_store.similarity_search(
                    query=query,
                    k=top_k,
                    filter=filter_condition
                )
            
            return documents
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def generate_key_insights(self, session_id: str, project_id: str = None, 
                            namespace: str = None) -> str:
        """
        Generate key insights using RAG approach
        """
        # Retrieve relevant documents
        insights_query = "key insights important points decisions outcomes conclusions findings"
        documents = self.retrieve_relevant_documents(
            query=insights_query,
            session_id=session_id,
            project_id=project_id,
            namespace=namespace,
            top_k=15
        )
        
        if not documents:
            return "No relevant documents found for generating insights."
        
        # Combine retrieved content
        retrieved_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Create prompt template
        insights_prompt = PromptTemplate.from_template("""
        You are a senior business analyst tasked with extracting key insights from meeting content.
        
        Based on the following retrieved meeting content, generate 5-7 key insights that capture the most important points, decisions, and outcomes discussed.
        
        Retrieved Meeting Content:
        {retrieved_content}
        
        Instructions:
        - Focus on actionable insights and important decisions
        - Highlight technical requirements, business needs, and constraints
        - Include any risks, opportunities, or concerns mentioned
        - Keep each insight concise but comprehensive
        - Prioritize insights that impact project success
        
        Format your response as a clear, structured list of key insights:
        """)
        
        # Generate insights
        try:
            prompt = insights_prompt.format(retrieved_content=retrieved_content)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error generating insights: {e}"
    
    def generate_action_items(self, session_id: str, project_id: str = None, 
                            namespace: str = None) -> str:
        """
        Generate action items using RAG approach
        """
        # Retrieve relevant documents
        action_query = "action items tasks responsibilities assignments deliverables deadlines next steps"
        documents = self.retrieve_relevant_documents(
            query=action_query,
            session_id=session_id,
            project_id=project_id,
            namespace=namespace,
            top_k=15
        )
        
        if not documents:
            return "No relevant documents found for generating action items."
        
        # Combine retrieved content
        retrieved_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Create prompt template
        action_prompt = PromptTemplate.from_template("""
        You are a project manager tasked with extracting action items from meeting content.
        
        Based on the following retrieved meeting content, generate a comprehensive list of action items that need to be completed.
        
        Retrieved Meeting Content:
        {retrieved_content}
        
        Instructions:
        - Extract specific tasks and responsibilities mentioned
        - Include any deadlines or timelines discussed
        - Identify who is responsible for each action (if mentioned)
        - Include follow-up items and next steps
        - Prioritize actions that are time-sensitive or critical
        
        Format your response as a structured list of action items with clear ownership and timelines where available:
        """)
        
        # Generate action items
        try:
            prompt = action_prompt.format(retrieved_content=retrieved_content)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error generating action items: {e}"
    
    def generate_meeting_summary(self, session_id: str, project_id: str = None, 
                               namespace: str = None) -> str:
        """
        Generate meeting summary using RAG approach
        """
        # Retrieve relevant documents
        summary_query = "meeting summary discussion topics agenda objectives outcomes"
        documents = self.retrieve_relevant_documents(
            query=summary_query,
            session_id=session_id,
            project_id=project_id,
            namespace=namespace,
            top_k=20
        )
        
        if not documents:
            return "No relevant documents found for generating meeting summary."
        
        # Combine retrieved content
        retrieved_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Create prompt template
        summary_prompt = PromptTemplate.from_template("""
        You are a professional meeting facilitator tasked with creating a comprehensive meeting summary.
        
        Based on the following retrieved meeting content, generate a clear and structured meeting summary.
        
        Retrieved Meeting Content:
        {retrieved_content}
        
        Instructions:
        - Provide a brief overview of the meeting purpose and participants
        - Summarize key discussion points and topics covered
        - Highlight important decisions made
        - Include any agreements reached or consensus achieved
        - Mention any concerns or challenges discussed
        - Keep the summary professional and well-organized
        
        Format your response as a professional meeting summary with appropriate sections:
        """)
        
        # Generate summary
        try:
            prompt = summary_prompt.format(retrieved_content=retrieved_content)
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error generating meeting summary: {e}"
    
    def generate_all_content(self, session_id: str, project_id: str = None, 
                           namespace: str = None) -> Dict[str, str]:
        """
        Generate all content types (insights, action items, summary) in one call
        """
        print(f"Generating content for session: {session_id}")
        if project_id:
            print(f"Project: {project_id}")
        if namespace:
            print(f"Namespace: {namespace}")
        
        results = {}
        
        # Generate key insights
        print("\n🔍 Generating key insights...")
        results['key_insights'] = self.generate_key_insights(session_id, project_id, namespace)
        
        # Generate action items
        print("\n📋 Generating action items...")
        results['action_items'] = self.generate_action_items(session_id, project_id, namespace)
        
        # Generate meeting summary
        print("\n📝 Generating meeting summary...")
        results['meeting_summary'] = self.generate_meeting_summary(session_id, project_id, namespace)
        
        return results
    
    def save_content_to_files(self, content: Dict[str, str], session_id: str, 
                            timestamp: str = None) -> Dict[str, str]:
        """
        Save generated content to files
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        for content_type, content_text in content.items():
            filename = f"{session_id}_{content_type}_{timestamp}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content_text)
                saved_files[content_type] = filename
                print(f"✅ Saved {content_type} to: {filename}")
            except Exception as e:
                print(f"❌ Error saving {content_type}: {e}")
        
        return saved_files

def main():
    """
    Main function to test the RAG content generator
    """
    print("🚀 RAG Content Generator Test")
    print("=" * 50)
    
    # Initialize generator
    generator = RAGContentGenerator()
    
    # Test parameters - replace with your actual session and project IDs
    session_id = "ad0b70aa-4fb7-4e74-a55a-5175a9b6fd24"
    project_id = "53a6dd9d-cae0-472e-af6e-a8ab11e39c26"  # Optional
    namespace = "53a6dd9d-cae0-472e-af6e-a8ab11e39c26"   # Optional
    
    # Generate all content
    try:
        content = generator.generate_all_content(
            session_id=session_id,
            project_id=project_id,
            namespace=namespace
        )
        
        print("\n" + "=" * 50)
        print("📊 GENERATED CONTENT")
        print("=" * 50)
        
        # Display results
        for content_type, content_text in content.items():
            print(f"\n🔸 {content_type.upper().replace('_', ' ')}")
            print("-" * 40)
            print(content_text)
            print("\n" + "=" * 50)
        
        # Save to files
        print("\n💾 Saving content to files...")
        saved_files = generator.save_content_to_files(content, session_id)
        
        print(f"\n✅ Content generation completed!")
        print(f"📁 Files saved: {list(saved_files.values())}")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")

if __name__ == "__main__":
    main() 