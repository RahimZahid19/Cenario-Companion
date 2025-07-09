import os
import csv
import io
from typing import Optional
from dotenv import load_dotenv
from pinecone import Pinecone, Vector
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize Cohere embedding model
embedding_model = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-english-v3.0"
)

# Initialize LLM (Groq + LLaMA3)
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# Build retrieval chain
def build_retrieval_chain(project_id: str, session_id: Optional[str] = None):
    filter_dict = {"project_id": project_id}
    if session_id:
        filter_dict["session_id"] = session_id

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key="text",
        namespace=project_id
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "filter": filter_dict
        }
    )

    prompt_template = ChatPromptTemplate.from_template("""
Use the context below to answer the question.

<context>
{context}
</context>

Question: {question}

Only output the CSV table. Do not include any additional explanation or text. If information is missing, use 'N/A'. Ensure the information is accurate and matches the context.
""")

    chain = (
        RunnableParallel(
            context=lambda x: retriever.invoke(x["question"]),
            question=lambda x: x["question"]
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain

# Parse LLM CSV output and upsert to Pinecone
def parse_csv_to_json(csv_text: str, project_id: str, session_id: Optional[str] = None):
    meeting_title = project_id

    try:
        chunk_id = f"{project_id}_chunk_0"
        fetch = index.fetch(ids=[chunk_id], namespace=project_id)

        if fetch and hasattr(fetch, "vectors") and chunk_id in fetch.vectors:
            vector_obj = fetch.vectors[chunk_id]
            metadata = getattr(vector_obj, 'metadata', {})
            meeting_title = metadata.get('meeting_title', project_id) if isinstance(metadata, dict) else project_id
    except Exception as e:
        print(f"Warning: Could not fetch meeting_title from Pinecone: {e}")

    reader = csv.DictReader(io.StringIO(csv_text))
    data = []

    for idx, row in enumerate(reader, start=1):
        new_row = {
            "id": f"REQ-{idx:03d}",
            "description": row.get("Description", ""),
            "category": row.get("Category", ""),
            "priority": row.get("Priority", ""),
            "session": session_id or meeting_title,
            "sources": row.get("Sources", ""),
            "project_id": project_id
        }

        try:
            embedding = embedding_model.embed_query(new_row["description"])
            vector = Vector(
                id=new_row["id"],
                values=embedding,
                metadata=new_row
            )
            index.upsert(vectors=[vector], namespace=project_id)
        except Exception as e:
            print(f"Error upserting {new_row['id']} to Pinecone: {e}")

        data.append(new_row)

    return data

# Parse user stories (optional)
def parse_user_stories(text: str):
    return [
        {"userStory": line.strip()}
        for line in text.splitlines()
        if line.strip().startswith("•")
    ]
