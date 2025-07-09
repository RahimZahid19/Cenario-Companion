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

def fetch_requirement_descriptions(project_id, session_id):
    namespace = project_id
    filter_query = {"type": {"$eq": "requirement"}}
    if session_id:
        filter_query["session_id"] = {"$eq": session_id}

    response = index.query(
        namespace=namespace,
        top_k=50,
        vector=[0.0] * 1024,  # Dummy vector to enable filtering
        filter=filter_query,
        include_metadata=True
    )

    return [match['metadata']['description'] for match in response['matches'] if 'description' in match['metadata']]


def generate_and_group_epics(descriptions, project_id, session_id):
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI product analyst. Based on the following requirements:

        {requirements}

        Group similar requirements under epics. Each epic must include:
        - Epic_Id (e.g. 1, 1.1, 2)
        - Epic_Title

        Respond ONLY with a CSV formatted as follows (exact headers required, no extra spaces or commentary):
        Epic_Id,Epic_Title
        1,Your First Epic Title
        2,Your Second Epic Title
        """
    )

    chain = prompt | llm | StrOutputParser()
    combined_reqs = "\n".join(f"- {r}" for r in descriptions)
    csv_output = chain.invoke({"requirements": combined_reqs})

    csv_lines = csv_output.strip().splitlines()
    clean_lines = [line for line in csv_lines if "Epic_Id" in line or "," in line]
    cleaned_csv = "\n".join(clean_lines)

    return parse_csv_to_grouped_json(cleaned_csv, descriptions, project_id, session_id)


def parse_csv_to_grouped_json(csv_str, descriptions, project_id, session_id):
    csv_str = csv_str.strip()
    reader = csv.DictReader(io.StringIO(csv_str))
    
    expected_headers = {"Epic_Id", "Epic_Title"}
    if not expected_headers.issubset(set(reader.fieldnames or [])):
        raise ValueError(f"Invalid CSV headers. Expected: {expected_headers}, Got: {reader.fieldnames}")
    
    grouped = {}
    id_counter = 1
    desc_index = 0

    for row in reader:
        title = row["Epic_Title"].strip()

        if title not in grouped:
            grouped[title] = {
                "id": str(id_counter),
                "title": title,
                "project_id": project_id,
                "session_id": session_id,
                "user_stories": []
            }
            id_counter += 1

        if desc_index >= len(descriptions):
            break

        desc = descriptions[desc_index].strip()
        story_id = f"{grouped[title]['id']}.{len(grouped[title]['user_stories']) + 1}"

        grouped[title]["user_stories"].append({
    "id": story_id,
    "question": desc,
    "answers": []  # Empty list instead of placeholder
})

        desc_index += 1

    return [epic for epic in grouped.values() if epic["user_stories"]]
