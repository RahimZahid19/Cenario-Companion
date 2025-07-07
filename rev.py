import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
import csv
import io

# Load API keys and settings from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Use Cohere embedding model (1024-d)
embedding_model = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-english-v3.0"
)

# Initialize LLM (Groq + LLaMA3)
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# Build LangChain retrieval pipeline
def build_retrieval_chain(namespace: str):
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key="text",
        namespace=namespace
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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

# Parse CSV to JSON
def parse_csv_to_json(csv_text: str, namespace=None):
    reader = csv.DictReader(io.StringIO(csv_text))
    data = []
    for row in reader:
        if "Session" in row and namespace:
            row["Session"] = namespace
        data.append(row)
    return data

# Parse user stories from raw output
def parse_user_stories(text: str):
    return [
        {"user_story": line.strip()}
        for line in text.splitlines()
        if line.strip().startswith("•")
    ]
