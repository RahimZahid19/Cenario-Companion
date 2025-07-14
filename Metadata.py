import re
from datetime import datetime
from typing import List, Dict, Optional
import hashlib
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uuid
from load_env import setup_env
setup_env()

from langchain.chat_models import init_chat_model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# Global variable to store the full metadata from API
full_metadata_from_api = None

def set_full_metadata(metadata_dict: dict):
    """
    Set the full metadata received from the API
    """
    global full_metadata_from_api
    full_metadata_from_api = metadata_dict
    print("=== FULL METADATA SET IN METADATA.PY ===")
    print(full_metadata_from_api)
    print("=== END OF FULL METADATA ===")

def get_full_metadata():
    """
    Get the full metadata that was set from the API
    """
    return full_metadata_from_api

class Metadata:
    def __init__(self, manual_fields: dict):
        """
        Initialize with manual fields collected before the meeting.
        manual_fields: dict containing all fields filled by the user (from frontend popup)
        """
        self.manual_fields = manual_fields.copy()
        # Parse stakeholders if needed
        if 'stakeholders' in self.manual_fields:
            stakeholders = self.manual_fields['stakeholders']
            if isinstance(stakeholders, str):
                self.manual_fields['stakeholders'] = self.parse_stakeholders(stakeholders)
        self.generated_fields = {}

    @staticmethod
    def parse_stakeholders(stakeholders_str: str) -> list:
        """
        Parses a string of stakeholders and roles into a list of 'Name (Role)' strings.
        Handles both 'Name (Role)' and 'Name is the Role' formats.
        """
        # Split by comma
        entries = [e.strip() for e in stakeholders_str.split(',') if e.strip()]
        result = []
        for entry in entries:
            # Try to match 'Name (Role)'
            match_paren = re.match(r"(.+?)\s*\((.+)\)", entry)
            if match_paren:
                name = match_paren.group(1).strip()
                role = match_paren.group(2).strip()
                result.append(f"{name} ({role})")
                continue
            # Try to match 'Name is the Role' or 'Name is Role'
            match_is = re.match(r"(.+?)\s+is(?:\s+the)?\s+(.+)", entry, re.IGNORECASE)
            if match_is:
                name = match_is.group(1).strip()
                role = match_is.group(2).strip()
                result.append(f"{name} ({role})")
                continue
            # Fallback: just use the entry as is
            result.append(entry)
        return result

    @staticmethod
    def load_transcript(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def generate_project_tags(transcript: str) -> list:
        prompt = (
            "Extract 3–7 relevant technical or project-related tags from the transcript below. "
            "Return only the tags, comma-separated, no introduction or numbering.\n\n"
            f"{transcript}"
        )
        response = llm.invoke(prompt).content.strip()
        # Split and clean tags
        tag_candidates = re.split(r",|\n", response)
        cleaned_tags = [tag.strip(" -•*0123456789.") for tag in tag_candidates if tag.strip()]
        # Remove duplicates
        return list(dict.fromkeys(cleaned_tags))

    def generate_after_meeting(self, transcript_path: str):
        """
        Call after the meeting ends. Generates project_tags and timestamp from transcript.
        """
        transcript = self.load_transcript(transcript_path)
        project_tags = self.generate_project_tags(transcript)
        timestamp = datetime.now().isoformat()
        self.generated_fields = {
            "project_tags": project_tags,
            "timestamp": timestamp
        }

    def get_metadata(self, chunk_index: int = 0) -> dict:
        """
        Returns the final flat metadata dict, merging manual and generated fields.
        chunk_index can be set if needed (default 0).
        """
        # Flatten project_tags to list or string as needed
        merged = {
            **self.manual_fields,
            **self.generated_fields,
            "chunk_index": chunk_index
        }
        return merged

def extract_project_fields_from_text(text: str) -> dict:
    # Limit text to first 4000 characters to avoid LLM token overflow
    limited_text = text[:4000]
    print("=== Extracted text for LLM ===")
    print(limited_text)
    print("=== End of extracted text ===")
    prompt = (
        "You are an expert assistant. Extract the following fields from the RFP text below and return them as a JSON object with these exact keys:\n"
        "- project_title\n- project_status\n- client_name\n- proposal_deadline\n"
        "- engagement_type\n- industry\n- software_type\n- client_introduction\n\n"
        "If a field is not found, return an empty string for that field.\n"
        "Respond ONLY with a valid JSON object, no explanation or extra text.\n"
        "Example output:\n"
        "{\n"
        "  \"project_title\": \"...\",\n"
        "  \"project_status\": \"...\",\n"
        "  \"client_name\": \"...\",\n"
        "  \"proposal_deadline\": \"...\",\n"
        "  \"engagement_type\": \"...\",\n"
        "  \"industry\": \"...\",\n"
        "  \"software_type\": \"...\",\n"
        "  \"client_introduction\": \"...\"\n"
        "}\n\n"
        "RFP Text:\n" + limited_text
    )
    response = llm.invoke(prompt).content.strip()
    print("=== LLM response ===")
    print(response)
    print("=== End of LLM response ===")
    import json
    try:
        fields = json.loads(response)
    except Exception:
        fields = {
            "project_title": "",
            "project_status": "",
            "client_name": "",
            "proposal_deadline": "",
            "engagement_type": "",
            "industry": "",
            "software_type": "",
            "client_introduction": ""
        }
    return fields

# app = FastAPI()

# # In-memory storage for demonstration
# projects_db = {}
# sessions_db = {}

class ProjectCreateRequest(BaseModel):
    project_title: str
    project_status: str
    client_name: str
    proposal_deadline: str
    engagement_type: str
    industry: str
    software_type: str
    client_introduction: str

class SessionCreateRequest(BaseModel):
    project_id: str
    meeting_title: str
    stakeholders: str  # Accept as a single string
    session_objective: str
    requirement_type: str
    type: str = "transcript"
    source: str = "meeting"
    meeting_id: Optional[str] = None
    followup_questions: bool = False
    file_uploaded: bool = False  # New field to track if file is uploaded

class FinalizeMetadataRequest(BaseModel):
    session_id: str
    chunk_index: int = 0

# metadata_dict = finalize_metadata()
# print(metadata_dict)

# @app.post("/projects")
# def create_project(req: ProjectCreateRequest):
#     project_id = str(uuid.uuid4())
#     projects_db[project_id] = req.dict()
#     return {"project_id": project_id, **req.dict()}

# @app.post("/sessions")
# def create_session(req: SessionCreateRequest):
#     session_id = str(uuid.uuid4())
#     # Generate meeting_id if not provided
#     meeting_id = req.meeting_id or f"mtg_{uuid.uuid4().hex[:8]}"
#     session_data = req.dict()
#     session_data["meeting_id"] = meeting_id
#     sessions_db[session_id] = session_data
#     return {"session_id": session_id, **session_data}

# @app.post("/finalize_metadata")
# def finalize_metadata(req: FinalizeMetadataRequest):
#     session = sessions_db.get(req.session_id)
#     if not session:
#         return {"error": "Session not found"}
#     project_id = session.get("project_id")
#     project = projects_db.get(project_id, {})
#     # Merge project and session fields
#     manual_fields = {**project, **session}
#     metadata = Metadata(manual_fields)
#     metadata.generate_after_meeting(req.transcript_path)
#     final_metadata = metadata.get_metadata(chunk_index=req.chunk_index)
#     return final_metadata

def receive_metadata_from_api(metadata_dict: dict):
    """
    Function to receive and process metadata returned from the finalize_metadata API endpoint
    This can be called from api.py or any other script
    """
    print("=== METADATA RECEIVED FROM API ===")
    print("Metadata dictionary received from finalize_metadata endpoint:")
    print(metadata_dict)
    print("=== END OF RECEIVED METADATA ===")
    
    # You can now process the metadata however you need
    print(f"\nProcessing metadata:")
    print(f"Project Title: {metadata_dict.get('project_title', 'N/A')}")
    print(f"Meeting Title: {metadata_dict.get('meeting_title', 'N/A')}")
    print(f"Project Tags: {metadata_dict.get('project_tags', [])}")
    print(f"Timestamp: {metadata_dict.get('timestamp', 'N/A')}")
    
    # You can store it, modify it, or use it for other purposes
    # For example, save to database, generate reports, etc.
    
    return metadata_dict

def process_metadata_for_vector_db(metadata_dict: dict):
    """
    Process metadata specifically for vector database storage
    """
    print("=== PROCESSING FOR VECTOR DATABASE ===")
    
    # Extract key fields for vector database
    vector_data = {
        "project_title": metadata_dict.get('project_title', ''),
        "client_name": metadata_dict.get('client_name', ''),
        "meeting_title": metadata_dict.get('meeting_title', ''),
        "session_objective": metadata_dict.get('session_objective', ''),
        "project_tags": metadata_dict.get('project_tags', []),
        "stakeholders": metadata_dict.get('stakeholders', []),
        "timestamp": metadata_dict.get('timestamp', ''),
        "full_metadata": metadata_dict  # Keep the full metadata as well
    }
    
    print("Vector database data prepared:")
    print(vector_data)
    print("=== END OF VECTOR PROCESSING ===")
    
    return vector_data

