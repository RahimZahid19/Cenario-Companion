from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from common import create_json_response, run_in_threadpool
from load_env import setup_env
import os
from datetime import datetime

# Setup environment
setup_env()

# Import LLM from langchain
from langchain.chat_models import init_chat_model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

router = APIRouter()

class RealtimeQuestionsResponse(BaseModel):
    status: str
    message: str
    questions: list
    transcript_length: int
    timestamp: str

@router.get("/realtime-questions")
async def get_realtime_questions(
    meeting_id: str = Query(None, description="Optional meeting ID for context")
):
    """
    Generate 3 follow-up questions based on the current content in chat.txt
    This API is designed to be called during meetings to enrich the transcript
    """
    try:
        # Read the current content from chat.txt
        chat_file_path = "chat.txt"
        
        if not os.path.exists(chat_file_path):
            return create_json_response({
                "status": "error",
                "message": "Chat file not found. Meeting may not have started yet.",
                "questions": [],
                "transcript_length": 0,
                "timestamp": datetime.now().isoformat()
            })
        
        # Read the current transcript content
        with open(chat_file_path, 'r', encoding='utf-8') as file:
            current_transcript = file.read().strip()
        
        # Check if there's any content to work with
        if not current_transcript:
            return create_json_response({
                "status": "success",
                "message": "No transcript content available yet. Please wait for the meeting to progress.",
                "questions": [],
                "transcript_length": 0,
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate follow-up questions using the LLM
        questions = await run_in_threadpool(generate_followup_questions, current_transcript, meeting_id)
        
        return create_json_response({
            "status": "success",
            "message": "Follow-up questions generated successfully",
            "questions": questions,
            "transcript_length": len(current_transcript),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return create_json_response({
            "status": "error",
            "message": f"Error generating real-time questions: {str(e)}",
            "questions": [],
            "transcript_length": 0,
            "timestamp": datetime.now().isoformat()
        }, status_code=500)

def generate_followup_questions(transcript: str, meeting_id: str = None) -> list:
    """
    Generate 3 follow-up questions based on the current transcript content
    """
    # Limit transcript to avoid token overflow (use last 3000 characters for recent context)
    recent_transcript = transcript[-3000:] if len(transcript) > 3000 else transcript
    
    context_info = f" for meeting ID: {meeting_id}" if meeting_id else ""
    
    prompt = f"""
    Based on the following ongoing meeting transcript{context_info}, generate exactly 3 insightful follow-up questions that would help enrich the conversation and gather more detailed information.

    The questions should:
    1. Be directly related to the topics being discussed
    2. Help clarify any ambiguous points
    3. Encourage deeper discussion of important topics
    4. Be practical and actionable
    5. Help uncover requirements or details that might be missing

    Format your response as a simple numbered list with no additional text:
    1. [Question 1]
    2. [Question 2]
    3. [Question 3]

    Current transcript content:
    {recent_transcript}
    """
    
    try:
        response = llm.invoke(prompt).content.strip()
        
        # Parse the response to extract questions
        questions = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.')) or line.startswith('•') or line.startswith('-')):
                # Remove numbering and clean up the question
                question = line.split('.', 1)[-1].strip() if '.' in line else line.strip(' -•')
                if question:
                    questions.append(question)
        
        # Ensure we have exactly 3 questions
        if len(questions) < 3:
            # If we have fewer than 3, add generic but helpful questions
            fallback_questions = [
                "Could you elaborate on the technical requirements discussed?",
                "What are the key success criteria for this project?",
                "Are there any potential challenges or risks we should consider?"
            ]
            questions.extend(fallback_questions[:3-len(questions)])
        
        return questions[:3]  # Return only the first 3 questions
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        # Return fallback questions if LLM fails
        return [
            "Could you provide more details about the current discussion?",
            "What are the next steps we should focus on?",
            "Are there any concerns or questions from your side?"
        ]

@router.get("/transcript-status")
async def get_transcript_status():
    """
    Get the current status of the chat.txt file (length, last modified, etc.)
    """
    try:
        chat_file_path = "chat.txt"
        
        if not os.path.exists(chat_file_path):
            return create_json_response({
                "status": "not_found",
                "message": "Chat file not found",
                "file_exists": False,
                "file_size": 0,
                "last_modified": None,
                "timestamp": datetime.now().isoformat()
            })
        
        # Get file stats
        file_stats = os.stat(chat_file_path)
        file_size = file_stats.st_size
        last_modified = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        
        # Read content to get character count
        with open(chat_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            char_count = len(content)
            line_count = len(content.split('\n')) if content else 0
        
        return create_json_response({
            "status": "success",
            "message": "Transcript status retrieved",
            "file_exists": True,
            "file_size": file_size,
            "character_count": char_count,
            "line_count": line_count,
            "last_modified": last_modified,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return create_json_response({
            "status": "error",
            "message": f"Error getting transcript status: {str(e)}",
            "file_exists": False,
            "timestamp": datetime.now().isoformat()
        }, status_code=500) 