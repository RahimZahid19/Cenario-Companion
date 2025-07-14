from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from common import create_json_response, run_in_threadpool
from load_env import setup_env
import os
from datetime import datetime
import re

# Setup environment
setup_env()

# Import LLM from langchain
from langchain.chat_models import init_chat_model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

router = APIRouter()

class QuestionData(BaseModel):
    question_id: str
    questions: str
    tag: str
    batch: str

class RealtimeQuestionsResponse(BaseModel):
    status: str
    message: str
    data: list[QuestionData]

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
                "data": []
            })
        
        # Read the current transcript content
        with open(chat_file_path, 'r', encoding='utf-8') as file:
            current_transcript = file.read().strip()
        
        # Check if there's any content to work with
        if not current_transcript:
            return create_json_response({
                "status": "success",
                "message": "No transcript content available yet. Please wait for the meeting to progress.",
                "data": []
            })
        
        # Generate follow-up questions using the LLM
        questions_data = await run_in_threadpool(generate_followup_questions, current_transcript, meeting_id)
        
        return create_json_response({
            "status": "success",
            "message": "Follow-up questions generated successfully",
            "data": questions_data
        })
        
    except Exception as e:
        return create_json_response({
            "status": "error",
            "message": f"Error generating real-time questions: {str(e)}",
            "data": []
        }, status_code=500)

def categorize_question(question: str) -> tuple[str, str]:
    """
    Categorize a question into tag and batch based on its content
    """
    question_lower = question.lower()
    
    # Define tags and their keywords
    tag_keywords = {
        "budget": ["cost", "budget", "price", "expense", "financial", "money", "pricing", "payment"],
        "technical": ["technical", "technology", "system", "architecture", "implementation", "integration", "api", "database", "performance"],
        "timeline": ["timeline", "schedule", "deadline", "timeframe", "duration", "when", "completion", "delivery"],
        "requirements": ["requirement", "specification", "feature", "functionality", "need", "must", "should"],
        "resources": ["resource", "team", "staff", "personnel", "skill", "expertise", "capacity"],
        "security": ["security", "authentication", "authorization", "privacy", "compliance", "access"],
        "scalability": ["scalable", "scale", "growth", "volume", "capacity", "performance", "load"],
        "integration": ["integration", "connect", "interface", "sync", "compatibility", "third-party"],
        "testing": ["test", "testing", "validation", "verification", "qa", "quality"],
        "deployment": ["deployment", "release", "launch", "go-live", "production", "environment"],
        "maintenance": ["maintenance", "support", "update", "upgrade", "monitoring", "backup"],
        "general": []  # fallback
    }
    
    # Define batches and their keywords
    batch_keywords = {
        "functional need": ["feature", "functionality", "capability", "behavior", "user", "business", "process"],
        "technical need": ["architecture", "system", "technical", "implementation", "technology", "infrastructure"],
        "business need": ["business", "commercial", "revenue", "market", "strategy", "roi", "value"],
        "operational need": ["operation", "process", "workflow", "procedure", "efficiency", "productivity"],
        "compliance need": ["compliance", "regulation", "standard", "policy", "governance", "audit"],
        "performance need": ["performance", "speed", "response", "throughput", "scalability", "optimization"],
        "security need": ["security", "privacy", "protection", "access", "authentication", "authorization"],
        "integration need": ["integration", "interface", "connectivity", "compatibility", "interoperability"],
        "general need": []  # fallback
    }
    
    # Find best matching tag
    best_tag = "general"
    for tag, keywords in tag_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            best_tag = tag
            break
    
    # Find best matching batch
    best_batch = "general need"
    for batch, keywords in batch_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            best_batch = batch
            break
    
    return best_tag, best_batch

def generate_followup_questions(transcript: str, meeting_id: str = None) -> list[dict]:
    """
    Generate 3 follow-up questions based on the current transcript content
    """
    # Limit transcript to avoid token overflow (use last 3000 characters for recent context)
    recent_transcript = transcript[-3000:] if len(transcript) > 3000 else transcript
    
    context_info = f" for meeting ID: {meeting_id}" if meeting_id else ""
    
    prompt = f"""
    Based on the following ongoing meeting transcript{context_info}, generate exactly 3 very short and concise follow-up questions.

    Requirements:
    1. Each question must be maximum 8-10 words
    2. Be directly related to the topics discussed
    3. Help clarify key points or gather missing details
    4. Use simple, direct language
    5. Focus on one specific aspect per question

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
                "What are the technical requirements?",
                "What's the project timeline?",
                "Any potential risks?"
            ]
            questions.extend(fallback_questions[:3-len(questions)])
        
        # Convert to the required format with categorization
        questions_data = []
        for i, question in enumerate(questions[:3]):
            # Generate sequential question ID
            question_id = f"Q{i+1}"
            
            # Categorize the question
            tag, batch = categorize_question(question)
            
            questions_data.append({
                "question_id": question_id,
                "questions": question,
                "tag": tag,
                "batch": batch
            })
        
        return questions_data
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        # Return fallback questions if LLM fails
        fallback_questions = [
            "What are the key details?",
            "What's the next step?",
            "Any concerns?"
        ]
        
        questions_data = []
        for i, question in enumerate(fallback_questions):
            question_id = f"Q{i+1}"
            tag, batch = categorize_question(question)
            
            questions_data.append({
                "question_id": question_id,
                "questions": question,
                "tag": tag,
                "batch": batch
            })
        
        return questions_data

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