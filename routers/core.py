from fastapi import APIRouter, Body, Query, UploadFile, File, Form, Path, Request
from pydantic import BaseModel
from common import create_json_response, run_in_threadpool
from constants import get_latest_main_transcript, BASE_DIR, TRANSCRIPT_MAP
from bot1 import extract_text_from_pdf, generate_questions_with_groq
from cleaner import convert_chat_to_transcript, get_latest_transcript
# Replace MeetBot import with Recall import
from Recall import start_meeting_bot, leave_meeting, is_bot_running, get_current_bot_id
from Metadata import ProjectCreateRequest, FinalizeMetadataRequest, Metadata, set_full_metadata
from VectorDatabaseStoring import read_transcript, get_embedding, upsert_to_pinecone, chunk_text, upsert_question_answer
import uuid, os, asyncio
from typing import List
from datetime import datetime
from fastapi.responses import JSONResponse, PlainTextResponse
import re
import time
from datetime import datetime
import uuid, os, asyncio, requests
import json

router = APIRouter()

projects_db = {}
sessions_db = {}
questions_db = {}
latest_metadata = None
bot_startup_lock = asyncio.Lock()
bot_state = "idle"  # "idle", "starting", "running", "stopping"
bot_join_task = None
current_meeting_id = None
last_join_time = 0
MIN_STARTUP_TIME = 3

class JoinMeetingRequest(BaseModel):
    meeting_id: str

class AnswerUpdate(BaseModel):
    id: str = None
    answer_key: str = None
    answer: str

class UpdateAnswersRequest(BaseModel):
    session_id: str
    answers: List[AnswerUpdate]

class ProcessMeetingEndRequest(BaseModel):
    session_id: str

@router.post("/join-meeting")
async def join_meeting(request: JoinMeetingRequest = Body(...)):
    global bot_state, bot_join_task, current_meeting_id, last_join_time
    
    async with bot_startup_lock:
        try:
            # Check if bot is already running or starting
            if bot_state in ["starting", "running"]:
                return create_json_response({
                    "status": "error",
                    "message": f"Bot is already {bot_state}. Please wait or stop the current session.",
                    "bot_running": True,
                    "bot_state": bot_state,
                    "current_meeting_id": current_meeting_id
                })
            
            # Validate meeting ID
            meeting_id = request.meeting_id.strip() if request.meeting_id else ""
            if not meeting_id:
                return create_json_response({
                    "status": "error",
                    "message": "Meeting ID is required",
                    "bot_running": False
                })
            
            # Construct the full Google Meet URL
            if meeting_id.startswith("https://meet.google.com/"):
                meeting_url = meeting_id
            else:
                meeting_url = f"https://meet.google.com/{meeting_id}"
            
            # Store the current meeting ID
            current_meeting_id = meeting_id
            bot_state = "starting"
            last_join_time = time.time()
            
            # Start Recall.ai bot in background
            async def start_bot_async():
                global bot_state, current_meeting_id
                try:
                    # Use Recall.ai bot instead of MeetBot
                    success = await run_in_threadpool(start_meeting_bot, meeting_url)
                    if success:
                        bot_state = "running"
                        print(f"✅ Recall.ai bot successfully joined meeting: {meeting_id}")
                    else:
                        bot_state = "idle"
                        current_meeting_id = None
                        print(f"❌ Failed to start Recall.ai bot for meeting: {meeting_id}")
                except Exception as e:
                    print(f"❌ Error starting Recall.ai bot: {e}")
                    bot_state = "idle"
                    current_meeting_id = None
                    
                    # Store the error for potential retrieval
                    global last_bot_error
                    last_bot_error = {
                        "timestamp": datetime.now().isoformat(),
                        "meeting_id": meeting_id,
                        "error": str(e)
                    }
            
            # Start the bot task
            bot_join_task = asyncio.create_task(start_bot_async())
            
            return create_json_response({
                "status": "success",
                "message": f"Recall.ai bot is starting to join meeting: {meeting_id}",
                "meeting_id": meeting_id,
                "meeting_url": meeting_url,
                "bot_running": True,
                "bot_state": "starting",
                "bot_type": "recall.ai"
            })
            
        except Exception as e:
            bot_state = "idle"
            current_meeting_id = None
            
            # Enhanced error message for Recall.ai
            error_str = str(e).lower()
            
            if "recall_api_key" in error_str:
                error_message = "Recall.ai API key is not set or invalid. Please check your .env file and ensure RECALL_API_KEY is configured."
            elif "authentication" in error_str:
                error_message = "Recall.ai authentication failed. Please check your API key and try again."
            elif "meeting_url" in error_str:
                error_message = "Invalid meeting URL format. Please provide a valid Google Meet URL."
            elif "network" in error_str or "connection" in error_str:
                error_message = "Network error connecting to Recall.ai API. Please check your internet connection and try again."
            else:
                error_message = f"Recall.ai bot failed to start: {str(e)}. Please check your configuration and try again."
            
            return create_json_response({
                "status": "error",
                "message": error_message,
                "bot_running": False,
                "bot_type": "recall.ai"
            }, status_code=500)

@router.post("/sessions")
async def create_session_with_file(
    project_id: str = Form(...),
    meeting_title: str = Form(...),
    stakeholders: str = Form(...),
    session_objective: str = Form(...),
    requirement_type: str = Form(...),
    followup_questions: bool = Form(False),
    file_uploaded: bool = Form(False),
    file: UploadFile = File(None),
):
    try:
        # Basic validation
        if not project_id.strip() or not meeting_title.strip():
            return create_json_response({
                "status": "error",
                "message": "Project ID and meeting title are required"
            }, status_code=400)
        session_id = str(uuid.uuid4())
        # Add timestamp to session data
        session_data = {
            "project_id": project_id,
            "meeting_title": meeting_title,
            "stakeholders": stakeholders,
            "session_objective": session_objective,
            "requirement_type": requirement_type,
            "followup_questions": followup_questions,
            "file_uploaded": file_uploaded,
            "session_id": session_id,
            "created_at": datetime.now().strftime("%Y-%m-%d")  # Add timestamp
        }
        sessions_db[session_id] = session_data
        questions = []
        if file and file.filename.endswith(".pdf"):
            try:
                path = os.path.join(BASE_DIR, f"{session_id}_{file.filename}")
                contents = await file.read()
                with open(path, "wb") as f:
                    f.write(contents)
                text = extract_text_from_pdf(path)
                os.remove(path)
                if text.strip():
                    raw_qs = await run_in_threadpool(lambda: generate_questions_with_groq(text, num_questions=10))
                    questions = [
                        {
                            "id": f"func-req-{i+1:03d}",
                            "answer_key": f"q{i+1:03d}",
                            "question": q,
                            "category": "Functional Requirement",
                            "dataFormat": "json",
                            "answer": "",
                            "source": ""
                        } for i, q in enumerate(raw_qs)
                    ]
                    questions_db[session_id] = questions
            except Exception as file_error:
                return create_json_response({
                    "status": "error",
                    "message": f"File processing error: {str(file_error)}"
                }, status_code=500)
        return create_json_response({
            "status": "success",
            "message": "Session created",
            **session_data,
            "data": questions
        })
    except Exception as e:
        return create_json_response({
            "status": "error",
            "message": f"Error creating session: {str(e)}"
        }, status_code=500)

@router.get("/cleaned-transcript")
async def get_cleaned_transcript_json(
    meeting_id: str = Query(..., description="Meeting ID to get cleaned transcript for")
):
    """Parse the latest transcript file and return structured transcript data."""
    try:
        # Enhanced validation for meeting_id parameter
        if not meeting_id:
            return create_json_response(
                {
                "status": False,
                "error_code": "MISSING_MEETING_ID",
                "message": "Meeting ID is required. Please provide a valid meeting_id parameter.",
                "document_type": "Cleaned Transcript",
                "data": [],
                "filename": "transcript",
                "total_messages": 0,
                "meeting_id": None,
                "debug_info": {
                    "provided_meeting_id": meeting_id,
                    "meeting_id_type": type(meeting_id).__name__,
                        "meeting_id_length": len(str(meeting_id)) if meeting_id else 0,
                    },
                },
                status_code=400,
            )
        
        # Check if meeting_id is a valid string and not just whitespace
        if not isinstance(meeting_id, str) or meeting_id.strip() == "":
            return create_json_response(
                {
                "status": False,
                "error_code": "INVALID_MEETING_ID",
                "message": "Meeting ID must be a non-empty string. Please provide a valid meeting_id parameter.",
                "document_type": "Cleaned Transcript",
                "data": [],
                "filename": "transcript",
                "total_messages": 0,
                "meeting_id": meeting_id,
                "debug_info": {
                    "provided_meeting_id": meeting_id,
                    "meeting_id_type": type(meeting_id).__name__,
                    "meeting_id_length": len(str(meeting_id)) if meeting_id else 0,
                        "is_whitespace_only": (
                            meeting_id.strip() == ""
                            if isinstance(meeting_id, str)
                            else False
                        ),
                    },
                },
                status_code=400,
            )
        
        # Clean the meeting_id
        meeting_id = meeting_id.strip()
        
        # Get the latest transcript file with enhanced error handling
        try:
            latest_transcript = get_latest_main_transcript()
            if not latest_transcript:
                return create_json_response(
                    {
                    "status": False,
                    "error_code": "NO_TRANSCRIPT_FOUND",
                    "message": "No transcript files found. Please ensure a meeting has been completed and transcript processing is finished.",
                    "document_type": "Cleaned Transcript",
                    "data": [],
                    "filename": "transcript",
                    "total_messages": 0,
                    "meeting_id": meeting_id,
                    "debug_info": {
                        "base_dir": BASE_DIR,
                            "available_files": (
                                [
                                    f
                                    for f in os.listdir(BASE_DIR)
                                    if f.startswith("transcript_")
                                ]
                                if os.path.exists(BASE_DIR)
                                else []
                            ),
                            "suggestion": "Try running /api/process-transcript first to generate transcript files",
                        },
                    },
                    status_code=404,
                )
        except Exception as transcript_error:
            return create_json_response(
                {
                "status": False,
                "error_code": "TRANSCRIPT_LOOKUP_ERROR",
                "message": f"Error finding transcript file: {str(transcript_error)}",
                "document_type": "Cleaned Transcript",
                "data": [],
                "filename": "transcript",
                "total_messages": 0,
                "meeting_id": meeting_id,
                "debug_info": {
                    "error_type": type(transcript_error).__name__,
                    "base_dir": BASE_DIR,
                        "available_files": (
                            [
                                f
                                for f in os.listdir(BASE_DIR)
                                if f.startswith("transcript_")
                            ]
                            if os.path.exists(BASE_DIR)
                            else []
                        ),
                    },
                },
                status_code=500,
            )
        
        # Read the latest transcript file with enhanced error handling
        try:
            with open(latest_transcript, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"📄 Read transcript: {len(content)} characters")
        except UnicodeDecodeError as decode_error:
            return create_json_response(
                {
                "status": False,
                "error_code": "TRANSCRIPT_ENCODING_ERROR",
                "message": f"Error reading transcript file: encoding issue - {str(decode_error)}",
                "document_type": "Cleaned Transcript",
                "data": [],
                "filename": latest_transcript,
                "total_messages": 0,
                "meeting_id": meeting_id,
                "debug_info": {
                    "file_path": latest_transcript,
                        "file_size": (
                            os.path.getsize(latest_transcript)
                            if os.path.exists(latest_transcript)
                            else 0
                        ),
                        "encoding_error": str(decode_error),
                    },
                },
                status_code=500,
            )
        except PermissionError as perm_error:
            return create_json_response(
                {
                "status": False,
                "error_code": "TRANSCRIPT_PERMISSION_ERROR",
                "message": f"Permission denied reading transcript file: {str(perm_error)}",
                "document_type": "Cleaned Transcript",
                "data": [],
                "filename": latest_transcript,
                "total_messages": 0,
                "meeting_id": meeting_id,
                "debug_info": {
                    "file_path": latest_transcript,
                        "permission_error": str(perm_error),
                    },
                },
                status_code=500,
            )
        except Exception as read_error:
            return create_json_response(
                {
                "status": False,
                "error_code": "TRANSCRIPT_READ_ERROR",
                "message": f"Error reading transcript file: {str(read_error)}",
                "document_type": "Cleaned Transcript",
                "data": [],
                "filename": latest_transcript,
                "total_messages": 0,
                "meeting_id": meeting_id,
                "debug_info": {
                    "file_path": latest_transcript,
                    "read_error": str(read_error),
                        "error_type": type(read_error).__name__,
                    },
                },
                status_code=500,
            )

        # Check if content is empty
        if not content.strip():
            return create_json_response(
                {
                "status": False,
                "error_code": "EMPTY_TRANSCRIPT",
                "message": "Transcript file is empty. No content available for processing.",
                "document_type": "Cleaned Transcript",
                "data": [],
                "filename": latest_transcript,
                "total_messages": 0,
                "meeting_id": meeting_id,
                "debug_info": {
                    "file_path": latest_transcript,
                    "file_size": len(content),
                    "content_preview": content[:100] if content else "",
                        "suggestion": "Check if the meeting was recorded properly or if transcript processing completed successfully",
                    },
                },
                status_code=400,
            )

        lines = content.splitlines()
        data = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Match: [HH:MM:SS] Speaker: message
            match = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s*(.+?):\s*(.+)", line.strip())
            if match:
                timestamp_str, speaker, message = match.groups()
                data.append(
                    {
                    "timestamp": timestamp_str,
                    "speaker": speaker.strip(),
                        "message": message.strip(),
                    }
                )
            else:
                print(f"⚠️ Unrecognized line format: {line}")

        return create_json_response(
            {
            "status": True,
            "document_type": "Cleaned Transcript",
            "data": data,
            "filename": latest_transcript,
            "last_modified": datetime.now().isoformat(),
            "total_messages": len(data),
            "meeting_id": meeting_id,
            "processing_info": {
                "file_path": latest_transcript,
                "content_length": len(content),
                "lines_processed": len(lines),
                    "messages_extracted": len(data),
                },
            }
        )

    except Exception as e:
        return create_json_response(
            {
            "status": False,
            "error_code": "UNEXPECTED_ERROR",
            "message": f"Unexpected error processing cleaned transcript: {str(e)}",
            "document_type": "Cleaned Transcript",
            "data": [],
            "filename": "transcript",
            "total_messages": 0,
            "meeting_id": meeting_id,
            "debug_info": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                    "traceback": (
                        str(e.__traceback__) if hasattr(e, "__traceback__") else None
                    ),
                },
            },
            status_code=500,
        )


@router.post("/finalize_metadata")
async def finalize_metadata(req: FinalizeMetadataRequest):
    try:
        if not req or not req.session_id:
            return create_json_response({
                "error": "Session ID is required"
            }, status_code=400)
            
        print("Available session IDs:", list(sessions_db.keys()))
        print("Requested session_id:", req.session_id)

        session = sessions_db.get(req.session_id)
        if not session:
            return create_json_response({"error": "Session not found"}, status_code=404)

        project_id = session.get("project_id")
        project = projects_db.get(project_id, {})
        manual_fields = {**project, **session}

        # Automatically get the latest transcript file
        from cleaner import get_latest_transcript
        latest_transcript = get_latest_transcript()
        if not latest_transcript:
            return create_json_response({"error": "No transcript file found"}, status_code=404)
        
        transcript_path = os.path.join(BASE_DIR, latest_transcript)

        metadata = Metadata(manual_fields)
        metadata.generate_after_meeting(transcript_path)

        # Read and chunk the transcript
        transcript_text = read_transcript(transcript_path)
        chunks = chunk_text(transcript_text)

        for idx, chunk in enumerate(chunks):
            try:
                final_metadata = metadata.get_metadata(chunk_index=idx)
                final_metadata["text"] = chunk  # optional: store the chunk text itself
                embedding = get_embedding(chunk)
                upsert_to_pinecone(embedding, final_metadata)
            except Exception as e:
                print(f"Embedding or upsert failed for chunk {idx}: {e}")

        # Store last chunk’s metadata globally
        global latest_metadata
        latest_metadata = final_metadata

        set_full_metadata(final_metadata)

        return create_json_response({
            "message": "Metadata finalized and all chunks upserted", 
            "chunks": len(chunks),
            "transcript_file": latest_transcript
        })

    except Exception as e:
        return create_json_response(
            {"error": f"Error finalizing metadata: {str(e)}"}, status_code=500
        )

# ... existing code ...

@router.post("/process-meeting-end")
async def process_meeting_end(req: ProcessMeetingEndRequest = Body(...)):
    """Process meeting end using Recall.ai bot, generate answers for session questions, and finalize metadata"""
    global bot_state, bot_join_task, current_meeting_id
    
    async with bot_startup_lock:
        try:
            # Get session_id from request
            session_id = req.session_id.strip() if req.session_id else ""
            if not session_id:
                return create_json_response({
                    "status": "error",
                    "message": "Session ID is required",
                    "bot_running": False,
                    "bot_state": bot_state
                }, status_code=400)
            
            # Check if session exists
            session = sessions_db.get(session_id)
            if not session:
                return create_json_response({
                    "status": "error",
                    "message": "Session not found",
                    "debug_info": {
                        "requested_session": session_id,
                        "available_sessions": list(sessions_db.keys()),
                    },
                    "bot_state": bot_state
                }, status_code=404)
            
            # Check if there's a meeting to end
            if not current_meeting_id:
                return create_json_response({
                    "status": "error",
                    "message": "No active meeting to end",
                    "bot_running": False,
                    "bot_state": bot_state
                })
            
            meeting_to_end = current_meeting_id
            
            # If bot is starting, cancel the join task
            if bot_state == "starting" and bot_join_task:
                bot_join_task.cancel()
                try:
                    await bot_join_task
                except asyncio.CancelledError:
                    pass
                bot_join_task = None
            
            # Set state to stopping
            bot_state = "stopping"
            
            # Leave meeting using Recall.ai bot
            print("🔄 Ending Recall.ai bot session...")
            result = await run_in_threadpool(leave_meeting)
            
            # Reset state and clear meeting ID
            bot_state = "idle"
            bot_join_task = None
            current_meeting_id = None
            
            # Process transcript and generate answers
            generated_count = 0
            metadata_finalized = False
            chunks_processed = 0
            transcript_file = None
            
            # The leave_meeting function returns the transcript filename if successful
            if result:
                if isinstance(result, str):
                    # If result is a string, it's the transcript filename
                    transcript_file = result
                    leave_success = True
                else:
                    # If result is boolean, it's the leave success status
                    leave_success = result
                    # Try to get the latest transcript
                    try:
                        transcript_file = await run_in_threadpool(get_latest_transcript)
                    except Exception as e:
                        print(f"❌ Error getting latest transcript: {e}")
                        transcript_file = None
                
                if transcript_file:
                    try:
                        # Read transcript
                        transcript_path = os.path.join(BASE_DIR, transcript_file)
                        if os.path.exists(transcript_path):
                            with open(transcript_path, "r", encoding="utf-8") as f:
                                transcript_text = f.read()
                            
                            # Get questions for this session
                            questions = questions_db.get(session_id, [])
                            
                            if questions and transcript_text.strip():
                                from cleaner import llm
                                
                                # PARALLEL ANSWER GENERATION
                                async def generate_answer_async(question, question_index):
                                    try:
                                        prompt = f"""
                                        You are analyzing a meeting transcript to answer specific questions.
                                        
                                        Question: {question.get('question', '')}

                                        Meeting Transcript:
                                        {transcript_text[:4000]}

                                        Instructions:
                                        - Only answer if the information is clearly and explicitly discussed in the transcript
                                        - Provide a direct, factual answer in 1-2 sentences
                                        - Do not mention the transcript or meeting in your answer
                                        - Do not use phrases like "based on the transcript" or "according to the meeting"
                                        - If the information is not discussed, simply state "Information not available"
                                        """

                                        result_llm = await run_in_threadpool(llm.invoke, prompt)
                                        answer = (
                                            result_llm.content.strip()
                                            if hasattr(result_llm, "content")
                                            else str(result_llm).strip()
                                        )
                                        
                                        # Clean up the answer
                                        answer = answer.replace("\n", " ").strip()
                                        
                                        # Function to check if answer is valid and meaningful
                                        def is_valid_answer(ans):
                                            if not ans or len(ans) < 15:
                                                return False
                                            
                                            # Convert to lowercase for checking
                                            ans_lower = ans.lower()
                                            
                                            # Reject answers that contain these phrases
                                            reject_phrases = [
                                                "no_answer", "blank", "not_discussed", "information not available",
                                                "not discussed", "not mentioned", "not found", "not addressed",
                                                "not covered", "no information", "not specified", "not clear",
                                                "unclear", "not available", "not explicitly", "does not mention",
                                                "not provided", "not stated", "not indicated", "no details",
                                                "not elaborated", "not explained", "transcript does not",
                                                "meeting does not", "conversation does not", "discussion does not",
                                                "based on the transcript", "according to the meeting",
                                                "from the transcript", "in the transcript", "the meeting transcript",
                                                "are not explicitly discussed", "is not explicitly discussed",
                                                "focuses on", "but does not mention", "however", "the conversation focuses"
                                            ]
                                            
                                            # If answer contains any reject phrases, it's not valid
                                            if any(phrase in ans_lower for phrase in reject_phrases):
                                                return False
                                            
                                            # Check if it's mostly negative/explanatory text
                                            negative_words = ["not", "no", "does", "doesn't", "cannot", "can't", "unable", "without"]
                                            words = ans.split()
                                            negative_count = sum(1 for word in words if word.lower() in negative_words)
                                            
                                            # If more than 30% of words are negative, likely not a real answer
                                            if len(words) > 0 and (negative_count / len(words)) > 0.3:
                                                return False
                                            
                                            return True
                                        
                                        if is_valid_answer(answer):
                                            # Valid answer found
                                            question["answer"] = answer
                                            question["source"] = "transcript"
                                            print(f"✅ Generated answer for: {question.get('question', '')[:50]}...")
                                            return True
                                        else:
                                            # Not discussed - leave blank
                                            question["answer"] = ""
                                            question["source"] = "transcript"
                                            print(f"⚠️ No valid answer for: {question.get('question', '')[:50]}...")
                                            return False
                                            
                                    except Exception as e:
                                        print(f"❌ Error generating answer for question {question_index}: {e}")
                                        question["answer"] = ""
                                        question["source"] = "error"
                                        return False
                                
                                # Collect questions that need processing
                                tasks = []
                                questions_to_process = []
                                
                                for i, question in enumerate(questions):
                                    # Skip if already answered manually
                                    if question.get("source") == "manual" and question.get("answer"):
                                        continue
                                    
                                    tasks.append(generate_answer_async(question, i + 1))
                                    questions_to_process.append(question)
                                
                                # Run all tasks in parallel
                                if tasks:
                                    print(f"🚀 Processing {len(tasks)} questions in parallel...")
                                    start_time = time.time()
                                    
                                    results = await asyncio.gather(*tasks, return_exceptions=True)
                                    
                                    # Count successful generations (only count non-empty answers)
                                    generated_count = sum(1 for result in results if result is True)
                                    
                                    end_time = time.time()
                                    processing_time = end_time - start_time
                                    
                                    print(f"✅ Completed {generated_count}/{len(tasks)} questions successfully in {processing_time:.2f} seconds")
                                
                                # Update questions in database
                                questions_db[session_id] = questions
                            
                            # FINALIZE METADATA AND STORE IN VECTOR DATABASE
                            print(f"🔄 Finalizing metadata for session: {session_id}")
                            try:
                                project_id = session.get("project_id")
                                project = projects_db.get(project_id, {})
                                manual_fields = {**project, **session}

                                # Create metadata object and generate metadata
                                metadata = Metadata(manual_fields)
                                metadata.generate_after_meeting(transcript_path)

                                # Read and chunk the transcript
                                transcript_text = read_transcript(transcript_path)
                                chunks = chunk_text(transcript_text)

                                # Process each chunk and store in vector database
                                for idx, chunk in enumerate(chunks):
                                    try:
                                        final_metadata = metadata.get_metadata(chunk_index=idx)
                                        final_metadata["text"] = chunk  # Store the chunk text
                                        embedding = get_embedding(chunk)
                                        upsert_to_pinecone(embedding, final_metadata)
                                        chunks_processed += 1
                                    except Exception as e:
                                        print(f"❌ Embedding or upsert failed for chunk {idx}: {e}")

                                # Store last chunk's metadata globally
                                global latest_metadata
                                latest_metadata = final_metadata

                                set_full_metadata(final_metadata)
                                metadata_finalized = True

                                print(f"✅ Metadata finalized and {chunks_processed} chunks stored in vector database")

                            except Exception as e:
                                print(f"❌ Error finalizing metadata: {e}")
                                metadata_finalized = False
                                
                    except Exception as e:
                        print(f"❌ Error processing transcript: {e}")
                        transcript_file = None
            else:
                leave_success = False
            
            # Return success response
            return create_json_response({
                "status": "success",
                "message": f"Successfully processed meeting end for session: {session_id} using Recall.ai",
                "session_id": session_id,
                "meeting_id": meeting_to_end,
                "bot_left": leave_success,
                "transcript_file": transcript_file,
                "answers_generated": generated_count,
                "metadata_finalized": metadata_finalized,
                "chunks_processed": chunks_processed,
                "bot_running": False,
                "bot_state": "idle",
                "bot_type": "recall.ai"
            })
            
        except Exception as e:
            bot_state = "idle"
            bot_join_task = None
            current_meeting_id = None
            return create_json_response({
                "status": "error",
                "message": f"Error processing meeting end with Recall.ai: {str(e)}",
                "bot_running": False,
                "bot_state": "idle",
                "bot_type": "recall.ai"
            }, status_code=500)

@router.get("/questions/generated/{session_id}")
async def get_generated_questions(
    session_id: str = Path(..., min_length=1, description="Session ID cannot be empty"),
    request: Request = None,
):
    """Get generated questions for a specific session with proper 400 handling"""
    try:
        # Log the complete incoming request
        print(f"Incoming request URL: {request.url}")
        print(f":mag: Looking for session: {session_id}")
        print(f":clipboard: Available sessions: {list(sessions_db.keys())}")
        # Validate if session_id is well-formed
        if not session_id.strip():
            return JSONResponse(
                {
                "status": "error",
                "message": "Session ID cannot be empty or blank.",
                    "data": [],
                },
                status_code=400,
            )
        questions = questions_db.get(session_id, [])
        session = sessions_db.get(session_id)
        print(f":bar_chart: Session found: {session is not None}")
        print(f":bar_chart: Questions found: {len(questions)}")
        if not session:
            return JSONResponse(
                {
                "status": "error",
                "message": "Session not found.",
                "data": [],
                "debug_info": {
                    "requested_session_id": session_id,
                    "available_sessions": list(sessions_db.keys()),
                        "total_sessions": len(sessions_db),
                    },
                },
                status_code=404,
            )
        return JSONResponse(
            {
            "status": "success",
            "message": "Session and questions retrieved successfully.",
            "session": session,
            "data": questions,
                "total_questions": len(questions),
            }
        )
    except Exception as e:
        print(f":x: Error in get_generated_questions: {str(e)}")
        return JSONResponse(
            {
            "status": "error",
            "message": f"Internal server error: {str(e)}",
                "data": [],
            },
            status_code=500,
        )

@router.put("/sessions/{session_id}/file-uploaded")
async def update_session_file_uploaded(session_id: str):
    """Update session to mark file as uploaded"""
    try:
        session = sessions_db.get(session_id)
        if not session:
            return create_json_response(
                {"status": False, "message": "Session not found"}, status_code=404
            )
        
        # Update the file_uploaded flag
        session["file_uploaded"] = True
        sessions_db[session_id] = session
        
        return create_json_response(
            {
            "status": True,
            "message": "Session updated successfully",
            "session_id": session_id,
                "file_uploaded": True,
            }
        )
    except Exception as e:
        return create_json_response(
            {"status": False, "message": f"Error updating session: {str(e)}"},
            status_code=500,
        )

@router.get("/questions/answer-from-transcript")
async def answer_question_from_transcript(
    question: str = Query(..., description="Question to answer"),
    meeting_id: str = Query(..., description="Meeting ID to search in transcript"),
):
    """Answer a specific question using the meeting transcript"""
    try:
        # Enhanced validation for parameters
        if not question or not meeting_id:
            return create_json_response(
                {
                "status": False,
                "error_code": "MISSING_PARAMETERS",
                "message": "Both question and meeting_id parameters are required.",
                "answer": "",
                "question": question,
                    "meeting_id": meeting_id,
                },
                status_code=400,
            )
        
        # Clean the parameters
        question = question.strip()
        meeting_id = meeting_id.strip()
        
        # Get the latest transcript file
        try:
            latest_transcript = get_latest_main_transcript()
            if not latest_transcript:
                return create_json_response(
                    {
                    "status": False,
                    "error_code": "NO_TRANSCRIPT_FOUND",
                    "message": "No transcript files found. Please ensure a meeting has been completed and transcript processing is finished.",
                    "answer": "",
                    "question": question,
                        "meeting_id": meeting_id,
                    },
                    status_code=404,
                )
            
            transcript_path = os.path.join(BASE_DIR, latest_transcript)
        except Exception as transcript_error:
            return create_json_response(
                {
                "status": False,
                "error_code": "TRANSCRIPT_LOOKUP_ERROR",
                "message": f"Error finding transcript file: {str(transcript_error)}",
                "answer": "",
                "question": question,
                    "meeting_id": meeting_id,
                },
                status_code=500,
            )
        
        # Read the transcript file
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        except Exception as read_error:
            return create_json_response(
                {
                "status": False,
                "error_code": "TRANSCRIPT_READ_ERROR",
                "message": f"Error reading transcript file: {str(read_error)}",
                "answer": "",
                "question": question,
                    "meeting_id": meeting_id,
                },
                status_code=500,
            )
        
        # Check if transcript content is empty
        if not transcript_text.strip():
            return create_json_response(
                {
                "status": False,
                "error_code": "EMPTY_TRANSCRIPT",
                "message": "Transcript file is empty. No content available for processing.",
                "answer": "",
                "question": question,
                    "meeting_id": meeting_id,
                },
                status_code=400,
            )
        
        # Generate answer using LLM
        try:
            from cleaner import llm
        except ImportError as import_error:
            return create_json_response(
                {
                "status": False,
                "error_code": "LLM_IMPORT_ERROR",
                "message": f"Error importing LLM module: {str(import_error)}",
                "answer": "",
                "question": question,
                    "meeting_id": meeting_id,
                },
                status_code=500,
            )
        
        answer_prompt = f"""
        Based on the following meeting transcript, answer this specific question:
        
        Question: {question}
        
        Meeting Transcript:
        {transcript_text[:4000]}
        
        Provide a clear, concise answer based only on the information in the transcript.
        If the answer is not found in the transcript, say "The answer to this question was not discussed in the meeting."
        """
        
        try:
            result = await run_in_threadpool(llm.invoke, answer_prompt)
            answer = (
                result.content.strip()
                if hasattr(result, "content")
                else str(result).strip()
            )
            
            # Clean up the answer
            answer = answer.replace("\n", " ").strip()
            
        except Exception as llm_error:
            print(f"LLM error: {llm_error}")
            # Fallback answer
            answer = "Unable to generate answer from transcript. Please review the meeting content manually."
        
        return create_json_response(
            {
            "status": True,
            "message": "Answer generated successfully",
            "answer": answer,
            "question": question,
            "meeting_id": meeting_id,
                "transcript_file": os.path.basename(transcript_path),
            }
        )
        
    except Exception as e:
        return create_json_response(
            {
            "status": False,
            "error_code": "UNEXPECTED_ERROR",
            "message": f"Unexpected error generating answer: {str(e)}",
            "answer": "",
            "question": question,
                "meeting_id": meeting_id,
            },
            status_code=500,
        )


# Add this new Pydantic model at the top with other models
class CreateProjectRequest(BaseModel):
    project_title: str
    client_name: str
    issue_date: str
    proposal_deadline: str
    engagement_type: str
    industry: str
    software_type: str
    client_introduction: str

# ... existing code ...

@router.post("/projects")
async def create_project(request: CreateProjectRequest = Body(...)):
    """Create a new project using JSON data in request body
    
    Example request body:
    {
        "project_title": "CRM Integration Phase 1",
        "client_name": "XYZ Manufacturing",
        "issue_date": "2024-06-15",
        "proposal_deadline": "2024-07-01",
        "engagement_type": "Fixed Bid",
        "industry": "Manufacturing",
        "software_type": "CRM",
        "client_introduction": "XYZ Manufacturing is a leading provider of industrial solutions."
    }
    """
    try:
        # Basic validation
        if not request.project_title.strip() or not request.client_name.strip():
            return create_json_response({
                "status": "error",
                "message": "Project title and client name are required"
            }, status_code=400)
            
        project_id = str(uuid.uuid4())
        
        # Create project data from request body
        project_data = {
            "project_title": request.project_title.strip(),
            "client_name": request.client_name.strip(),
            "issue_date": request.issue_date.strip(),
            "proposal_deadline": request.proposal_deadline.strip(),
            "engagement_type": request.engagement_type.strip(),
            "industry": request.industry.strip(),
            "software_type": request.software_type.strip(),
            "client_introduction": request.client_introduction.strip(),
            "created_at": datetime.now().strftime("%Y-%m-%d")  # Add timestamp
        }
        
        projects_db[project_id] = project_data
        
        return create_json_response({
            "status": "success",
            "message": "Project created successfully",
            "project_id": project_id,
            **project_data
        })
        
    except Exception as e:
        return create_json_response(
            {
                "status": "error",
                "message": f"Error creating project: {str(e)}"
            },
            status_code=500,
        )


@router.get("/get_all_sessions")
async def get_all_sessions(project_id: str = Query(..., description="Project ID to filter sessions")):
    """Get all session IDs with their respective session details for a specific project from Pinecone"""
    try:
        # Validate project_id
        if not project_id or not project_id.strip():
            return create_json_response(
                {
                    "status": False,
                    "message": "Project ID is required",
                    "totalSessions": 0,
                    "data": []
                },
                status_code=400
            )
        
        project_id = project_id.strip()
        
        try:
            # Initialize Pinecone connection
            from pinecone import Pinecone
            import os
            
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
            
            # Check if namespace exists
            stats = index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            
            if project_id not in namespaces:
                return create_json_response({
                    "status": True,
                    "message": "No sessions found for this project",
                    "totalSessions": 0,
                    "data": []
                })
            
            # Query vectors from the namespace to get session metadata
            # Use a dummy vector to query all vectors in the namespace
            dummy_vector = [0.0] * 1024  # Cohere embed-english-v3.0 dimension
            
            query_response = index.query(
                vector=dummy_vector,
                top_k=10000,  # High number to get all vectors
                include_metadata=True,
                namespace=project_id,
                filter={"type": {"$ne": "requirement"}}  # Exclude requirement vectors
            )
            
            if not query_response.matches:
                # Try without filter to see what we get
                query_response = index.query(
                    vector=dummy_vector,
                    top_k=10000,
                    include_metadata=True,
                    namespace=project_id
                )
            
            if not query_response.matches:
                return create_json_response({
                    "status": True,
                    "message": "No sessions found for this project",
                    "totalSessions": 0,
                    "data": []
                })
            
            # Extract unique session data from vector metadata
            sessions_data = {}
            
            for match in query_response.matches:
                metadata = match.metadata
                session_id = metadata.get('session_id')
                
                # Skip requirement vectors as they don't have session metadata
                if metadata.get('type') == 'requirement':
                    continue
                
                if session_id and session_id not in sessions_data:
                    # Map the actual stored field names to the expected response format
                    session_info = {
                        "projectId": metadata.get('project_id', project_id),
                        "meetingTitle": metadata.get('meeting_title', 'Untitled Meeting'),
                        "stakeholders": metadata.get('stakeholders', []),
                        "sessionObjective": metadata.get('session_objective', ''),
                        "requirementType": metadata.get('requirement_type', ''),
                        "followupQuestions": metadata.get('followup_questions', False),
                        "fileUploaded": metadata.get('file_uploaded', False),
                        "sessionId": session_id,
                        "timestamp": metadata.get('created_at', datetime.now().strftime("%Y-%m-%d"))
                    }
                    sessions_data[session_id] = session_info
            
            # If no session data found from transcript chunks, create basic session info from requirement vectors
            if not sessions_data:
                for match in query_response.matches:
                    metadata = match.metadata
                    session_id = metadata.get('session_id')
                    
                    if session_id and session_id not in sessions_data:
                        # Create basic session info from available data
                        session_info = {
                            "projectId": metadata.get('project_id', project_id),
                            "meetingTitle": "Untitled Meeting",
                            "stakeholders": [],
                            "sessionObjective": "",
                            "requirementType": "",
                            "followupQuestions": False,
                            "fileUploaded": False,
                            "sessionId": session_id,
                            "timestamp": datetime.now().strftime("%Y-%m-%d")
                        }
                        sessions_data[session_id] = session_info
            
            # Convert to list for response
            sessions_list = list(sessions_data.values())
            
            # Sort by timestamp (most recent first)
            sessions_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return create_json_response({
                "status": True,
                "message": "Sessions retrieved successfully",
                "totalSessions": len(sessions_list),
                "data": sessions_list
            })
            
        except Exception as pinecone_error:
            print(f"Pinecone query error: {pinecone_error}")
            
            # Fallback to local sessions_db if Pinecone fails
            print("Falling back to local sessions database...")
            
            filtered_sessions = {
                session_id: session_data
                for session_id, session_data in sessions_db.items()
                if session_data.get("project_id") == project_id
            }
            
            if not filtered_sessions:
                return create_json_response({
                    "status": True,
                    "message": "No sessions found for this project",
                    "totalSessions": 0,
                    "data": []
                })
            
            # Format session data for response (fallback) - keeping exact structure
            sessions_list = []
            for session_id, session_data in filtered_sessions.items():
                session_info = {
                    "projectId": session_data.get("project_id"),
                    "meetingTitle": session_data.get("meeting_title"),
                    "stakeholders": session_data.get("stakeholders"),
                    "sessionObjective": session_data.get("session_objective"),
                    "requirementType": session_data.get("requirement_type"),
                    "followupQuestions": session_data.get("followup_questions"),
                    "fileUploaded": session_data.get("file_uploaded"),
                    "sessionId": session_id,
                    "timestamp": session_data.get("created_at", datetime.now().strftime("%Y-%m-%d"))
                }
                sessions_list.append(session_info)
            
            return create_json_response({
                "status": True,
                "message": "Sessions retrieved successfully",
                "totalSessions": len(sessions_list),
                "data": sessions_list
            })
            
    except Exception as e:
        print(f"Error in get_all_sessions: {e}")
        return create_json_response({
            "status": False,
            "message": f"Error retrieving sessions: {str(e)}",
            "totalSessions": 0,
            "data": []
        }, status_code=500)


@router.get("/projects")
async def get_all_projects():
    """Get all projects"""
    try:
        # Check if there are any projects
        if not projects_db:
            return create_json_response(
                {
                    "status": True,
                    "message": "No projects found",
                    "total_projects": 0,
                    "projects": {}
                }
            )
        
        # Return all projects with their details
        return create_json_response(
            {
                "status": True,
                "message": f"Retrieved {len(projects_db)} projects successfully",
                "total_projects": len(projects_db),
                "projects": projects_db
            }
        )
        
    except Exception as e:
        return create_json_response(
            {
                "status": False,
                "error_code": "UNEXPECTED_ERROR",
                "message": f"Error retrieving projects: {str(e)}",
                "total_projects": 0,
                "projects": {}
            },
            status_code=500,
        )        

@router.delete("/sessions/{identifier}")
async def delete_session(identifier: str = Path(..., description="Session ID or session name to delete")):
    """Delete a session by session_id or session name (meeting_title)"""
    try:
        if not identifier or not identifier.strip():
            return create_json_response(
                {
                    "status": False,
                    "message": "Session identifier is required",
                    "deleted": False
                },
                status_code=400,
            )
        
        identifier = identifier.strip()
        session_to_delete = None
        session_id_to_delete = None
        
        # First, try to find by session_id (exact match)
        if identifier in sessions_db:
            session_to_delete = sessions_db[identifier]
            session_id_to_delete = identifier
        else:
            # If not found by ID, search by meeting_title (session name)
            for session_id, session_data in sessions_db.items():
                if session_data.get("meeting_title", "").lower() == identifier.lower():
                    session_to_delete = session_data
                    session_id_to_delete = session_id
                    break
        
        # If session not found by either ID or name
        if not session_to_delete:
            return create_json_response(
                {
                    "status": False,
                    "message": f"Session not found with identifier: {identifier}",
                    "deleted": False,
                    "available_sessions": list(sessions_db.keys()),
                    "available_session_names": [
                        session.get("meeting_title", "Unnamed") 
                        for session in sessions_db.values()
                    ]
                },
                status_code=404,
            )
        
        # Delete the session from sessions_db
        deleted_session = sessions_db.pop(session_id_to_delete)
        
        # Also delete associated questions if they exist
        deleted_questions = []
        if session_id_to_delete in questions_db:
            deleted_questions = questions_db.pop(session_id_to_delete)
        
        return create_json_response(
            {
                "status": True,
                "message": f"Session deleted successfully",
                "deleted": True,
                "deleted_session_id": session_id_to_delete,
                "deleted_session_name": deleted_session.get("meeting_title", "Unnamed"),
                "deleted_session_data": deleted_session,
                "deleted_questions_count": len(deleted_questions),
                "remaining_sessions": len(sessions_db)
            }
        )
        
    except Exception as e:
        return create_json_response(
            {
                "status": False,
                "error_code": "UNEXPECTED_ERROR",
                "message": f"Error deleting session: {str(e)}",
                "deleted": False
            },
            status_code=500,
        )


@router.post("/filling_from_file")
async def filling_from_file(
    file: UploadFile = File(..., description="Document file to extract project information from")
):
    """
    Extract project information from uploaded document to pre-fill create project form.
    
    Accepts PDF files and extracts relevant project data including:
    - projectTitle, clientName, issueDate, proposalDeadline
    - engagementType, industry, softwareType, clientIntroduction
    
    Usage: Upload a document (SOW, proposal, etc.) and get structured data back
    to pre-fill the create project form fields.
    """
    try:
        # Validate file
        if not file:
            return create_json_response({
                "status": "error",
                "message": "File is required",
                "data": None
            }, status_code=400)
        
        # Check file type
        if not file.filename.lower().endswith(".pdf"):
            return create_json_response({
                "status": "error",
                "message": "Only PDF files are currently supported",
                "data": None
            }, status_code=400)
        
        # Create temporary file path
        temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(BASE_DIR, temp_filename)
        
        try:
            # Save uploaded file temporarily
            contents = await file.read()
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            # Extract text from PDF
            text = extract_text_from_pdf(temp_path)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            if not text.strip():
                return create_json_response({
                    "status": "error",
                    "message": "No readable text found in the document",
                    "data": None
                }, status_code=400)
            
            # Use LLM to extract project information
            extracted_data = await run_in_threadpool(
                lambda: extract_project_data_from_text(text)
            )
            
            return create_json_response({
                "status": "success",
                "message": "Project information extracted successfully",
                "data": extracted_data
            })
            
        except Exception as processing_error:
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return create_json_response({
                "status": "error",
                "message": f"Error processing file: {str(processing_error)}",
                "data": None
            }, status_code=500)
            
    except Exception as e:
        return create_json_response({
            "status": "error",
            "message": f"Error uploading file: {str(e)}",
            "data": None
        }, status_code=500)


def extract_project_data_from_text(text: str) -> dict:
    """
    Extract project-related information from document text using LLM.
    
    Args:
        text: The document text to analyze
        
    Returns:
        Dictionary containing extracted project information in camelCase
    """
    # Get Groq API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY not found in environment variables")
    
    # Debug: Print first 500 characters of extracted text
    print(f"🔍 DEBUG: Extracted text preview (first 500 chars):")
    print(f"'{text[:500]}...'")
    print(f"📏 Total text length: {len(text)} characters")
    
    # Updated prompt with camelCase field names
    prompt = f"""
    Extract project information from the following document text.
    
    Document Text:
    {text[:4000]}
    
    Extract the following information and return ONLY a valid JSON object with no additional text or explanation:
    
    {{
        "projectTitle": "The title or name of the project",
        "clientName": "The client or company name",
        "issueDate": "Document issue date or project start date (format: YYYY-MM-DD if available)",
        "proposalDeadline": "Proposal submission deadline (format: YYYY-MM-DD if available)",
        "engagementType": "Type of engagement (e.g., 'Fixed Price', 'Time & Materials', 'Retainer', etc.)",
        "industry": "Industry sector (e.g., 'Healthcare', 'Finance', 'Technology', 'Manufacturing', etc.)",
        "softwareType": "Type of software being developed (e.g., 'Web Application', 'Mobile App', 'Desktop Software', etc.)",
        "clientIntroduction": "Brief description of the client or project context"
    }}
    
    IMPORTANT: 
    - Return ONLY the JSON object above with the extracted values
    - If any information is not found or not provided, use an empty string ("")
    - No additional text, explanation, or notes
    - Use camelCase for field names exactly as shown
    """
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are a data extraction assistant. You return only valid JSON objects with no additional text or explanation. Never include introductory text or notes. If information is not provided, use empty strings. Use camelCase for field names."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # Debug: Print the raw LLM response
        print(f"🤖 DEBUG: Raw LLM response:")
        print(f"'{content}'")
        
        # Extract JSON from the response (handle cases where LLM adds extra text)
        json_content = extract_json_from_text(content)
        
        try:
            extracted_data = json.loads(json_content)
            print(f"✅ DEBUG: Successfully parsed JSON")
            
            # Validate that we have the expected structure (updated field names in camelCase)
            expected_fields = [
                "projectTitle", "clientName", "issueDate", "proposalDeadline",
                "engagementType", "industry", "softwareType", "clientIntroduction"
            ]
            
            # Create ordered dictionary with correct field order
            ordered_data = {}
            for field in expected_fields:
                if field in extracted_data:
                    # Keep extracted value, but ensure empty strings for missing data
                    ordered_data[field] = extracted_data[field] if extracted_data[field] else ""
                else:
                    # Field not found, use empty string
                    ordered_data[field] = ""
            
            # Debug: Show what was extracted
            print(f"📋 DEBUG: Extracted data:")
            for key, value in ordered_data.items():
                print(f"  {key}: '{value}'")
            
            return ordered_data
            
        except json.JSONDecodeError as e:
            print(f"❌ DEBUG: JSON parsing failed: {e}")
            print(f"Content that failed to parse: '{json_content}'")
            # If JSON parsing fails, return empty structure with correct field order in camelCase
            return {
                "projectTitle": "",
                "clientName": "",
                "issueDate": "",
                "proposalDeadline": "",
                "engagementType": "",
                "industry": "",
                "softwareType": "",
                "clientIntroduction": ""
            }
            
    except requests.RequestException as e:
        print(f"❌ DEBUG: API request failed: {e}")
        raise Exception(f"Error calling Groq API: {str(e)}")
    except Exception as e:
        print(f"❌ DEBUG: Unexpected error: {e}")
        raise Exception(f"Error extracting project data: {str(e)}")


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON object from text that may contain additional content.
    
    Args:
        text: Text that contains a JSON object
        
    Returns:
        Just the JSON object as a string
    """
    # Find the first '{' and last '}' to extract the JSON object
    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_content = text[start_index:end_index + 1]
        print(f"🔧 DEBUG: Extracted JSON: '{json_content}'")
        return json_content
    
    # If no JSON found, return the original text
    print(f"⚠️ DEBUG: No JSON brackets found, returning original text")
    return text        