from fastapi import APIRouter, Body, Query, UploadFile, File, Form, Path, Request
from pydantic import BaseModel
from common import create_json_response, run_in_threadpool
from constants import get_latest_main_transcript, BASE_DIR, TRANSCRIPT_MAP
from bot1 import extract_text_from_pdf, generate_questions_with_groq
from cleaner import convert_chat_to_transcript, get_latest_transcript
from MeetBot import start_meeting_bot, leave_meeting
from Metadata import ProjectCreateRequest, FinalizeMetadataRequest, Metadata, set_full_metadata
import uuid, os, asyncio
from typing import List
from datetime import datetime
from fastapi.responses import JSONResponse, PlainTextResponse
import re
import time




router = APIRouter()

projects_db = {}
sessions_db = {}
questions_db = {}
latest_metadata = None
bot_startup_lock = asyncio.Lock()
bot_state = "idle"  # "idle", "starting", "running", "stopping"
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
    global last_join_time, bot_state
    
    async with bot_startup_lock:
        try:
            # Check current bot state
            if bot_state in ["starting", "running", "stopping"]:
                return create_json_response(
                    {
                        "status": "error",
                        "bot_running": True,
                        "message": f"Bot is currently {bot_state}. Please wait.",
                        "bot_state": bot_state
                    },
                    status_code=409,  # Conflict
                )

            # Validate the meeting ID
            meeting_id = request.meeting_id.strip() if request.meeting_id else ""
            if not meeting_id:
                return create_json_response(
                    {
                        "status": "error",
                        "bot_running": False,
                        "message": "Meeting ID is required",
                    },
                    status_code=400,
                )

            # Set state to starting
            bot_state = "starting"
            last_join_time = time.time()

            # Construct the full Google Meet URL
            meeting_url = f"https://meet.google.com/{meeting_id}"
            
            # Use run_in_threadpool for the synchronous start_meeting_bot function
            started = await run_in_threadpool(start_meeting_bot, meeting_url)
            
            if started:
                # Give browser time to fully initialize
                await asyncio.sleep(2)
                bot_state = "running"
                
                return create_json_response(
                    {
                        "status": "started",
                        "message": "Bot joined the meeting successfully",
                        "meeting_id": meeting_id,
                        "meeting_url": meeting_url,
                        "bot_state": bot_state
                    }
                )
            else:
                bot_state = "running"  # Already running case
                return create_json_response(
                    {
                        "status": "already running",
                        "message": "Bot is already running",
                        "meeting_id": meeting_id,
                        "meeting_url": meeting_url,
                        "bot_state": bot_state
                    }
                )
                
        except Exception as e:
            bot_state = "idle"  # Reset state on error
            return create_json_response(
                {
                    "status": "error",
                    "bot_running": False,
                    "message": f"Error starting bot: {str(e)}",
                    "bot_state": bot_state
                },
                status_code=500,
            )

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
        session_data = {
            "project_id": project_id,
            "meeting_title": meeting_title,
            "stakeholders": stakeholders,
            "session_objective": session_objective,
            "requirement_type": requirement_type,
            "followup_questions": followup_questions,
            "file_uploaded": file_uploaded,
            "session_id": session_id,
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

# @router.post("/questions/generated")
# async def generate_questions_from_pdf(
#     file: UploadFile = File(...),
#     session_id: str = Query(..., description="Session ID to associate questions with"),
# ):
#     # Check if session exists and has file_uploaded flag
#     session = sessions_db.get(session_id)
#     if not session:
#         return create_json_response({"error": "Session not found"}, status_code=404)
    
#     if not session.get("file_uploaded", False):
#         return create_json_response(
#             {
#             "error": "File upload not enabled for this session. Please set file_uploaded to true when creating the session."
#             },
#             status_code=400,
#         )

#     if not file.filename.lower().endswith(".pdf"):
#         return create_json_response(
#             {"error": "Only PDF files are supported."}, status_code=400
#         )
#     try:
#         contents = await file.read()
#         temp_path = os.path.join(BASE_DIR, file.filename)
#         with open(temp_path, "wb") as f:
#             f.write(contents)
#         text = extract_text_from_pdf(temp_path)
#         os.remove(temp_path)
#         questions = generate_questions_with_groq(text, num_questions=10)

#         # Format questions as required
#         data_formats = ["json", "xml", "json", "csv"]
#         questions = [
#             {
#                 "id": f"func-req-{idx+1:03d}",
#                 "answer_key": f"q{idx+1:03d}",
#                 "question": q,
#                 "category": "Functional Requirement",
#                 "dataFormat": data_formats[idx % len(data_formats)],
#                 "answer": "",
#                 "source": "",
#             }
#             for idx, q in enumerate(questions)
#         ]
        
#         # Store questions in memory
#         questions_db[session_id] = questions
        
#         response = {
#             "status": "success",
#             "message": "Questions generated and stored successfully.",
#             **session,
#             "data": questions,
#         }
#         return create_json_response(response)
#     except Exception as e:
#         return create_json_response({"error": str(e)}, status_code=500)

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
        metadata = Metadata(manual_fields)
        metadata.generate_after_meeting(req.transcript_path)
        final_metadata = metadata.get_metadata(chunk_index=req.chunk_index)
        
        # Store globally for backend access
        global latest_metadata
        latest_metadata = final_metadata
        
        print("=== FULL METADATA ===")
        print(final_metadata)
        print("=== END OF FULL METADATA ===")
        
        # Send full metadata to Metadata.py
        set_full_metadata(final_metadata)
        
        # Return the metadata - this is what you'll get from the API call
        return create_json_response(final_metadata)
    except Exception as e:
        return create_json_response(
            {"error": f"Error finalizing metadata: {str(e)}"}, status_code=500
        )


@router.post("/process-meeting-end")
async def process_meeting_end(req: ProcessMeetingEndRequest = Body(...)):
    """Manually trigger the meeting end processing workflow with session_id and make bot leave"""
    global last_join_time, bot_state
    
    async with bot_startup_lock:
        try:
            # Check if bot is in a valid state to stop
            if bot_state == "idle":
                return create_json_response(
                    {
                        "status": False,
                        "message": "No bot is currently running",
                        "bot_state": bot_state
                    },
                    status_code=400,
                )
            
            if bot_state == "stopping":
                return create_json_response(
                    {
                        "status": False,
                        "message": "Bot is already being stopped. Please wait.",
                        "bot_state": bot_state
                    },
                    status_code=409,
                )

            # If bot is still starting, wait for it to finish or timeout
            if bot_state == "starting":
                print("⏳ Bot is still starting. Waiting for it to complete...")
                wait_count = 0
                max_wait = 15  # Maximum 15 seconds to wait
                
                while bot_state == "starting" and wait_count < max_wait:
                    await asyncio.sleep(1)
                    wait_count += 1
                
                if bot_state == "starting":
                    # Force reset if stuck in starting state
                    bot_state = "idle"
                    return create_json_response(
                        {
                            "status": False,
                            "message": "Bot startup timed out. State reset.",
                            "bot_state": bot_state
                        },
                        status_code=408,  # Request Timeout
                    )

            # Check if enough time has passed since joining
            time_since_start = time.time() - last_join_time
            if last_join_time > 0 and time_since_start < MIN_STARTUP_TIME:
                wait_time = MIN_STARTUP_TIME - time_since_start
                return create_json_response(
                    {
                        "status": False,
                        "message": f"Please wait {wait_time:.1f} more seconds before ending the meeting. Bot is still stabilizing.",
                        "retry_after": wait_time,
                        "bot_state": bot_state
                    },
                    status_code=429,
                )

            # Set state to stopping
            bot_state = "stopping"

            session_id = req.session_id.strip() if req.session_id else ""
            print(f"🔍 Debug: Received request with session_id: {session_id}")
            
            session = sessions_db.get(session_id)
            if not session:
                bot_state = "idle"  # Reset state
                print(f"❌ Session {session_id} not found")
                return create_json_response(
                    {
                        "status": False,
                        "message": "Session not found",
                        "debug_info": {
                            "requested_session": session_id,
                            "available_sessions": list(sessions_db.keys()),
                        },
                        "bot_state": bot_state
                    },
                    status_code=404,
                )

            print(f"✅ Session found. Attempting to leave meeting...")

            # Properly shut down bot using run_in_threadpool
            try:
                from MeetBot import leave_meeting

                leave_success = await run_in_threadpool(leave_meeting)
                
                # Give extra time for cleanup
                await asyncio.sleep(2)
                
            except Exception as leave_err:
                print(f"⚠️ Bot shutdown failed: {leave_err}")
                leave_success = False

            # Reset state to idle after cleanup
            bot_state = "idle"
            last_join_time = 0

            # Try to process transcript
            from cleaner import convert_chat_to_transcript, get_latest_transcript

            result = await run_in_threadpool(convert_chat_to_transcript)
            if not result:
                result = await run_in_threadpool(get_latest_transcript)

            meeting_id = "fch-wctx-ujo"
            generated_count = 0

            if result:
                try:
                    transcript_path = os.path.join(BASE_DIR, result)
                    if not os.path.exists(transcript_path):
                        raise FileNotFoundError("Transcript file not found.")

                    with open(transcript_path, "r", encoding="utf-8") as f:
                        transcript_text = f.read()

                    from cleaner import llm
                    questions = questions_db.get(session_id, [])

                    for question in questions:
                        if question.get("source") == "manual" and question.get("answer"):
                            continue

                        try:
                            prompt = f"""
                            Based on the following meeting transcript, answer this specific question:

                            Question: {question.get('question', '')}

                            Meeting Transcript:
                            {transcript_text[:3000]}

                            Provide a clear, concise answer based only on the information in the transcript.
                            If the answer is not found in the transcript or is unclear, respond with exactly "NOT_DISCUSSED".
                            Keep the answer to 1-2 sentences maximum.
                            Only provide an answer if the information is explicitly discussed in the transcript.
                            """

                            result_llm = await run_in_threadpool(llm.invoke, prompt)
                            answer = (
                                result_llm.content.strip()
                                if hasattr(result_llm, "content")
                                else str(result_llm).strip()
                            )
                            answer = answer.replace("\n", " ").strip()

                            if any(
                                token in answer.upper()
                                for token in [
                                    "NOT_DISCUSSED",
                                    "NOT DISCUSSED", 
                                    "UNKNOWN",
                                    "UNCLEAR",
                                    "NOT FOUND IN THE TRANSCRIPT",
                                ]
                            ):
                                question["answer"] = ""
                                question["source"] = ""
                            else:
                                question["answer"] = answer
                                question["source"] = "generated"

                            generated_count += 1

                        except Exception as llm_error:
                            print(f"⚠️ LLM error for question {question.get('id', 'unknown')}: {llm_error}")
                            question["answer"] = ""
                            question["source"] = ""
                            generated_count += 1

                    questions_db[session_id] = questions
                    print(f"✅ Generated answers for {generated_count} questions")

                except Exception as answer_error:
                    print(f"❌ Error during answer generation: {answer_error}")

                return create_json_response(
                    {
                        "status": True,
                        "message": "Meeting processing completed successfully",
                        "session_id": session_id,
                        "meeting_id": meeting_id,
                        "bot_left": leave_success,
                        "transcript_file": result,
                        "answers_generated": generated_count,
                        "documents_generated": [],
                        "bot_state": bot_state
                    }
                )

            else:
                return create_json_response(
                    {
                        "status": False,
                        "message": "Bot left the meeting, but no transcript was captured. Captions may have been disabled or no one spoke.",
                        "session_id": session_id,
                        "meeting_id": meeting_id,
                        "bot_left": leave_success,
                        "transcript_file": None,
                        "answers_generated": 0,
                        "documents_generated": [],
                        "bot_state": bot_state
                    },
                    status_code=200,
                )

        except Exception as e:
            import traceback
            
            # Reset state on any error
            bot_state = "idle"
            last_join_time = 0

            print(f"❌ Error in process_meeting_end: {str(e)}")
            print(traceback.format_exc())
            return create_json_response(
                {
                    "status": False,
                    "message": f"Error processing meeting end: {str(e)}",
                    "session_id": getattr(req, "session_id", "unknown"),
                    "bot_state": bot_state
                },
                status_code=500,
            )

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


@router.post("/projects")
async def create_project(req: ProjectCreateRequest):
    """Create a new project"""
    try:
        if not req:
            return create_json_response({
                "status": "error",
                "message": "Request body is required"
            }, status_code=400)
            
        project_id = str(uuid.uuid4())
        project_data = req.dict()
        projects_db[project_id] = project_data
        
        return create_json_response({
            "status": "success",
            "message": "Project created successfully",
            "project_id": project_id,
            **project_data
        })
    except Exception as e:
        return create_json_response({
            "status": "error",
            "message": f"Error creating project: {str(e)}"
        }, status_code=500)

@router.get("/get_all_sessions")
async def get_all_sessions():
    """Get all session IDs with their respective session details"""
    try:
        # Check if there are any sessions
        if not sessions_db:
            return create_json_response(
                {
                    "status": True,
                    "message": "No sessions found",
                    "total_sessions": 0,
                    "sessions": {}
                }
            )
        
        # Return all sessions with their details
        return create_json_response(
            {
                "status": True,
                "message": f"Retrieved {len(sessions_db)} sessions successfully",
                "total_sessions": len(sessions_db),
                "sessions": sessions_db
            }
        )
        
    except Exception as e:
        return create_json_response(
            {
                "status": False,
                "error_code": "UNEXPECTED_ERROR",
                "message": f"Error retrieving sessions: {str(e)}",
                "total_sessions": 0,
                "sessions": {}
            },
            status_code=500,
        )

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