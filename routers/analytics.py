from fastapi import APIRouter, Query, Request
from common import create_json_response
from constants import get_latest_main_transcript, BASE_DIR, TRANSCRIPT_MAP
from cleaner import llm
import os, json
from fastapi.responses import JSONResponse, PlainTextResponse
import asyncio

def run_in_threadpool(func, *args, **kwargs):
    """Run a synchronous function in a thread pool"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))

router = APIRouter()



@router.get("/action-items")
async def get_action_items(request: Request, meeting_id: str = Query(...), use_latest: bool = Query(True)):
    # Check for cleaned transcript file and return 202 if not ready
    latest_transcript = get_latest_main_transcript()
    if not latest_transcript:
        return JSONResponse(
            {
                "status": False,
                "message": "Cleaned transcript not yet available. Try again shortly.",
            },
            status_code=202,
        )  # 202 = Accepted but not ready

    try:
        meeting_id = meeting_id.strip()

        if not meeting_id:
            return create_json_response(
                {"status": False, "message": "Meeting ID cannot be empty."},
                status_code=400,
            )

        if use_latest:
            try:
                latest_transcript = get_latest_main_transcript()
                if not latest_transcript:
                    return create_json_response(
                        {"status": False, "message": "No transcript file found."},
                        status_code=404,
                    )
                transcript_path = os.path.join(BASE_DIR, latest_transcript)
            except Exception:
                return create_json_response(
                    {"status": False, "message": "Error locating transcript file."},
                    status_code=500,
                )
        else:
            transcript_path = os.path.join(BASE_DIR, "transcript_3.txt")

        if not os.path.exists(transcript_path):
            return create_json_response(
                {"status": False, "message": "Transcript file not found."},
                status_code=404,
            )

        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            print(f"📄 Read transcript: {len(transcript_text)} characters")
        except Exception:
            return create_json_response(
                {"status": False, "message": "Error reading transcript file."},
                status_code=500,
            )

        if not transcript_text.strip():
            return create_json_response(
                {
                "status": False,
                    "message": "Transcript file is empty. Please ensure the meeting was recorded with captions enabled.",
                    "debug_info": {
                        "file_path": transcript_path,
                        "file_size": (
                            os.path.getsize(transcript_path)
                            if os.path.exists(transcript_path)
                            else 0
                        ),
                        "content_length": len(transcript_text),
                    },
                },
                status_code=400,
            )

        if len(transcript_text.strip()) < 50:
            return create_json_response(
                {
                "status": False,
                    "message": "Transcript too short to generate action items. Need at least 50 characters.",
                    "debug_info": {
                        "content_length": len(transcript_text.strip()),
                        "content_preview": transcript_text[:100],
                    },
                },
                status_code=400,
            )

        try:
            from cleaner import llm
        except ImportError:
            return create_json_response(
                {"status": False, "message": "Failed to import LLM."}, status_code=500
            )

        action_items_prompt = f"""
        Based on the following meeting transcript, generate a list of action items. 
        Each action item should include:
        - A clear, actionable task
        - Who is responsible (if mentioned)
        - Due date or timeline (if mentioned)
        - Priority level (High/Medium/Low)
        - Timestamp when it was mentioned (extract from [mm:ss] format in transcript)

        Meeting Transcript:
        {transcript_text[:3000]}

        Return ONLY a valid JSON object:
        {{
            "action_items": [
                {{
                    "task": "description of the task",
                    "assignee": "person responsible",
                    "due_date": "timeline or due date",
                    "priority": "High/Medium/Low",
                    "timestamp": "mm:ss when mentioned"
                }}
            ]
        }}
        """

        try:
            result = await run_in_threadpool(llm.invoke, action_items_prompt)
            content = (
                result.content.strip()
                if hasattr(result, "content")
                else str(result).strip()
            )
            cleaned_content = content

            for prefix in [
                "here are the action items:",
                "json format:",
                "the action items are:",
            ]:
                if cleaned_content.lower().startswith(prefix):
                    cleaned_content = cleaned_content[len(prefix) :].strip()

            if "```json" in cleaned_content:
                cleaned_content = (
                    cleaned_content.split("```json")[1].split("```")[-2].strip()
                )
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[-2].strip()

            start_idx = cleaned_content.find("{")
            end_idx = cleaned_content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_content = cleaned_content[start_idx : end_idx + 1]

            action_data = json.loads(cleaned_content)
            action_items = action_data.get("action_items", [])
        except Exception:
            action_items = [
                {
                "task": "Review meeting transcript and identify key action items",
                "assignee": "TBD",
                "due_date": "TBD",
                "priority": "Medium",
                    "timestamp": "TBD",
                }
            ]

        # Contradiction detection prompt
        contradictions_prompt = f"""
        Based on the following meeting transcript, identify any contradictions or changes in answers.
        Look for instances where someone said one thing and then later said something different.
        Focus only on actual contradictions in the conversation.

        Meeting Transcript:
        {transcript_text[:2000]}

        Return ONLY a valid JSON object:
        {{
            "contradictions": [
                {{
                    "before": {{"text": "original statement", "speaker": "person name", "timestamp": "mm:ss"}},
                    "after": {{"text": "contradictory statement", "speaker": "person name", "timestamp": "mm:ss"}},
                    "topic": "what topic this contradiction relates to"
                }}
            ]
        }}
        """

        try:
            result = await run_in_threadpool(llm.invoke, contradictions_prompt)
            content = (
                result.content.strip()
                if hasattr(result, "content")
                else str(result).strip()
            )
            cleaned_content = content

            for prefix in [
                "here are the contradictions:",
                "json format:",
                "the contradictions are:",
            ]:
                if cleaned_content.lower().startswith(prefix):
                    cleaned_content = cleaned_content[len(prefix) :].strip()

            if "```json" in cleaned_content:
                cleaned_content = (
                    cleaned_content.split("```json")[1].split("```")[-2].strip()
                )
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[-2].strip()

            start_idx = cleaned_content.find("{")
            end_idx = cleaned_content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_content = cleaned_content[start_idx : end_idx + 1]

            contradiction_data = json.loads(cleaned_content)
            contradictions = contradiction_data.get("contradictions", [])
        except Exception:
            contradictions = []

        return create_json_response(
            {
            "status": True,
            "message": "Action items and contradictions generated successfully.",
            "action_items": action_items,
            "contradictions": contradictions,
            "meeting_id": meeting_id,
                "transcript_file": os.path.basename(transcript_path),
            }
        )

    except Exception as e:
        return create_json_response(
            {"status": False, "message": f"Unexpected error: {str(e)}"}, status_code=500
        )

@router.get("/key-insights")
async def get_key_insights(request: Request, meeting_id: str = Query(...)):
    """
    Generate key insights from the cleaned transcript file.
    """
    print(f"🔍 KEY-INSIGHTS REQUEST RECEIVED:")
    print(f"   Meeting ID: {meeting_id}")
    print(f"   Request URL: {request.url}")
    print(f"   Request method: {request.method}")
    print(f"   Request headers: {dict(request.headers)}")
    
    try:
        meeting_id = meeting_id.strip()

        if not meeting_id:
            print("❌ Meeting ID is empty after strip")
            return create_json_response(
                {"status": False, "message": "Meeting ID cannot be empty."},
                status_code=400,
            )

        print(f"✅ Meeting ID validated: '{meeting_id}'")
        
        try:
            latest_transcript = get_latest_main_transcript()
            print(f"📄 Latest transcript found: {latest_transcript}")
            if not latest_transcript:
                print("❌ No transcript file found")
                return create_json_response(
                    {"status": False, "message": "No transcript file found."},
                    status_code=404,
                )
            transcript_path = os.path.join(BASE_DIR, latest_transcript)
            print(f"📁 Transcript path: {transcript_path}")
        except Exception as e:
            print(f"❌ Error locating transcript: {e}")
            return create_json_response(
                {"status": False, "message": "Error locating transcript file."},
                status_code=500,
            )

        if not os.path.exists(transcript_path):
            print(f"❌ Transcript file does not exist: {transcript_path}")
            return create_json_response(
                {"status": False, "message": "Transcript file not found."},
                status_code=404,
            )

        print(f"✅ Transcript file exists: {transcript_path}")
        
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            print(f"📖 Transcript read successfully: {len(transcript_text)} characters")
        except Exception as e:
            print(f"❌ Error reading transcript: {e}")
            return create_json_response(
                {"status": False, "message": "Error reading transcript file."},
                status_code=500,
            )

        if not transcript_text.strip():
            print("❌ Transcript is empty")
            return create_json_response(
                {"status": False, "message": "Transcript file is empty."},
                status_code=400,
            )

        if len(transcript_text.strip()) < 50:
            print(f"❌ Transcript too short: {len(transcript_text.strip())} characters")
            return create_json_response(
                {
                "status": False,
                    "message": "Transcript too short to generate insights.",
                },
                status_code=400,
            )

        print(f"✅ Transcript validation passed: {len(transcript_text.strip())} characters")
        
        try:
            from cleaner import llm
            print("✅ LLM imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import LLM: {e}")
            return create_json_response(
                {"status": False, "message": "Failed to import LLM."}, status_code=500
            )

        key_insights_prompt = f"""
        Based on the following meeting transcript, generate key insights and takeaways.
        Focus on:
        - Main decisions made
        - Important agreements reached
        - Key challenges identified
        - Strategic insights
        - Critical points discussed

        Meeting Transcript:
        {transcript_text[:2000]}

        Return ONLY a valid JSON object:
        {{
            "key_insights": [
                {{
                    "insight": "description of the insight",
                    "category": "decision/agreement/challenge/strategy/critical",
                    "importance": "High/Medium/Low",
                    "timestamp": "mm:ss when mentioned"
                }}
            ]
        }}
        """

        try:
            result = await run_in_threadpool(llm.invoke, key_insights_prompt)
            content = (
                result.content.strip()
                if hasattr(result, "content")
                else str(result).strip()
            )
            print("LLM raw output:", repr(content))
            cleaned_content = content

            for prefix in [
                "here are the key insights:",
                "json format:",
                "the insights are:",
            ]:
                if cleaned_content.lower().startswith(prefix):
                    cleaned_content = cleaned_content[len(prefix) :].strip()

            if "```json" in cleaned_content:
                cleaned_content = (
                    cleaned_content.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[1].strip()

            start_idx = cleaned_content.find("{")
            end_idx = cleaned_content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_content = cleaned_content[start_idx : end_idx + 1]

            insights_data = json.loads(cleaned_content)
            key_insights = insights_data.get("key_insights", [])
        except Exception as e:
            print("LLM/JSON error:", str(e))
            key_insights = [
                {
                "insight": "Review meeting transcript for key insights",
                "category": "critical",
                "importance": "High",
                    "timestamp": "TBD",
                }
            ]

        return create_json_response(
            {
            "status": True,
            "message": "Key insights generated successfully.",
            "key_insights": key_insights,
            "meeting_id": meeting_id,
                "transcript_file": os.path.basename(transcript_path),
            }
        )

    except Exception as e:
        return create_json_response(
            {"status": False, "message": f"Unexpected error: {str(e)}"}, status_code=500
        )


@router.get("/meeting-summary")
async def get_meeting_summary(request: Request, meeting_id: str = Query(...)):
    """
    Load the already generated meeting summary from a file named
    transcript_<meeting_id>_Meetingsummary.txt
    """
    try:
        # Basic validation
        if not meeting_id.strip():
            return create_json_response(
                {
                "status": False,
                "error_code": "MISSING_MEETING_ID",
                "message": "Meeting ID is required.",
                "summary": "",
                    "meeting_id": None,
                },
                status_code=400,
            )

        meeting_id = meeting_id.strip()
        
        # Get the latest transcript number and use that for the summary file
        latest_transcript = get_latest_main_transcript()
        if not latest_transcript:
            return create_json_response(
                {
                "status": False,
                "error_code": "NO_TRANSCRIPT_FOUND",
                "message": "No transcript files found.",
                "summary": "",
                    "meeting_id": meeting_id,
                },
                status_code=404,
            )
        
        # Extract the number from the latest transcript (e.g., "transcript_7.txt" -> "7")
        transcript_number = latest_transcript.replace("transcript_", "").replace(
            ".txt", ""
        )
        summary_filename = f"transcript_{transcript_number}_Meeting_Summary.txt"
        summary_path = os.path.join(BASE_DIR, summary_filename)

        if not os.path.exists(summary_path):
            return create_json_response(
                {
                "status": False,
                "error_code": "SUMMARY_NOT_FOUND",
                "message": f"No summary file found for meeting ID: {meeting_id}",
                "summary": "",
                    "meeting_id": meeting_id,
                },
                status_code=404,
            )

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = f.read().strip()
        except Exception as read_error:
            return create_json_response(
                {
                "status": False,
                "error_code": "SUMMARY_READ_ERROR",
                "message": f"Could not read summary file: {str(read_error)}",
                "summary": "",
                    "meeting_id": meeting_id,
                },
                status_code=500,
            )

        if not summary:
            return create_json_response(
                {
                "status": False,
                "error_code": "EMPTY_SUMMARY",
                "message": "Summary file is empty.",
                "summary": "",
                    "meeting_id": meeting_id,
                },
                status_code=400,
            )

        return create_json_response(
            {
            "status": True,
            "message": "Meeting summary loaded successfully.",
            "summary": summary,
            "meeting_id": meeting_id,
            "summary_file": summary_filename,
                "summary_length": len(summary),
            }
        )

    except Exception as e:
        return create_json_response(
            {
            "status": False,
            "error_code": "UNEXPECTED_ERROR",
            "message": f"Unexpected error while loading summary: {str(e)}",
            "summary": "",
                "meeting_id": meeting_id,
            },
            status_code=500,
        )


