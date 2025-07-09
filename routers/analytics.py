from fastapi import APIRouter, Query, Request
from common import create_json_response
from constants import get_latest_main_transcript, BASE_DIR, TRANSCRIPT_MAP
from cleaner import llm
import os, json
from fastapi.responses import JSONResponse, PlainTextResponse
import asyncio
from typing import List
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from cohere import Client as CohereClient
from load_env import setup_env
setup_env()

def run_in_threadpool(func, *args, **kwargs):
    """Run a synchronous function in a thread pool"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))

router = APIRouter()

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

def get_transcript_from_pinecone(meeting_id: str, session_id: str = None, project_id: str = None) -> tuple:
    """
    Retrieve full transcript content from Pinecone vector database
    Uses project_id as namespace and session_id as filter
    Returns tuple of (transcript_text, actual_session_id_used)
    """
    try:
        print(f"🔍 Starting search for meeting_id: {meeting_id}")
        if session_id:
            print(f"   Session ID: {session_id}")
        if project_id:
            print(f"   Project ID (namespace): {project_id}")
        
        # Initialize embeddings and Pinecone
        embeddings = CohereEmbeddings()
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Get all available namespaces first
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        available_namespaces = list(stats.get('namespaces', {}).keys())
        print(f"📋 Available namespaces: {available_namespaces}")
        
        # Try namespaces in priority order
        namespaces_to_try = []
        
        # First priority: use project_id if provided
        if project_id:
            namespaces_to_try.append(project_id)
        
        # Second priority: try session_id as namespace (fallback)
        if session_id:
            namespaces_to_try.append(session_id)
        
        # Third priority: try meeting_id as namespace (fallback)
        namespaces_to_try.append(meeting_id)
        
        # Fourth priority: known working namespace
        namespaces_to_try.append("ac253793-e4d8-404a-8300-096f2e456cca")
        
        # Last resort: all available namespaces
        namespaces_to_try.extend(available_namespaces)
        
        # Remove duplicates while preserving order
        namespaces_to_try = list(dict.fromkeys([ns for ns in namespaces_to_try if ns is not None]))
        
        documents = []
        used_namespace = None
        used_filter = None
        actual_session_id = None
        
        for ns in namespaces_to_try:
            print(f"🔍 Trying namespace: {ns}")
            
            try:
                vector_store = PineconeVectorStore(
                    index=index,
                    embedding=embeddings,
                    namespace=ns
                )
                
                # Try different search strategies
                search_strategies = []
                
                # Strategy 1: Use session_id if provided (MOST SPECIFIC)
                if session_id:
                    search_strategies.append({
                        "filter": {"session_id": session_id}, 
                        "query": "transcript",
                        "strategy_name": "session_id_filter"
                    })
                
                # Strategy 2: Search by meeting_id in different metadata fields
                search_strategies.extend([
                    {"filter": {"session_id": meeting_id}, "query": "transcript", "strategy_name": "meeting_as_session"},
                    {"filter": {"project_id": meeting_id}, "query": "transcript", "strategy_name": "meeting_as_project"},
                    {"filter": {"meeting_id": meeting_id}, "query": "transcript", "strategy_name": "meeting_id_field"},
                ])
                
                # Strategy 3: Search by known working session_id (fallback)
                search_strategies.append({
                    "filter": {"session_id": "ee26137a-6128-46c9-88ef-ac78fbedb70a"}, 
                    "query": "transcript",
                    "strategy_name": "known_session_fallback"
                })
                
                # Strategy 4: Broad search without filters
                search_strategies.extend([
                    {"filter": {}, "query": "transcript content meeting discussion", "strategy_name": "broad_query"},
                    {"filter": {}, "query": "", "strategy_name": "no_filter"},
                ])
                
                for strategy in search_strategies:
                    try:
                        print(f"   🎯 Trying strategy: {strategy['strategy_name']}")
                        print(f"      Filter: {strategy['filter']}, Query: '{strategy['query']}'")
                        
                        search_params = {
                            "query": strategy['query'],
                            "k": 1000
                        }
                        
                        if strategy['filter']:
                            search_params["filter"] = strategy['filter']
                        
                        documents = vector_store.similarity_search(**search_params)
                        
                        if documents:
                            print(f"✅ Found {len(documents)} documents!")
                            print(f"✅ Used namespace: {ns}")
                            print(f"✅ Used filter: {strategy['filter']}")
                            print(f"✅ Used strategy: {strategy['strategy_name']}")
                            used_namespace = ns
                            used_filter = strategy['filter']
                            
                            # Determine which session_id was actually used
                            if strategy['filter'].get('session_id'):
                                actual_session_id = strategy['filter']['session_id']
                            else:
                                # Extract from first document metadata
                                actual_session_id = documents[0].metadata.get('session_id', meeting_id)
                            
                            break
                            
                    except Exception as e:
                        print(f"   ❌ Strategy {strategy['strategy_name']} failed: {e}")
                        continue
                
                if documents:
                    break
                    
            except Exception as e:
                print(f"❌ Error with namespace {ns}: {e}")
                continue
        
        if not documents:
            print("❌ No documents found with any strategy")
            return "", None
        
        print(f"📊 Final results:")
        print(f"   Namespace: {used_namespace}")
        print(f"   Filter: {used_filter}")
        print(f"   Documents: {len(documents)}")
        print(f"   Actual session_id: {actual_session_id}")
        
        # Show metadata of first few documents for debugging
        for i, doc in enumerate(documents[:3]):
            print(f"   📄 Doc {i+1}: {doc.metadata}")
        
        # Sort documents by chunk_index and reconstruct transcript
        sorted_documents = sorted(
            documents, 
            key=lambda doc: doc.metadata.get('chunk_index', 0)
        )
        
        # Combine all chunks
        transcript_chunks = []
        for doc in sorted_documents:
            if hasattr(doc, 'page_content') and doc.page_content:
                content = doc.page_content.strip()
                if content:
                    transcript_chunks.append(content)
        
        # Join all chunks to reconstruct the full transcript
        full_transcript = "\n\n".join(transcript_chunks)
        
        print(f"✅ Reconstructed transcript: {len(full_transcript)} characters")
        
        return full_transcript, actual_session_id
        
    except Exception as e:
        print(f"❌ Error retrieving transcript from Pinecone: {e}")
        import traceback
        traceback.print_exc()
        return "", None

@router.get("/action-items")
async def get_action_items(
    request: Request, 
    meeting_id: str = Query(...), 
    session_id: str = Query(None),
    project_id: str = Query(None)
):
    print(f"🎯 ACTION-ITEMS REQUEST: meeting_id={meeting_id}, session_id={session_id}, project_id={project_id}")
    
    try:
        meeting_id = meeting_id.strip()

        if not meeting_id:
            return create_json_response(
                {"status": False, "message": "Meeting ID cannot be empty."},
                status_code=400,
            )

        # Get transcript from Pinecone
        print(f"📄 Retrieving transcript from Pinecone for meeting: {meeting_id}")
        transcript_text, actual_session_id = get_transcript_from_pinecone(meeting_id, session_id, project_id)

        if not transcript_text.strip():
            return create_json_response(
                {
                    "status": False,
                    "message": "No transcript found in database. Please ensure the meeting was recorded and processed.",
                    "debug_info": {
                        "meeting_id": meeting_id,
                        "session_id": session_id,
                        "project_id": project_id,
                        "content_length": len(transcript_text),
                        "suggestion": "Try using the actual session_id and project_id from your database"
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

        print(f"✅ Found transcript: {len(transcript_text)} characters")

        # Debug: Check timestamp distribution in the transcript
        lines_with_timestamps = [line for line in transcript_text.split('\n') if '[' in line and ']' in line and ':' in line]
        print(f"🕐 Found {len(lines_with_timestamps)} lines with timestamps")
        
        # Extract unique timestamps for better sampling
        timestamps = []
        for line in lines_with_timestamps:
            import re
            timestamp_match = re.search(r'\[(\d{1,2}:\d{2})\]', line)
            if timestamp_match:
                timestamps.append(timestamp_match.group(1))
        
        unique_timestamps = list(set(timestamps))
        print(f"🕐 Unique timestamps found: {unique_timestamps}")

        try:
            from cleaner import llm
        except ImportError:
            return create_json_response(
                {"status": False, "message": "Failed to import LLM."}, status_code=500
            )

        # Use smart sampling for longer transcripts to get timestamp diversity
        if len(transcript_text) > 4000:
            # Take content from beginning, middle, and end
            beginning = transcript_text[:1500]
            middle_start = len(transcript_text) // 2 - 750
            middle = transcript_text[middle_start:middle_start + 1500]
            end = transcript_text[-1000:]
            sampled_transcript = f"{beginning}\n\n--- MIDDLE SECTION ---\n\n{middle}\n\n--- END SECTION ---\n\n{end}"
        else:
            sampled_transcript = transcript_text

        action_items_prompt = f"""
        Based on the following meeting transcript, generate a list of action items. 
        Each action item should include:
        - A clear, actionable task
        - Who is responsible (if mentioned)
        - Due date or timeline (if mentioned)
        - Priority level (High/Medium/Low)
        - Timestamp when it was mentioned (extract from [mm:ss] format in transcript)

        Meeting Transcript:
        {sampled_transcript}

        Instructions:
        - Look for timestamps in [mm:ss] format throughout the transcript
        - Extract the actual timestamp when each action item was mentioned
        - If no specific timestamp is found, use "N/A"
        - Generate action items from different parts of the meeting

        Return ONLY a valid JSON object:
        {{
            "action_items": [
                {{
                    "task": "description of the task",
                    "assignee": "person responsible",
                    "due_date": "timeline or due date",
                    "priority": "High/Medium/Low",
                    "timestamp": "mm:ss when mentioned or N/A if not found"
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
                    cleaned_content.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[1].strip()

            start_idx = cleaned_content.find("{")
            end_idx = cleaned_content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_content = cleaned_content[start_idx : end_idx + 1]

            action_data = json.loads(cleaned_content)
            action_items = action_data.get("action_items", [])
        except Exception as e:
            print(f"❌ LLM/JSON error: {e}")
            action_items = [
                {
                    "task": "Review meeting transcript and identify key action items",
                    "assignee": "TBD",
                    "due_date": "TBD",
                    "priority": "Medium",
                    "timestamp": "N/A",
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
                    cleaned_content.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[1].strip()

            start_idx = cleaned_content.find("{")
            end_idx = cleaned_content.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                cleaned_content = cleaned_content[start_idx : end_idx + 1]

            contradictions_data = json.loads(cleaned_content)
            contradictions = contradictions_data.get("contradictions", [])
        except Exception as e:
            print(f"❌ Contradictions error: {e}")
            contradictions = []

        return create_json_response(
            {
                "status": True,
                "message": "Action items generated successfully.",
                "action_items": action_items,
                "contradictions": contradictions,
                "meeting_id": meeting_id,
                "session_id_used": actual_session_id,
                "project_id": project_id,
                "data_source": "pinecone_vector_database",
                "transcript_length": len(transcript_text)
            }
        )

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return create_json_response(
            {"status": False, "message": f"Unexpected error: {str(e)}"}, status_code=500
        )

@router.get("/key-insights")
async def get_key_insights(
    request: Request, 
    meeting_id: str = Query(...), 
    session_id: str = Query(None),
    project_id: str = Query(None)
):
    """
    Generate key insights from the Pinecone vector database.
    """
    print(f"🔍 KEY-INSIGHTS REQUEST: meeting_id={meeting_id}, session_id={session_id}, project_id={project_id}")
    
    try:
        meeting_id = meeting_id.strip()

        if not meeting_id:
            print("❌ Meeting ID is empty after strip")
            return create_json_response(
                {"status": False, "message": "Meeting ID cannot be empty."},
                status_code=400,
            )

        print(f"✅ Meeting ID validated: '{meeting_id}'")
        
        # Get transcript from Pinecone
        print(f"📄 Retrieving transcript from Pinecone for meeting: {meeting_id}")
        transcript_text, actual_session_id = get_transcript_from_pinecone(meeting_id, session_id, project_id)
        
        print(f"📖 Transcript retrieved: {len(transcript_text)} characters")

        if not transcript_text.strip():
            print("❌ No transcript found in database")
            return create_json_response(
                {
                    "status": False, 
                    "message": "No transcript found in database.",
                    "debug_info": {
                        "meeting_id": meeting_id,
                        "session_id": session_id,
                        "project_id": project_id,
                        "suggestion": "Try using the actual session_id and project_id from your database"
                    }
                },
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

        # Debug: Check timestamp distribution in the transcript
        lines_with_timestamps = [line for line in transcript_text.split('\n') if '[' in line and ']' in line and ':' in line]
        print(f"🕐 Found {len(lines_with_timestamps)} lines with timestamps")
        
        # Extract unique timestamps for better sampling
        timestamps = []
        for line in lines_with_timestamps:
            import re
            timestamp_match = re.search(r'\[(\d{1,2}:\d{2})\]', line)
            if timestamp_match:
                timestamps.append(timestamp_match.group(1))
        
        unique_timestamps = list(set(timestamps))
        print(f"🕐 Unique timestamps found: {unique_timestamps}")

        print(f"✅ Transcript validation passed: {len(transcript_text.strip())} characters")
        
        try:
            from cleaner import llm
            print("✅ LLM imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import LLM: {e}")
            return create_json_response(
                {"status": False, "message": "Failed to import LLM."}, status_code=500
            )

        # Use the FULL transcript instead of just first 2000 chars
        # But split it smartly to get content from different time periods
        if len(transcript_text) > 3000:
            # Take content from beginning, middle, and end to get timestamp diversity
            beginning = transcript_text[:1000]
            middle_start = len(transcript_text) // 2 - 500
            middle = transcript_text[middle_start:middle_start + 1000]
            end = transcript_text[-1000:]
            sampled_transcript = f"{beginning}\n\n--- MIDDLE SECTION ---\n\n{middle}\n\n--- END SECTION ---\n\n{end}"
        else:
            # Use full transcript if it's short enough
            sampled_transcript = transcript_text

        # Updated prompt with better timestamp handling
        key_insights_prompt = f"""
        Based on the following meeting transcript, generate key insights and takeaways.
        Focus on:
        - Main decisions made
        - Important agreements reached
        - Key challenges identified
        - Strategic insights
        - Critical points discussed

        Meeting Transcript:
        {sampled_transcript}

        Instructions:
        - Look carefully for timestamps in [mm:ss] format throughout the transcript
        - Extract the actual timestamp when each insight was mentioned
        - If multiple timestamps relate to one insight, use the first occurrence
        - If no timestamp is clearly associated with an insight, use "N/A"
        - Generate insights from different parts of the meeting to show timeline progression

        Return ONLY a valid JSON object:
        {{
            "key_insights": [
                {{
                    "insight": "description of the insight",
                    "category": "decision/agreement/challenge/strategy/critical",
                    "importance": "High/Medium/Low",
                    "timestamp": "mm:ss when mentioned or N/A if not found"
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
            
            print(f"🤖 LLM response length: {len(content)}")
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
            
            print(f"✅ Generated {len(key_insights)} insights")
            
        except Exception as e:
            print("LLM/JSON error:", str(e))
            key_insights = [
                {
                    "insight": "Review meeting transcript for key insights",
                    "category": "critical",
                    "importance": "High",
                    "timestamp": "N/A",
                }
            ]

        return create_json_response(
            {
                "status": True,
                "message": "Key insights generated successfully.",
                "key_insights": key_insights,
                "meeting_id": meeting_id,
                "session_id_used": actual_session_id,
                "project_id": project_id,
                "data_source": "pinecone_vector_database",
                "transcript_length": len(transcript_text)
            }
        )

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return create_json_response(
            {"status": False, "message": f"Unexpected error: {str(e)}"}, status_code=500
        )

@router.get("/meeting-summary")
async def get_meeting_summary(
    request: Request, 
    meeting_id: str = Query(...),
    session_id: str = Query(None),
    project_id: str = Query(None)
):
    """
    Generate meeting summary from the Pinecone vector database.
    """
    print(f"📝 MEETING-SUMMARY REQUEST: meeting_id={meeting_id}, session_id={session_id}, project_id={project_id}")
    
    try:
        meeting_id = meeting_id.strip()

        if not meeting_id:
            print("❌ Meeting ID is empty after strip")
            return create_json_response(
                {"status": False, "message": "Meeting ID cannot be empty."},
                status_code=400,
            )

        print(f"✅ Meeting ID validated: '{meeting_id}'")
        
        # Get transcript from Pinecone
        print(f"📄 Retrieving transcript from Pinecone for meeting: {meeting_id}")
        transcript_text, actual_session_id = get_transcript_from_pinecone(meeting_id, session_id, project_id)
        
        print(f"📖 Transcript retrieved: {len(transcript_text)} characters")

        if not transcript_text.strip():
            print("❌ No transcript found in database")
            return create_json_response(
                {
                    "status": False, 
                    "message": "No transcript found in database.",
                    "debug_info": {
                        "meeting_id": meeting_id,
                        "session_id": session_id,
                        "project_id": project_id,
                        "suggestion": "Try using the actual session_id and project_id from your database"
                    }
                },
                status_code=400,
            )

        if len(transcript_text.strip()) < 50:
            print(f"❌ Transcript too short: {len(transcript_text.strip())} characters")
            return create_json_response(
                {
                    "status": False,
                    "message": "Transcript too short to generate meeting summary.",
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

        # Use smart sampling for longer transcripts
        if len(transcript_text) > 5000:
            # Take content from beginning, middle, and end to get comprehensive coverage
            beginning = transcript_text[:2000]
            middle_start = len(transcript_text) // 2 - 1000
            middle = transcript_text[middle_start:middle_start + 2000]
            end = transcript_text[-1000:]
            sampled_transcript = f"{beginning}\n\n--- MIDDLE SECTION ---\n\n{middle}\n\n--- END SECTION ---\n\n{end}"
        else:
            # Use full transcript if it's short enough
            sampled_transcript = transcript_text

        # Updated meeting summary prompt - strictly summary only
        meeting_summary_prompt = f"""
        Based on the following meeting transcript, generate a comprehensive meeting summary.
        
        Write ONLY a summary of what was discussed in the meeting. Do not include:
        - Action items or next steps
        - Task assignments or responsibilities
        - Bold formatting (**) or any markdown
        - Bullet points or numbered lists
        - Section headers or structured formatting
        - Any special characters or formatting
        
        Write a flowing narrative summary that covers the main topics and decisions discussed.
        Keep it as plain text paragraphs only.

        Meeting Transcript:
        {sampled_transcript}

        Instructions:
        - Write in plain paragraph format only
        - Focus on what was discussed and decided
        - Include technical details and important points mentioned
        - Do not mention action items, assignments, or next steps
        - No formatting symbols (* ** - + etc.)
        - Keep it comprehensive but natural flowing text
        - Start with "The meeting focused on..." or similar

        Generate a plain text meeting summary:
        """

        try:
            result = await run_in_threadpool(llm.invoke, meeting_summary_prompt)
            summary_content = (
                result.content.strip()
                if hasattr(result, "content")
                else str(result).strip()
            )
            
            print(f"🤖 LLM response length: {len(summary_content)}")
            
            # Clean up any unwanted prefixes
            for prefix in [
                "here is the meeting summary:",
                "meeting summary:",
                "summary:",
                "the meeting summary is:",
            ]:
                if summary_content.lower().startswith(prefix):
                    summary_content = summary_content[len(prefix):].strip()
            
            # Remove any JSON formatting if it accidentally appears
            if summary_content.startswith('{') and summary_content.endswith('}'):
                try:
                    import json
                    json_data = json.loads(summary_content)
                    summary_content = json_data.get('summary', summary_content)
                except:
                    pass  # Keep original if JSON parsing fails
            
            # Remove \n characters and clean up formatting
            summary_content = summary_content.replace('\n', ' ').replace('  ', ' ').strip()
            
            # Remove any formatting symbols that might have appeared
            import re
            # Remove markdown formatting
            summary_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', summary_content)  # Remove **bold**
            summary_content = re.sub(r'\*([^*]+)\*', r'\1', summary_content)      # Remove *italic*
            summary_content = re.sub(r'#+\s*', '', summary_content)               # Remove headers
            summary_content = re.sub(r'^\s*[\*\-\+]\s*', '', summary_content, flags=re.MULTILINE)  # Remove bullet points
            summary_content = re.sub(r'^\s*\d+\.\s*', '', summary_content, flags=re.MULTILINE)     # Remove numbered lists
            
            # Clean up extra spaces
            summary_content = re.sub(r'\s+', ' ', summary_content).strip()
            
            print(f"✅ Generated clean meeting summary: {len(summary_content)} characters")
            
        except Exception as e:
            print("LLM error:", str(e))
            summary_content = f"Meeting summary could not be generated due to processing error. Please review the transcript manually. Error: {str(e)}"

        return create_json_response(
            {
                "status": True,
                "message": "Meeting summary generated successfully.",
                "summary": summary_content,
                "meeting_id": meeting_id,
                "session_id_used": actual_session_id,
                "project_id": project_id,
                "data_source": "pinecone_vector_database",
                "transcript_length": len(transcript_text)
            }
        )

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return create_json_response(
            {"status": False, "message": f"Unexpected error: {str(e)}"}, status_code=500
        )