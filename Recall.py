import requests
import os
import time
import json
from datetime import datetime
from load_env import setup_env

# Global variables to track bot state
bot_running = False
current_bot_id = None
current_headers = None

def setup_api():
    """Setup API configuration and validate environment variables."""
    setup_env()
    
    api_key = os.getenv("RECALL_API_KEY")
    if not api_key:
        raise ValueError("RECALL_API_KEY is not set. Make sure your .env file includes it.")
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Token {api_key}"
    }
    
    return headers

def clear_chat_file():
    """Clear chat.txt file for new meeting."""
    try:
        with open("chat.txt", "w", encoding="utf-8") as f:
            f.write("")
        print("✅ Cleared chat.txt for new meeting")
    except Exception as e:
        print(f"⚠️ Warning: Could not clear chat.txt: {e}")

def create_bot(headers, meeting_url):
    """Create and start a Recall.ai bot with transcription enabled."""
    url = "https://us-west-2.recall.ai/api/v1/bot/"
    
    data = {
        "meeting_url": meeting_url,
        "bot_name": "Cenario Companion",
        "recording_config": {
           "transcript": {
                "provider": {
                    "meeting_captions": {}
                }
            },
            "video_mixed_layout": "speaker_view",
            "start_recording_on": "participant_join"
        }
    }
    
    print("🚀 Starting Recall.ai bot...")
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code not in [200, 201]:
        print(f"❌ Failed to create bot. Status: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    bot_data = response.json()
    bot_id = bot_data["id"]
    
    print(f"✅ Bot created successfully!")
    print(f"📋 Bot ID: {bot_id}")
    print(f"🔗 Meeting URL: {bot_data['meeting_url']}")
    print(f"⏰ Join time: {bot_data['join_at']}")
    
    return bot_id

def leave_meeting_bot(headers, bot_id):
    """Send request to make the bot leave the meeting."""
    url = f"https://us-west-2.recall.ai/api/v1/bot/{bot_id}/leave_call/"
    
    print("\n🚪 Sending leave meeting request...")
    response = requests.post(url, headers=headers)
    
    if response.status_code not in [200, 201]:
        print(f"❌ Failed to leave meeting. Status: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    print("✅ Bot left the meeting successfully!")
    return True

def get_bot_status(headers, bot_id):
    """Get bot status and transcript data."""
    url = f"https://us-west-2.recall.ai/api/v1/bot/{bot_id}/"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch bot data. Status: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    return response.json()

def check_bot_actually_running():
    """Check if the bot is actually running by querying the Recall.ai API."""
    global bot_running, current_bot_id, current_headers
    
    if not bot_running or not current_bot_id or not current_headers:
        return False
    
    try:
        bot_data = get_bot_status(current_headers, current_bot_id)
        if not bot_data:
            # If we can't get bot data, assume it's not running
            reset_bot_state()
            return False
        
        # Check the bot's status
        status_changes = bot_data.get('status_changes', [])
        if status_changes:
            latest_status = status_changes[-1]
            bot_status = latest_status.get('code', 'unknown')
            
            # If bot is done, ended, or failed, it's not running anymore
            if bot_status in ['done', 'ended', 'failed', 'error']:
                print(f"🔄 Bot status is '{bot_status}', resetting state...")
                reset_bot_state()
                return False
        
        # If we get here, the bot is still active
        return True
        
    except Exception as e:
        print(f"⚠️ Error checking bot status: {e}")
        # If there's an error, assume it's not running and reset
        reset_bot_state()
        return False

def reset_bot_state():
    """Reset the global bot state variables."""
    global bot_running, current_bot_id, current_headers
    bot_running = False
    current_bot_id = None
    current_headers = None
    print("🔄 Bot state reset")

def wait_for_transcript_ready(headers, bot_id, max_wait=60):
    """Poll until transcript is ready or timeout."""
    print("⏱️ Waiting for transcription to finalize...")
    
    call_ended_count = 0
    max_call_ended_wait = 10  # Wait max 10 seconds after call_ended with no recordings
    
    for i in range(max_wait):
        bot_data = get_bot_status(headers, bot_id)
        if not bot_data:
            print(f"❌ Failed to get bot status during polling (attempt {i+1})")
            time.sleep(1)
            continue
            
        # Check if bot is done processing
        status_changes = bot_data.get('status_changes', [])
        recordings = bot_data.get('recordings', [])
        
        if status_changes:
            latest_status = status_changes[-1]
            bot_status = latest_status.get('code', 'unknown')
            
            if bot_status == 'done':
                # Bot is done, now check if transcript is ready
                if recordings:
                    for recording in recordings:
                        media_shortcuts = recording.get('media_shortcuts', {})
                        transcript_info = media_shortcuts.get('transcript')
                        
                        if transcript_info:
                            transcript_status = transcript_info.get('status', {})
                            transcript_code = transcript_status.get('code')
                            
                            if transcript_code == 'done':
                                print("✅ Transcript is ready!")
                                return bot_data
                            elif transcript_code == 'processing':
                                print(f"⏳ Transcript processing... ({i+1}/{max_wait}s)")
                            elif transcript_code == 'failed':
                                print("❌ Transcript processing failed!")
                                return bot_data
                            else:
                                print(f"⏳ Transcript status: {transcript_code} ({i+1}/{max_wait}s)")
                        else:
                            print(f"⏳ No transcript info available yet ({i+1}/{max_wait}s)")
                else:
                    print(f"⏳ No recordings available yet ({i+1}/{max_wait}s)")
            elif bot_status == 'call_ended':
                call_ended_count += 1
                if recordings:
                    print(f"⏳ Call ended, processing recordings... ({i+1}/{max_wait}s)")
                else:
                    print(f"⏳ Call ended, no recordings yet ({call_ended_count}/{max_call_ended_wait}s)")
                    
                    # If call ended and no recordings after reasonable wait, assume no content
                    if call_ended_count >= max_call_ended_wait:
                        print("⚠️ Call ended with no recordings after reasonable wait.")
                        print("💡 This usually means:")
                        print("   - No one spoke during the meeting")
                        print("   - Captions were not enabled in Google Meet")
                        print("   - The meeting was too short to generate recordings")
                        print("   - This was a test join/leave cycle")
                        return bot_data  # Return data even without recordings
            else:
                print(f"⏳ Bot status: {bot_status} ({i+1}/{max_wait}s)")
                call_ended_count = 0  # Reset counter if status changes
        else:
            print(f"⏳ No status changes available yet ({i+1}/{max_wait}s)")
        
        time.sleep(1)
    
    print(f"⚠️ Timeout after {max_wait} seconds - transcript may not be ready")
    return bot_data # Return whatever we have

def download_transcript(headers, download_url):
    """Download transcript from the provided URL."""
    print("📥 Downloading transcript...")
    response = requests.get(download_url, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Failed to download transcript. Status: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    try:
        return response.json()
    except json.JSONDecodeError:
        print("❌ Failed to parse transcript JSON")
        return None

def save_transcript_to_chat_file(transcript_data):
    """Save transcript data to chat.txt in the same format as MeetBot."""
    if not transcript_data:
        print("❌ No transcript data to save")
        return False
    
    try:
        # Process the transcript data and save to chat.txt
        with open("chat.txt", "w", encoding="utf-8") as f:
            if isinstance(transcript_data, list):
                for entry in transcript_data:
                    if isinstance(entry, dict):
                        # Extract participant information
                        participant = entry.get('participant', {})
                        speaker_name = participant.get('name', 'Unknown Speaker')
                        
                        # Extract words information
                        words = entry.get('words', [])
                        if words:
                            # Group words by timestamp to create natural sentences
                            current_sentence = ""
                            current_timestamp = None
                            
                            for word_info in words:
                                text = word_info.get('text', '').strip()
                                start_timestamp = word_info.get('start_timestamp', {})
                                relative_time = start_timestamp.get('relative', 0)
                                
                                if text:
                                    # Convert seconds to HH:MM:SS format
                                    hours = int(relative_time // 3600)
                                    minutes = int((relative_time % 3600) // 60)
                                    seconds = int(relative_time % 60)
                                    timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                                    
                                    # Start new sentence or continue current one
                                    if current_timestamp is None:
                                        current_timestamp = timestamp_str
                                        current_sentence = text
                                    else:
                                        # If time gap is small, continue sentence
                                        if abs(relative_time - (int(current_timestamp[-2:]) + int(current_timestamp[-5:-3])*60 + int(current_timestamp[:2])*3600)) < 5:
                                            current_sentence += " " + text
                                        else:
                                            # Save current sentence and start new one
                                            if current_sentence.strip():
                                                f.write(f"[{current_timestamp}] {speaker_name}: {current_sentence.strip()}\n")
                                            current_timestamp = timestamp_str
                                            current_sentence = text
                            
                            # Write the last sentence
                            if current_sentence.strip():
                                f.write(f"[{current_timestamp}] {speaker_name}: {current_sentence.strip()}\n")
            
        print("✅ Transcript saved to chat.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error saving transcript to chat.txt: {e}")
        return False

def process_transcript_and_save(transcript_data):
    """Process transcript data and save to chat.txt, then trigger cleaner."""
    if not transcript_data:
        print("❌ No transcript data available")
        return None
    
    recordings = transcript_data.get('recordings', [])
    
    if not recordings:
        print("⚠️ No recordings found. The bot may not have recorded anything.")
        print("💡 This could be because:")
        print("   - No one spoke during the meeting")
        print("   - Captions were not enabled in Google Meet")
        print("   - The meeting was too short")
        print("   - This was a test join/leave cycle")
        
        # Still clear chat.txt and create an empty transcript file for consistency
        clear_chat_file()
        try:
            from cleaner import convert_chat_to_transcript
            result = convert_chat_to_transcript()
            if result:
                print(f"✅ Empty transcript file created: {result}")
                return result
            else:
                print("⚠️ Could not create empty transcript file")
        except Exception as e:
            print(f"❌ Error creating empty transcript: {e}")
        
        return None
    
    print(f"📝 Processing {len(recordings)} recordings...")
    
    # Clear chat.txt first
    clear_chat_file()
    
    # Process each recording
    for i, recording in enumerate(recordings):
        print(f"\n--- Processing Recording {i+1} ---")
        
        # Check if transcript exists in media_shortcuts
        media_shortcuts = recording.get('media_shortcuts', {})
        transcript_info = media_shortcuts.get('transcript')
        
        if not transcript_info:
            print("⚠️ No transcript found in media_shortcuts")
            continue
            
        # Check if transcript is done
        transcript_status = transcript_info.get('status', {})
        transcript_code = transcript_status.get('code')
        
        if transcript_code != 'done':
            print(f"⚠️ Transcript not ready. Status: {transcript_code}")
            continue
            
        # Get download URL
        transcript_data_info = transcript_info.get('data', {})
        download_url = transcript_data_info.get('download_url')
        
        if not download_url:
            print("⚠️ No transcript download URL found")
            continue
        
        # Download and save transcript
        headers = setup_api()
        transcript_content = download_transcript(headers, download_url)
        
        if transcript_content:
            save_transcript_to_chat_file(transcript_content)
            
            # Process with existing cleaner
            try:
                from cleaner import convert_chat_to_transcript
                result = convert_chat_to_transcript()
                if result:
                    print(f"✅ Transcript processed successfully: {result}")
                    return result
                else:
                    print("⚠️ Transcript processing failed")
            except Exception as e:
                print(f"❌ Error processing transcript: {e}")
    
    return None

# Integration functions for FastAPI
def start_meeting_bot(meeting_url: str):
    """Start Recall.ai bot for the meeting (called from FastAPI)."""
    global bot_running, current_bot_id, current_headers
    
    try:
        # First, check if bot is actually running (not just the flag)
        if bot_running:
            print("🔍 Checking if bot is actually running...")
            if check_bot_actually_running():
                print("❌ Bot is actually still running in another meeting")
                return False
            else:
                print("✅ Bot state was stale, now reset. Proceeding with new meeting...")
        
        # Clear chat.txt when starting a new meeting
        clear_chat_file()
        
        # Setup API
        headers = setup_api()
        
        # Create bot
        bot_id = create_bot(headers, meeting_url)
        if not bot_id:
            return False
        
        # Store global state
        current_bot_id = bot_id
        current_headers = headers
        bot_running = True
        
        print(f"✅ Recall.ai bot started successfully with ID: {bot_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start Recall.ai bot: {e}")
        reset_bot_state()
        return False

def leave_meeting():
    """Leave the meeting and process transcript (called from FastAPI)."""
    global bot_running, current_bot_id, current_headers
    
    try:
        if not bot_running or not current_bot_id:
            print("⚠️ Bot is not running according to local state")
            reset_bot_state()
            return False
        
        # Leave meeting
        leave_success = leave_meeting_bot(current_headers, current_bot_id)
        
        if leave_success:
            # Wait for transcript to be ready (with better timeout handling)
            bot_data = wait_for_transcript_ready(current_headers, current_bot_id, max_wait=30)  # Reduced timeout
            if bot_data:
                # Process and save transcript (handles empty recordings gracefully)
                result = process_transcript_and_save(bot_data)
                if result:
                    print(f"✅ Meeting ended and transcript processed: {result}")
                else:
                    print("✅ Meeting ended (no transcript content)")
                
                # Reset state after processing
                reset_bot_state()
                return result or "empty_meeting"  # Return something to indicate success
            else:
                print("⚠️ Could not retrieve transcript data")
        
        # Reset state regardless of success/failure
        reset_bot_state()
        return leave_success
        
    except Exception as e:
        print(f"❌ Error leaving meeting: {e}")
        # Reset state even on error
        reset_bot_state()
        return False

def is_bot_running():
    """Check if bot is currently running (with actual status check)."""
    return check_bot_actually_running()

def get_current_bot_id():
    """Get current bot ID."""
    return current_bot_id

def force_reset_bot_state():
    """Force reset the bot state (useful for debugging or manual intervention)."""
    reset_bot_state()
    print("🔄 Bot state forcefully reset")

# Keep the original main function for standalone usage
def main():
    """Main function for standalone usage."""
    try:
        # Setup API
        headers = setup_api()
        
        # Configuration
        meeting_url = "https://meet.google.com/your-meeting-id"  # Replace with your meeting URL
        
        # Clear chat.txt
        clear_chat_file()
        
        # Step 1: Create and start bot
        bot_id = create_bot(headers, meeting_url)
        if not bot_id:
            return
        
        # Step 2: Wait for user input
        print(f"\n⏳ Bot is joining the meeting...")
        print("📱 The bot will start recording when participants join.")
        print("🎤 Remember to turn on captions in Google Meet for transcript capture!")
        input("Press Enter when you want to end the meeting and fetch the transcript...")
        
        # Step 3: Leave the meeting
        if not leave_meeting_bot(headers, bot_id):
            return
        
        # Step 4: Wait for transcript readiness and process
        bot_data = wait_for_transcript_ready(headers, bot_id, max_wait=60)
        if not bot_data:
            print("❌ Failed to get bot data after polling")
            return
        
        # Step 5: Process and save transcript
        result = process_transcript_and_save(bot_data)
        if result:
            print(f"✅ Meeting completed and transcript saved: {result}")
        else:
            print("⚠️ Meeting completed but transcript processing failed")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()