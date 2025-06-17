from playwright.sync_api import sync_playwright
import time
from datetime import datetime
import os
import re
from difflib import SequenceMatcher
import requests
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import threading
from queue import Queue
load_dotenv()

# CONFIG
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set your key via env variable
DOCUMENT_FOLDER = "upload_document"

# ==== PDF & QUESTION GENERATION ==== #

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def generate_questions_with_groq(text, num_questions=10):
    prompt = f"Based on the following Statement of Work (SOW), generate {num_questions} professional and concise questions that would help gather the necessary information to recreate a similar SOW for a new project."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": 'llama3-8b-8192',
        "messages": [
            {"role": "system", "content": """
            You are a professional business analyst assistant.

            Your task is to read a Statement of Work (SOW) document and generate a list of clear, information-gathering questions that would help a user provide the necessary details to recreate a similar SOW for a different project.

            - Only output the questions.
            - Do not include any explanation, headings, or extra commentary.
            - Make the questions plain, professional, and comprehensive.
            """},
            {"role": "user", "content": prompt + "\n\n" + text}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        lines = content.strip().split("\n")
        questions = []
        for line in lines:
            clean = line.strip("-•1234567890. ) ").strip()
            if clean.lower().startswith("here are"):
                continue
            if clean.endswith("?"):
                questions.append(clean)
        return questions
    else:
        raise Exception(f"Groq API Error {response.status_code}: {response.text}")

# ==== CAPTION CLEANING SETUP ==== #

def get_next_transcript_filename(base="transcript", ext=".txt"):
    existing = [
        int(re.search(rf"{base}_(\d+){ext}", f).group(1))
        for f in os.listdir()
        if re.match(rf"{base}_(\d+){ext}", f)
    ]
    next_index = max(existing) + 1 if existing else 1
    return f"{base}_{next_index}{ext}"

transcript_filename = get_next_transcript_filename()

def log_caption(text):
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {text}"
        print(log_line)
        with open(transcript_filename, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"Error logging caption: {e}")

# Buffers
last_captions = {}
sentence_buffer = {}
last_speech_time = {}
recent_statements = {}
seen_captions_time = {}
last_logged_sentences = {}
conversation_buffer = []
last_analysis_time = time.time()
ANALYSIS_INTERVAL = 30  # Analyze conversation every 30 seconds

# Shared queues for API access
live_captions_queue = Queue()
live_analysis_queue = Queue()

# Store thread reference
default_bot_thread = None

def normalize(text):
    return re.sub(r"[^\w\s]", "", text).lower().strip()

def remove_duplicate_prefix(new, old):
    if not old:
        return new.strip()
    norm_new = normalize(new)
    norm_old = normalize(old)
    i = 0
    while i < min(len(norm_new), len(norm_old)) and norm_new[i] == norm_old[i]:
        i += 1
    original_index = i
    while original_index < len(new) and new[original_index] in " ,.-":
        original_index += 1
    return new[original_index:].strip()

def is_tail_repeat(new, old):
    if not new or not old:
        return False
    return normalize(new) in normalize(old[-50:])

def is_similar(a, b, threshold=0.92):
    return SequenceMatcher(None, a, b).ratio() > threshold

def try_click(page, selector, timeout=3000):
    try:
        page.locator(selector).click(timeout=timeout)
        return True
    except:
        return False

def process_caption(speaker: str, caption: str, current_time: float) -> tuple:
    """Process a single caption and return (should_log, final_sentence)"""
    try:
        if not speaker or not caption or len(caption) < 3:
            return False, ""

        key = (speaker, caption)
        if key not in seen_captions_time:
            seen_captions_time[key] = current_time

        time_elapsed = current_time - seen_captions_time[key]
        last_logged = last_captions.get(speaker, "")

        if time_elapsed >= 0.3:
            cleaned = remove_duplicate_prefix(caption, last_logged)
            norm_cleaned = normalize(cleaned)
            norm_full = normalize(caption)

            if not cleaned or is_tail_repeat(cleaned, last_logged):
                return False, ""

            if speaker not in recent_statements:
                recent_statements[speaker] = []

            if any(is_similar(norm_full, normalize(prev)) for prev in recent_statements[speaker][-5:]):
                return False, ""

            last_speech_time[speaker] = current_time
            last_captions[speaker] = caption
            recent_statements[speaker].append(caption)
            if len(recent_statements[speaker]) > 10:
                recent_statements[speaker] = recent_statements[speaker][-10:]

            buffer = sentence_buffer.get(speaker, "")
            buffer += " " + cleaned
            final_sentence = buffer.strip()
            sentence_buffer[speaker] = final_sentence

            should_log = False
            if final_sentence.endswith((".", "!", "?")) or len(final_sentence.split()) >= 10:
                last_logged_full = last_logged_sentences.get(speaker, "")
                if not is_similar(normalize(final_sentence), normalize(last_logged_full)):
                    should_log = True
                    last_logged_sentences[speaker] = final_sentence

            return should_log, final_sentence

        return False, ""
    except Exception as e:
        print(f"Error processing caption: {e}")
        return False, ""

def analyze_conversation(text: str) -> str:
    try:
        prompt = (
            "Analyze the following conversation segment and extract:\n"
            "1. Key decisions made\n"
            "2. Action items (with owners and deadlines if mentioned)\n"
            "3. Suggested follow-up questions\n"
            "4. Potential risks or issues\n\n"
            "Conversation:\n"
            f"\"\"\"\n{text}\n\"\"\"\n\n"
            "Return the analysis in a structured format."
        )

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": 'llama3-8b-8192',
            "messages": [
                {"role": "system", "content": "You are a professional meeting assistant that analyzes conversations in real-time."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Groq API Error {response.status_code}: {response.text}")
    except Exception as e:
        return f"Error during conversation analysis: {str(e)}"

def process_transcript_files(transcript_filename=None):
    try:
        from cleaner import format_transcript, get_latest_transcript, generate_documents
        latest = get_latest_transcript()
        if transcript_filename is not None and latest and latest == transcript_filename:
            format_transcript(latest)
            generate_documents(latest)
            return True
        else:
            print("Transcript not found or mismatch in filename.")
            return False
    except Exception as e:
        print(f"Error during transcript cleaning: {e}")
        return False

def start_meeting_bot(meeting_url="https://meet.google.com/fch-wctx-ujo"):
    def bot_main():
        global transcript_filename, last_analysis_time, conversation_buffer
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=False)
                context = browser.new_context(
                    storage_state="auth.json",
                    permissions=[],
                    device_scale_factor=1,
                    is_mobile=False,
                    viewport={"width": 1280, "height": 720}
                )

                context.add_init_script("""
                    navigator.mediaDevices.getUserMedia = async () => {
                        throw new Error("Requested device not found");
                    };
                """)

                page = context.new_page()
                page.goto(meeting_url)
                page.wait_for_load_state("domcontentloaded")
                time.sleep(5)

                try_click(page, "text=Continue without microphone and camera")
                if not try_click(page, "text=Switch here"):
                    if not try_click(page, "text=Join now"):
                        try_click(page, "text=Ask to join")

                try_click(page, "button[aria-label='Turn on captions']")

                try:
                    while True:
                        leave_button = page.locator("div.VYBDae-Bz112c-RLmnJb")
                        if leave_button.count() == 0:
                            break

                        current_time = time.time()
                        try:
                            speakers = page.locator("span.NWpY1d").all_inner_texts()
                            captions = page.locator("div.ygicle.VbkSUe").all_inner_texts()

                            for i in range(min(len(speakers), len(captions))):
                                speaker = speakers[i].strip()
                                caption = captions[i].strip()
                                should_log, final_sentence = process_caption(speaker, caption, current_time)
                                if should_log:
                                    log_caption(f"[{speaker}] {final_sentence}")
                                    conversation_buffer.append(f"[{speaker}] {final_sentence}")
                                    live_captions_queue.put({"speaker": speaker, "caption": final_sentence})

                            # Analyze conversation periodically
                            if current_time - last_analysis_time >= ANALYSIS_INTERVAL:
                                if conversation_buffer:
                                    analysis = analyze_conversation("\n".join(conversation_buffer[-10:]))
                                    live_analysis_queue.put(analysis)
                                    conversation_buffer = []
                                    last_analysis_time = current_time
                        except Exception as e:
                            pass
                        time.sleep(0.3)
                except KeyboardInterrupt:
                    pass
                finally:
                    browser.close()
                    process_transcript_files(transcript_filename)
            except Exception as e:
                if 'browser' in locals():
                    browser.close()
    global default_bot_thread
    if default_bot_thread is None or not default_bot_thread.is_alive():
        default_bot_thread = threading.Thread(target=bot_main, daemon=True)
        default_bot_thread.start()
        return True
    return False

def get_live_captions(max_items=20):
    items = []
    while not live_captions_queue.empty() and len(items) < max_items:
        items.append(live_captions_queue.get())
    return items

def get_live_analysis(max_items=5):
    items = []
    while not live_analysis_queue.empty() and len(items) < max_items:
        items.append(live_analysis_queue.get())
    return items

# ==== MAIN SCRIPT ==== #

if __name__ == "__main__":
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                storage_state="auth.json",
                permissions=[],
                device_scale_factor=1,
                is_mobile=False,
                viewport={"width": 1280, "height": 720}
            )

            context.add_init_script("""
                navigator.mediaDevices.getUserMedia = async () => {
                    throw new Error("Requested device not found");
                };
            """)

            page = context.new_page()
            page.goto("https://meet.google.com/fch-wctx-ujo")
            page.wait_for_load_state("domcontentloaded")
            time.sleep(5)

            print("Simulating no mic and no camera found (devices missing).")
            print(f"Writing transcript to {transcript_filename}")

            try_click(page, "text=Continue without microphone and camera")
            if not try_click(page, "text=Switch here"):
                if not try_click(page, "text=Join now"):
                    try_click(page, "text=Ask to join")

            try_click(page, "button[aria-label='Turn on captions']")

            # === Load PDF and Print Questions === #
            try:
                if not os.path.exists(DOCUMENT_FOLDER):
                    raise FileNotFoundError(f"Folder '{DOCUMENT_FOLDER}' not found. Please create it and add PDFs.")

                pdf_files = [f for f in os.listdir(DOCUMENT_FOLDER) if f.endswith(".pdf")]
                if not pdf_files:
                    raise FileNotFoundError("No PDF files found in the 'upload_document' folder.")

                file_path = os.path.join(DOCUMENT_FOLDER, pdf_files[0])
                print(f"\nReading PDF: {pdf_files[0]}")
                text = extract_text_from_pdf(file_path)

                print("\nGenerating Questions using Groq...")
                questions = generate_questions_with_groq(text)

                print("\nGenerated Questions:")
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q}")
            except Exception as e:
                print(f"Failed to generate questions: {e}")

            print("\nCapturing clean captions with speaker repeat deduplication...\n")

            try:
                while True:
                    leave_button = page.locator("div.VYBDae-Bz112c-RLmnJb")
                    if leave_button.count() == 0:
                        print("'Leave call' button disappeared — exiting.")
                        break

                    current_time = time.time()

                    try:
                        speakers = page.locator("span.NWpY1d").all_inner_texts()
                        captions = page.locator("div.ygicle.VbkSUe").all_inner_texts()

                        for i in range(min(len(speakers), len(captions))):
                            speaker = speakers[i].strip()
                            caption = captions[i].strip()

                            should_log, final_sentence = process_caption(speaker, caption, current_time)

                            if should_log:
                                log_caption(f"[{speaker}] {final_sentence}")
                                conversation_buffer.append(f"[{speaker}] {final_sentence}")

                        # Analyze conversation periodically
                        if current_time - last_analysis_time >= ANALYSIS_INTERVAL:
                            if conversation_buffer:
                                analysis = analyze_conversation("\n".join(conversation_buffer[-10:]))
                                print("\n=== Real-time Analysis ===")
                                print(analysis)
                                print("========================\n")
                                
                                conversation_buffer = []
                                last_analysis_time = current_time

                    except Exception as e:
                        print(f"Error processing captions: {e}")

                    time.sleep(0.3)

            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Exiting...")

            finally:
                browser.close()
                print("Browser closed. Cleaning transcript...")

                try:
                    from cleaner import format_transcript, get_latest_transcript, generate_documents

                    latest = get_latest_transcript()
                    if latest and latest == transcript_filename:
                        format_transcript(latest)
                        generate_documents(latest)
                    else:
                        print("Transcript not found or mismatch in filename.")
                except Exception as e:
                    print(f"Error during transcript cleaning: {e}")

                print("Goodbye!")

        except Exception as e:
            print(f"Fatal error: {e}")
            if 'browser' in locals():
                browser.close()
