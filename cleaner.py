import os
import re
from datetime import datetime
import shutil

from load_env import setup_env
setup_env()

from langchain.chat_models import init_chat_model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

def generate_sow_from_transcript(transcript_text: str) -> str:
    prompt = (
        "You are a professional business analyst.\n\n"
        "Based on the following meeting transcript, generate a formal Statement of Work (SOW) document with these sections:\n"
        "1. Project Overview\n"
        "2. Objectives\n"
        "3. Scope of Work\n"
        "4. Deliverables\n"
        "5. Timeline (estimate if not mentioned)\n"
        "6. Assumptions\n"
        "7. Risks and Mitigation\n"
        "8. Acceptance Criteria\n"
        "9. Stakeholders\n\n"
        "Meeting Transcript:\n"
        f"\"\"\"\n{transcript_text.strip()}\n\"\"\"\n\n"
        "Only return the completed SOW. Do not include extra commentary."
    )

    result = llm.invoke(prompt)
    content = result.content.strip() if hasattr(result, "content") else str(result).strip()

    # Strip any common unnecessary preambles LLMs sometimes include
    clean_intro_patterns = [
        "here is the statement of work",
        "statement of work",
        "sow document",
        "as requested",
    ]

    for phrase in clean_intro_patterns:
        if content.lower().startswith(phrase):
            content = content.split("\n", 1)[-1].strip()
            break

    return content


def clean_with_llm(text: str) -> str:
    prompt = (
        "You will be given a transcript formatted as [mm:ss] Speaker: sentence.\n"
        "Your task is to:\n"
        "- Remove duplicated words or phrases\n"
        "- Complete broken or incomplete sentences\n"
        "- Fix grammar and fluency\n"
        "- Keep the timestamp and speaker format exactly as it is\n\n"
        "Return ONLY the cleaned transcript — do NOT include any explanation or extra commentary.\n\n"
        f"{text.strip()}"
    )

    result = llm.invoke(prompt)
    content = result.content.strip() if hasattr(result, "content") else str(result).strip()

    # Strip any known intro phrases LLM might include
    clean_intro_patterns = [
        "here is the cleaned transcript",
        "cleaned version",
        "here you go",
        "as requested",
    ]

    for phrase in clean_intro_patterns:
        if content.lower().startswith(phrase):
            content = content.split("\n", 1)[-1].strip()
            break

    return content


def format_transcript(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    formatted = []
    last_speaker = None
    last_time = None
    buffer = []

    for line in lines:
        match = re.match(r"\[(\d{2}:\d{2}:\d{2})\] \[(.*?)\] (.+)", line.strip())
        if not match:
            continue

        timestamp_str, speaker, text = match.groups()
        timestamp_obj = datetime.strptime(timestamp_str, "%H:%M:%S")
        mmss = f"[{timestamp_obj.minute:02d}:{timestamp_obj.second:02d}]"

        if speaker != last_speaker:
            if last_speaker and buffer:
                formatted.append(f"{last_time} {last_speaker}: {' '.join(buffer)}")
                buffer = []

            last_speaker = speaker
            last_time = mmss

        buffer.append(text)

    if last_speaker and buffer:
        formatted.append(f"{last_time} {last_speaker}: {' '.join(buffer)}")

    raw_text = "\n\n".join(formatted)

    # Optional: create backup before overwriting
    backup_path = filename.replace(".txt", "_backup.txt")
    shutil.copy(filename, backup_path)

    cleaned_text = clean_with_llm(raw_text)

    # Overwrite original file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"Cleaned and overwritten: {filename}")
    print(f"Backup saved as: {backup_path}")


def get_latest_transcript():
    files = [f for f in os.listdir('.') if f.startswith("transcript_") and f.endswith(".txt")]
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0] if files else None

def load_transcript(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()

def save_sow_to_file(sow_text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(sow_text)

def sow(latest):
    transcript_text = load_transcript(latest)
    print("\n Generating Statement of Work (SOW)...")

    # Generate SOW
    sow_text = generate_sow_from_transcript(transcript_text)

    # Create output filename (using the transcript filename without extension)
    file_base = os.path.splitext(latest)[0]  # Remove .txt extension
    sow_file = f"{file_base}_SOW.txt"
        
    # Save SOW
    save_sow_to_file(sow_text, sow_file)
    print(f" SOW saved to '{sow_file}'")
    

def generate_document_from_transcript(transcript_text: str, doc_type: str) -> str:
    prompts = {
        "SOW": (
            "You are a professional business analyst.\n\n"
            "Based on the following meeting transcript, generate a formal Statement of Work (SOW) document with these sections:\n"
            "1. Project Overview\n"
            "2. Objectives\n"
            "3. Scope of Work\n"
            "4. Deliverables\n"
            "5. Timeline (estimate if not mentioned)\n"
            "6. Assumptions\n"
            "7. Risks and Mitigation\n"
            "8. Acceptance Criteria\n"
            "9. Stakeholders\n\n"
            "Meeting Transcript:\n"
            f"\"\"\"\n{transcript_text.strip()}\n\"\"\"\n\n"
            "Generate a comprehensive SOW of approximately 4000 words. Include specific details, examples, and clear explanations for each section. "
            "Only return the completed SOW. Do not include extra commentary."
        ),
        "BRD": (
            "You are a senior business analyst.\n\n"
            "Based on the following meeting transcript, generate a detailed Business Requirements Document (BRD) with these sections:\n"
            "1. Executive Summary\n"
            "2. Project Background\n"
            "3. Business Objectives\n"
            "4. Stakeholder Analysis\n"
            "5. Functional Requirements\n"
            "6. Non-Functional Requirements\n"
            "7. Business Rules\n"
            "8. Process Flows\n"
            "9. Data Requirements\n"
            "10. User Acceptance Criteria\n"
            "11. Constraints and Assumptions\n"
            "12. Risks and Mitigation Strategies\n\n"
            "Meeting Transcript:\n"
            f"\"\"\"\n{transcript_text.strip()}\n\"\"\"\n\n"
            "Generate a comprehensive BRD of approximately 4000 words. Include detailed requirements, process flows, and clear acceptance criteria. "
            "Only return the completed BRD. Do not include extra commentary."
        ),
        "Project_Scope": (
            "You are a project management expert.\n\n"
            "Based on the following meeting transcript, generate a detailed Project Scope Document with these sections:\n"
            "1. Project Overview\n"
            "2. Business Case\n"
            "3. Project Objectives\n"
            "4. Scope Description\n"
            "5. Deliverables\n"
            "6. Milestones\n"
            "7. Technical Requirements\n"
            "8. Constraints\n"
            "9. Assumptions\n"
            "10. Dependencies\n"
            "11. Risks and Mitigation\n"
            "12. Success Criteria\n\n"
            "Meeting Transcript:\n"
            f"\"\"\"\n{transcript_text.strip()}\n\"\"\"\n\n"
            "Generate a comprehensive Project Scope document of approximately 4000 words. Include specific details about project boundaries, deliverables, and success criteria. "
            "Only return the completed document. Do not include extra commentary."
        ),
        "Meeting_Summary": (
            "You are a professional meeting facilitator.\n\n"
            "Based on the following meeting transcript, generate a comprehensive meeting summary with these sections:\n"
            "1. Meeting Overview\n"
            "2. Key Discussion Points\n"
            "3. Decisions Made\n"
            "4. Action Items (with owners and deadlines)\n"
            "5. Risks and Issues Identified\n"
            "6. Next Steps\n"
            "7. Follow-up Questions\n"
            "8. Timeline Updates\n"
            "9. Resource Requirements\n"
            "10. Stakeholder Updates\n\n"
            "Meeting Transcript:\n"
            f"\"\"\"\n{transcript_text.strip()}\n\"\"\"\n\n"
            "Generate a detailed meeting summary of approximately 4000 words. Include specific details about decisions, action items, and next steps. "
            "Only return the completed summary. Do not include extra commentary."
        ),
        "Technical_Design": (
            "You are a senior technical architect.\n\n"
            "Based on the following meeting transcript, generate a comprehensive Technical Design Document with these sections:\n"
            "1. System Overview\n"
            "2. Architecture Design\n"
            "3. Technical Requirements\n"
            "4. System Components\n"
            "5. Data Models\n"
            "6. API Specifications\n"
            "7. Security Requirements\n"
            "8. Performance Requirements\n"
            "9. Integration Points\n"
            "10. Deployment Strategy\n"
            "11. Testing Strategy\n"
            "12. Maintenance Plan\n\n"
            "Meeting Transcript:\n"
            f"\"\"\"\n{transcript_text.strip()}\n\"\"\"\n\n"
            "Generate a detailed Technical Design Document of approximately 4000 words. Include specific technical details, diagrams, and implementation guidelines. "
            "Only return the completed document. Do not include extra commentary."
        )
    }

    if doc_type not in prompts:
        raise ValueError(f"Unsupported document type: {doc_type}")

    result = llm.invoke(prompts[doc_type])
    content = result.content.strip() if hasattr(result, "content") else str(result).strip()

    # Strip any common unnecessary preambles LLMs sometimes include
    clean_intro_patterns = [
        "here is the",
        "as requested",
        "based on the transcript",
        "following is the",
    ]

    for phrase in clean_intro_patterns:
        if content.lower().startswith(phrase):
            content = content.split("\n", 1)[-1].strip()
            break

    return content

def save_document_to_file(doc_text: str, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(doc_text)

def generate_documents(latest: str):
    transcript_text = load_transcript(latest)
    print("\nGenerating documents from transcript...")

    # Create output filename base (using the transcript filename without extension)
    file_base = os.path.splitext(latest)[0]  # Remove .txt extension
    
    # Generate all document types
    doc_types = ["SOW", "BRD", "Project_Scope", "Meeting_Summary", "Technical_Design"]
    
    for doc_type in doc_types:
        print(f"\nGenerating {doc_type}...")
        try:
            doc_text = generate_document_from_transcript(transcript_text, doc_type)
            doc_file = f"{file_base}_{doc_type}.txt"
            save_document_to_file(doc_text, doc_file)
            print(f"{doc_type} saved to '{doc_file}'")
        except Exception as e:
            print(f"Error generating {doc_type}: {e}")

if __name__ == "__main__":
    latest = get_latest_transcript()
    if latest:
        # Process the transcript
        format_transcript(latest)
        
        # Generate all documents
        generate_documents(latest)
    else:
        print("No transcript file found.")


