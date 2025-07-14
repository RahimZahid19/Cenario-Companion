import os
import re
from datetime import datetime
import shutil

from load_env import setup_env
setup_env()

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192", temperature=0)

def generate_sow_from_transcript(transcript_text: str) -> str:
    prompt = f"""
    Based on the following meeting transcript, generate a comprehensive Statement of Work (SOW) document.
    
    Transcript:
    {transcript_text}
    
    Please create a detailed SOW that includes:
    1. Project Overview and Objectives
    2. Scope of Work and Deliverables
    3. Timeline and Milestones
    4. Technical Requirements
    5. Budget and Resource Allocation
    6. Risk Assessment and Mitigation
    7. Success Criteria and KPIs
    8. Terms and Conditions
    
    Format the response as a professional SOW document with clear sections and bullet points.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error generating SOW: {e}")
        return f"Error generating SOW: {e}"

def clean_with_llm(text: str) -> str:
    # Clean the text using LLM
    prompt = f"""
    Clean up the following transcript text. Remove any unnecessary repetitions, fix grammatical errors, 
    and improve readability while maintaining the original meaning and context. 
    Format it as a proper meeting transcript with clear speaker attributions.
    
    Text to clean:
    {text}
    
    Return only the cleaned transcript without any additional comments or explanations.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error cleaning transcript: {e}")
        return text  # Return original if cleaning fails

def format_transcript(filename):
    # Read the transcript file
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split into lines and process
    lines = content.strip().split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Try to extract timestamp and speaker
            # Pattern: [timestamp] Speaker: text
            match = re.match(r'\[(.*?)\]\s*(.*?):\s*(.*)', line)
            if match:
                timestamp, speaker, text = match.groups()
                formatted_lines.append(f"[{timestamp}] {speaker}: {text}")
            else:
                # If no pattern match, just add the line
                formatted_lines.append(line)
    
    # Join back and write to file
    formatted_content = '\n'.join(formatted_lines)
    
    # Save formatted version
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(formatted_content)
    
    print(f"✅ Formatted transcript saved to {filename}")
    return formatted_content

def get_latest_transcript():
    # Get the latest transcript file
    transcript_files = [f for f in os.listdir('.') if f.startswith('transcript_') and f.endswith('.txt')]
    if not transcript_files:
        return None
    
    latest_file = max(transcript_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    return latest_file

def load_transcript(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def save_sow_to_file(sow_text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(sow_text)

def sow(latest):
    if latest:
        transcript_text = load_transcript(latest)
        sow_text = generate_sow_from_transcript(transcript_text)
        
        base_name = latest.replace('.txt', '')
        sow_filename = f"{base_name}_SOW.txt"
        save_sow_to_file(sow_text, sow_filename)
        print(f"✅ SOW generated and saved to {sow_filename}")
    else:
        print("❌ No transcript file found")

def generate_document_from_transcript(transcript_text: str, doc_type: str) -> str:
    """Generate different types of documents from transcript"""
    
    if doc_type == "summary":
        prompt = f"""
        Based on the following meeting transcript, generate a comprehensive meeting summary.
        
        Transcript:
        {transcript_text}
        
        Please create a detailed summary that includes:
        1. Meeting Overview
        2. Key Discussion Points
        3. Decisions Made
        4. Action Items
        5. Next Steps
        
        Format the response as a professional meeting summary.
        """
    else:
        return f"Document type '{doc_type}' not supported"
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error generating {doc_type}: {e}")
        return f"Error generating {doc_type}: {e}"

def save_document_to_file(doc_text: str, filename: str):
    """Save document text to file"""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(doc_text)

def generate_documents(latest: str):
    """Generate documents from the latest transcript - DISABLED"""
    print("📄 Document generation is disabled")
    return
    
    # This code is commented out to disable automatic document generation
    # if latest:
    #     transcript_text = load_transcript(latest)
    #     base_name = latest.replace('.txt', '')
    #     
    #     # Generate Meeting Summary
    #     summary_text = generate_document_from_transcript(transcript_text, "summary")
    #     summary_filename = f"{base_name}_Meeting_Summary.txt"
    #     save_document_to_file(summary_text, summary_filename)
    #     print(f"✅ Meeting Summary generated and saved to {summary_filename}")
    # else:
    #     print("❌ No transcript file found for document generation")

def get_next_transcript_number():
    """Get the next available transcript number"""
    transcript_files = [f for f in os.listdir('.') if f.startswith('transcript_') and f.endswith('.txt')]
    
    if not transcript_files:
        return 1
    
    # Extract numbers from filenames
    numbers = []
    for filename in transcript_files:
        match = re.search(r'transcript_(\d+)\.txt', filename)
        if match:
            numbers.append(int(match.group(1)))
    
    if not numbers:
        return 1
    
    return max(numbers) + 1

def convert_chat_to_transcript():
    """Convert chat.txt to a numbered transcript file - RAW COPY ONLY"""
    chat_file = "chat.txt"
    
    if not os.path.exists(chat_file):
        print(f"❌ {chat_file} not found")
        return None
    
    # Check if chat.txt is empty
    if os.path.getsize(chat_file) == 0:
        print(f"❌ {chat_file} is empty")
        return None
    
    # Get next transcript number
    next_number = get_next_transcript_number()
    transcript_filename = f"transcript_{next_number}.txt"
    
    try:
        # Read chat.txt
        with open(chat_file, 'r', encoding='utf-8') as file:
            chat_content = file.read().strip()
        
        if not chat_content:
            print(f"❌ {chat_file} is empty")
            return None
        
        # Save raw content directly to transcript file (NO LLM PROCESSING)
        with open(transcript_filename, 'w', encoding='utf-8') as file:
            file.write(chat_content)
        
        print(f"✅ Raw transcript saved to {transcript_filename}")
        
        # Clear chat.txt for next session
        with open(chat_file, 'w', encoding='utf-8') as file:
            file.write("")
        
        print(f"✅ {chat_file} cleared for next session")
        
        return transcript_filename
        
    except Exception as e:
        print(f"❌ Error converting chat to transcript: {e}")
        return None

def reset_processing_state():
    """Reset the processing state by clearing chat.txt"""
    chat_file = "chat.txt"
    
    try:
        with open(chat_file, 'w', encoding='utf-8') as file:
            file.write("")
        print(f"✅ {chat_file} cleared - ready for new session")
        return True
    except Exception as e:
        print(f"❌ Error clearing {chat_file}: {e}")
        return False

# For testing
if __name__ == "__main__":
    # Test the conversion
    result = convert_chat_to_transcript()
    if result:
        print(f"Conversion successful: {result}")
    else:
        print("Conversion failed")