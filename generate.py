# main.py (modified for folder input)
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load the Groq API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(" Groq API key not found. Make sure it's set in the .env file.")

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Folder containing your documents
DOCUMENT_FOLDER = r"C:\Users\abdul.basit\Documents\Python Test codes\upload_document"


def get_predefined_questions():
    return [
  "Who is involved in the project and why are we doing it?",
  "What is the project about?",
  "What are we trying to achieve?",

]

import requests

def generate_questions_with_groq(text, num_questions=10):
    prompt = f"Based on the following Statement of Work (SOW), generate {num_questions} professional and concise questions that would help gather the necessary information to recreate a similar SOW for a new project."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": 'llama3-8b-8192',# "llama3-70b-8192",  
        "messages": [
            {"role": "system", "content": """
            You are a professional business analyst assistant.

            Your task is to read a Statement of Work (SOW) document and generate a list of clear, information-gathering questions that would help a user provide the necessary details to recreate a similar SOW for a different project.

            - Only output the questions.
            - Do not include any explanation, headings, or extra commentary.
            - Make the questions plain, professional, and comprehensive.
            """}
            ,
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        lines = content.strip().split("\n")

        # Remove header or non-question lines
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


def load_transcript(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()



def generate_sow_from_transcript(transcript_text):
    prompt = f"""
You are a professional business analyst.

Based on the following meeting transcript, generate a formal Statement of Work (SOW) document with the following sections:

1. Project Overview  
2. Objectives  
3. Scope of Work  
4. Deliverables  
5. Timeline (estimate if not mentioned)  
6. Assumptions  
7. Risks and Mitigation  
8. Acceptance Criteria  
9. Stakeholders  

Meeting Transcript:
\"\"\"
{transcript_text}
\"\"\"

Only return the completed SOW. Do not include extra commentary.
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert at drafting professional documents like SOWs."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        return content
    else:
        raise Exception(f"Groq API Error {response.status_code}: {response.text}")


def save_sow_to_file(sow_text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(sow_text)

# def generate_answers_from_transcript(questions):
#     # Step 1: Get live transcript
#     transcript_chunks = get_live_captions(max_items=50)
#     transcript_text = " ".join(transcript_chunks).strip()
#     # Step 2: Ask Groq to answer each question
#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     results = []
#     for question in questions:
#         prompt = f"""You are an AI business assistant. Answer the following question based only on the provided transcript.
# Transcript:
# \"\"\"
# {transcript_text}
# \"\"\"
# Question: {question}
# If the answer is not clearly present, say "Not enough information." Keep the response concise and professional."""
#         data = {
#             "model": 'llama3-8b-8192',
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.3
#         }
#         response = requests.post(url, headers=headers, json=data)
#         if response.status_code == 200:
#             answer = response.json()['choices'][0]['message']['content'].strip()
#             results.append({"question": question, "answer": answer})
#         else:
#             results.append({"question": question, "answer": f"[Error {response.status_code}]"})
#     return results



# Main execution
if __name__ == "__main__":
    if not os.path.exists(DOCUMENT_FOLDER):
        raise FileNotFoundError(f" Folder '{DOCUMENT_FOLDER}' not found. Please create it and add PDFs.")

    pdf_files = [f for f in os.listdir(DOCUMENT_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(" No PDF files found in the 'documents' folder.")

    for file_name in pdf_files:
        file_path = os.path.join(DOCUMENT_FOLDER, file_name)
        print(f"\n Reading: {file_name}")
        text = extract_text_from_pdf(file_path)

        print("\n Generating Questions...")
        questions = generate_questions_with_groq(text)
        print("\n Questions:")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}")

        print("\n Generating Statement of Work (SOW)...")
        transcript_path = r"C:\Users\sameer.uddin\Desktop\QA\transcript.txt"
        transcript_text = load_transcript(transcript_path)

        sow_text = generate_sow_from_transcript(transcript_text)

        sow_file = f"{file_name.replace('.pdf', '')}_SOW.txt"
        save_sow_to_file(sow_text, sow_file)
        print(f" SOW saved to '{sow_file}'")





