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
import platform
import asyncio

# Windows-specific asyncio fix for sync_playwright
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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