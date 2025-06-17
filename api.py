import re
import os
from fastapi import FastAPI, Query, HTTPException, Request, UploadFile, File
from fastapi.responses import PlainTextResponse, JSONResponse
from bot import start_meeting_bot, get_live_captions, get_live_analysis, generate_questions_with_groq, extract_text_from_pdf
from cleaner import get_latest_transcript

app = FastAPI()

TRANSCRIPT_MAP = {
    "raw-transcript": "{base}_backup.txt",
    "cleaned-transcript": "{base}.txt",
    "document/sow": "{base}_SOW.txt",
    "document/brd": "{base}_BRD.txt",
    "document/project-scope": "{base}_Project_Scope.txt",
    "document/technical-design": "{base}_Technical_Design.txt",
    "document/meeting-summary": "{base}_Meeting_Summary.txt",
}

# Always use the directory where this script is located
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def get_latest_main_transcript():
    files = [f for f in os.listdir(BASE_DIR) if re.match(r"transcript_\d+\.txt$", f)]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(BASE_DIR, f)), reverse=True)
    return files[0] if files else None

@app.post("/api/join-meeting")
def join_meeting():
    started = start_meeting_bot()
    return {"status": "started" if started else "already running"}

@app.get("/api/live-captions")
def live_captions():
    captions = get_live_captions()
    analysis = get_live_analysis()
    return JSONResponse({"captions": captions, "analysis": analysis})

@app.get("/api/meeting-status")
def meeting_status(meeting_id: str = Query(...)):
    return {"meeting_id": meeting_id, "status": "unknown (implement logic as needed)"}

@app.get("/api/raw-transcript", response_class=PlainTextResponse)
def get_raw_transcript():
    return _get_latest_transcript_file("raw-transcript")

@app.get("/api/cleaned-transcript", response_class=PlainTextResponse)
def get_cleaned_transcript():
    return _get_latest_transcript_file("cleaned-transcript")

@app.get("/api/document/sow", response_class=PlainTextResponse)
def get_sow():
    return _get_latest_transcript_file("document/sow")

@app.get("/api/document/brd", response_class=PlainTextResponse)
def get_brd():
    return _get_latest_transcript_file("document/brd")

@app.get("/api/document/project-scope", response_class=PlainTextResponse)
def get_project_scope():
    return _get_latest_transcript_file("document/project-scope")

@app.get("/api/document/technical-design", response_class=PlainTextResponse)
def get_technical_design():
    return _get_latest_transcript_file("document/technical-design")

@app.get("/api/document/meeting-summary", response_class=PlainTextResponse)
def get_meeting_summary():
    return _get_latest_transcript_file("document/meeting-summary")

@app.post("/api/questions/generated")
async def generate_questions_from_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        contents = await file.read()
        temp_path = os.path.join(BASE_DIR, file.filename)
        with open(temp_path, 'wb') as f:
            f.write(contents)
        text = extract_text_from_pdf(temp_path)
        os.remove(temp_path)
        questions = generate_questions_with_groq(text, num_questions=10)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_latest_transcript_file(doc_type: str):
    latest = get_latest_main_transcript()
    if not latest:
        raise HTTPException(status_code=404, detail="No transcript file found.")
    base = os.path.splitext(latest)[0]
    filename = TRANSCRIPT_MAP[doc_type].format(base=base)
    abs_filename = os.path.join(BASE_DIR, filename)
    if not os.path.exists(abs_filename):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")
    with open(abs_filename, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000) 