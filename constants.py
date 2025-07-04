import os
import re
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

TRANSCRIPT_MAP = {
    "raw-transcript": "{base}_backup.txt",
    "cleaned-transcript": "{base}.txt",
    "document/sow": "{base}_SOW.txt",
    "document/brd": "{base}_BRD.txt",
    "document/project-scope": "{base}_Project_Scope.txt",
    "document/technical-design": "{base}_Technical_Design.txt",
    "document/meeting-summary": "{base}_Meeting_Summary.txt",
}

def get_latest_main_transcript():
    try:
        files = [f for f in os.listdir(BASE_DIR) if re.match(r"transcript_\d+\.txt$", f)]
        if not files:
            print("No transcript files found.")
            return None
        files_with_mtime = [
            (f, os.path.getmtime(os.path.join(BASE_DIR, f))) for f in files
        ]
        files_with_mtime.sort(key=lambda x: x[1], reverse=True)
        latest_file = files_with_mtime[0][0]
        print(f"Latest transcript file detected: {latest_file}")
        return latest_file
    except Exception as e:
        print(f"Error in get_latest_main_transcript: {e}")
        return None
