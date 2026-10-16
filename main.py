from fastapi import FastAPI, UploadFile, File
import whisper
import shutil # provides high level operations for managing files, directory trees and archives
import os

app = FastAPI(title="Offline Whisper Transcription Service")

# Standard practice to keep a root health-check endpoint 
@app.get("/")
async def root():
    return {"message": "OpenMRS Offline Voice Transcription Service is running!"}

# Load the lightweight model into memory on startup
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded successfully!")

# using @ as a decorator, purpose : attach extra behaviour to the function transcribe_audio 
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the incoming audio file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run Whisper locally offline
        result = model.transcribe(temp_file_path)
        transcript_text = result["text"]
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return {
        "filename": file.filename,
        "transcript": transcript_text.strip()
    }