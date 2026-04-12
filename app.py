import os
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

class TranscriptRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())

@app.post("/clean")
async def clean_transcript(req: TranscriptRequest):
    prompt = (
        "Clean up this voice transcript. Fix grammar and punctuation, "
        "remove filler words (um, uh, like, you know, sort of), fix run-on sentences. "
        "Keep the meaning and tone exactly as intended. "
        "Return only the cleaned text, nothing else.\n\n"
        + req.text
    )
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=30,
    )
    data = response.json()
    cleaned = data["candidates"][0]["content"]["parts"][0]["text"]
    return {"cleaned": cleaned}
