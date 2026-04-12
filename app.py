import os
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "anthropic/claude-haiku-3-5"

class TranscriptRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())

@app.post("/clean")
async def clean_transcript(req: TranscriptRequest):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": (
                    "Clean up this voice transcript. Fix grammar and punctuation, "
                    "remove filler words (um, uh, like, you know, sort of), fix run-on sentences. "
                    "Keep the meaning and tone exactly as intended. "
                    "Return only the cleaned text, nothing else.\n\n"
                    + req.text
                )
            }],
            "max_tokens": 1024,
        },
        timeout=30,
    )
    cleaned = response.json()["choices"][0]["message"]["content"]
    return {"cleaned": cleaned}
