import os
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-v3.2"

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
                    "Clean up this voice transcript into readable, properly punctuated text. "
                    "The input has no punctuation — you must add it. "
                    "Capitalise the start of sentences. Add commas, full stops, and question marks where needed. "
                    "Remove filler words (um, uh, like, you know, sort of). "
                    "Fix run-on sentences by breaking them up. "
                    "Fix speech recognition errors by reading the full sentence and paragraph to understand intended meaning — "
                    "use whole-context inference, not just adjacent words. "
                    "Keep the meaning and tone exactly as intended. "
                    "Return only the cleaned text, nothing else.\n\n"
                    + req.text
                )
            }],
            "max_tokens": 1024,
        },
        timeout=30,
    )
    data = response.json()
    if "choices" not in data:
        raise ValueError(f"OpenRouter error: {data}")
    return {"cleaned": data["choices"][0]["message"]["content"]}
