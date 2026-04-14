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


@app.get("/shared/{filename}")
async def shared_file(filename: str):
    from fastapi.responses import Response
    from fastapi import HTTPException
    safe = Path(filename).name
    # Docker: shared/ is inside app dir; local dev: shared/ is sibling at apps/shared/
    file_path = Path(__file__).parent / "shared" / safe
    if not file_path.is_file():
        file_path = Path(__file__).parent.parent / "shared" / safe
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return Response(
        content=file_path.read_text(encoding="utf-8"),
        media_type="application/javascript",
    )

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
            "max_tokens": 8192,
        },
        timeout=30,
    )
    data = response.json()
    if "choices" not in data:
        raise ValueError(f"OpenRouter error: {data}")
    return {"cleaned": data["choices"][0]["message"]["content"]}
