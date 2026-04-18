import json
import os
import re
import sys
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-v3.2"

MIN_WORD_CHARS = 3

_SUSPICIOUS_STARTS = (
    "i ", "i'", "i\u2019", "sure", "certainly", "okay", "of course",
    "please", "here is", "here's", "understood",
)
_SUSPICIOUS_SUBSTRINGS = (
    "provide the transcript", "provide the text", "provide the voice",
    "i will process", "i understand", "i'll clean", "i will clean",
    "as an ai", "happy to help",
)


def _looks_like_meta_response(raw_input: str, model_output: str) -> bool:
    if not model_output:
        return False
    stripped = model_output.strip().lower()
    if not stripped:
        return False
    if stripped.startswith(_SUSPICIOUS_STARTS):
        return True
    if any(s in stripped for s in _SUSPICIOUS_SUBSTRINGS):
        return True
    # Output wildly longer than input with no shared vocabulary is suspicious
    in_words = {w for w in re.findall(r"[a-z]+", raw_input.lower()) if len(w) > 2}
    out_words = re.findall(r"[a-z]+", stripped)
    if in_words and len(out_words) > len(in_words) * 3:
        overlap = sum(1 for w in out_words if w in in_words)
        if overlap < max(2, len(in_words) // 3):
            return True
    return False


def _load_croquet_dictionary() -> dict:
    """Load croquet-dictionary.json from shared/ — tries app dir first, then parent."""
    for base in (Path(__file__).parent, Path(__file__).parent.parent):
        p = base / "shared" / "croquet-dictionary.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return {"terms": [], "players": []}


_DICTIONARY = _load_croquet_dictionary()
_DICTIONARY_HINT = (
    "The following are valid croquet terms that may appear in the transcript — "
    "correct any misheard words to match these exactly: "
    + ", ".join(_DICTIONARY.get("terms", []))
    + ". "
    "The following are Queensland croquet player names — "
    "correct any misheard names to match these exactly: "
    + ", ".join(_DICTIONARY.get("players", []))
    + ". "
) if (_DICTIONARY.get("terms") or _DICTIONARY.get("players")) else ""

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

_SYSTEM_PROMPT = (
    "You are a transcript cleaner. You receive raw voice-recognition text "
    "and return ONLY the cleaned text. Rules:\n"
    "- Add punctuation (commas, full stops, question marks).\n"
    "- Capitalise the start of sentences.\n"
    "- Remove filler words (um, uh, like, you know, sort of).\n"
    "- Fix run-on sentences by breaking them up.\n"
    "- Fix speech-recognition errors using whole-sentence context.\n"
    "- Keep meaning and tone exactly as intended.\n"
    "You NEVER respond conversationally. You NEVER ask for input. "
    "You NEVER explain what you are doing. If the input is empty, whitespace, "
    "a single word, or otherwise not a usable transcript, return it verbatim. "
    "Output is the cleaned transcript text and nothing else.\n"
    + _DICTIONARY_HINT
)


@app.post("/clean")
async def clean_transcript(req: TranscriptRequest):
    raw = req.text or ""
    word_chars = sum(1 for c in raw if c.isalpha())
    if word_chars < MIN_WORD_CHARS:
        return {"cleaned": raw}

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw},
            ],
            "max_tokens": 8192,
        },
        timeout=30,
    )
    data = response.json()
    if "choices" not in data:
        raise ValueError(f"OpenRouter error: {data}")
    cleaned = data["choices"][0]["message"]["content"]

    if _looks_like_meta_response(raw, cleaned):
        print(f"[suspicious-output] falling back to raw. in={raw!r} out={cleaned!r}", file=sys.stderr)
        return {"cleaned": raw}

    return {"cleaned": cleaned}
