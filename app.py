import json
import logging
import os
import re
import sys
import time
import unicodedata
import uuid as uuid_lib
from contextlib import asynccontextmanager
from contextvars import ContextVar

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase 5: Structured JSON logging
#
# Every log line is emitted as a single JSON object to stdout.
# Coolify's log viewer handles line-based output, and JSON is grep/jq-able.
#
# REQUEST_ID ContextVar carries the per-request correlation ID.  Every log
# call in the request handler picks it up automatically via the formatter.
# ---------------------------------------------------------------------------
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")


class _JsonFormatter(logging.Formatter):
    """Serialise every log record as a single JSON line."""

    _SKIP_KEYS = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        super().format(record)
        out: dict = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(""),
        }
        for key, value in record.__dict__.items():
            if key not in self._SKIP_KEYS:
                out[key] = value
        if record.exc_text:
            out["exc"] = record.exc_text
        return json.dumps(out, default=str)


def _configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).propagate = False


_configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default timeouts (Phase 5 — httpx per-phase timeouts)
# connect=5s  : TCP handshake must complete quickly
# read=30s    : normal API responses (OpenRouter JSON endpoints)
# write=30s   : request body upload
# pool=5s     : waiting for a free connection from the pool
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)


@asynccontextmanager
async def lifespan(application: FastAPI):
    client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
    application.state.http = client
    logger.info("Application startup", extra={"event": "app_startup"})
    try:
        yield
    finally:
        await client.aclose()
        logger.info("Application shutdown", extra={"event": "app_shutdown"})


app = FastAPI(lifespan=lifespan)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-v4-lite"

MIN_WORD_CHARS = 3
MAX_CHUNK_CHARS = 2_000

_META_START_RE = re.compile(
    r"^\s*(sure|certainly|okay|of course|please|here is|here's|understood|"
    r"i (can|will|am happy|'ll clean|'ll process)|i['\u2019]m happy)\b",
    re.IGNORECASE,
)
_SUSPICIOUS_SUBSTRINGS = (
    "provide the transcript", "provide the text", "provide the voice",
    "i will process", "i understand", "i'll clean", "i will clean",
    "as an ai", "happy to help",
)

# ---------------------------------------------------------------------------
# Phase 5: Prompt-injection two-layer defence + input normalization
# ---------------------------------------------------------------------------
_ZERO_WIDTH_TO_SPACE = str.maketrans({
    "\u200b": " ", "\u200c": " ", "\u200d": " ",
    "\u200e": " ", "\u200f": " ", "\u2060": " ",
    "\ufeff": " ",
})
_PROMPT_INJECTION_KEYWORDS = (
    "ignore previous", "ignore all previous", "disregard previous", "disregard instructions",
    "system prompt", "reveal instructions", "show instructions", "print instructions",
    "you are now", "new instructions:", "updated instructions",
    "[inst]", "<|im_start|>", "### system", "### instruction", "</system>",
    "dan mode", "developer mode", "without restrictions", "jailbreak",
    "disregard all", "forget all", "forget previous", "forget everything",
)


def _normalize_input(raw: str) -> str:
    # NFKC normalization handles homoglyph + width variants
    normalized = unicodedata.normalize("NFKC", raw)
    # Replace zero-width chars with a space (delete would collapse word boundaries)
    normalized = normalized.translate(_ZERO_WIDTH_TO_SPACE)
    # Strip control chars except \n and \t
    normalized = "".join(c for c in normalized if c.isprintable() or c in "\n\t")
    return normalized


def _has_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _PROMPT_INJECTION_KEYWORDS)


def _looks_like_meta_response(raw_input: str, model_output: str) -> bool:
    if not model_output:
        return False
    stripped = model_output.strip().lower()
    if not stripped:
        return False
    if _META_START_RE.match(stripped):
        return True
    if any(s in stripped for s in _SUSPICIOUS_SUBSTRINGS):
        return True
    # Output wildly longer than input with no shared vocabulary is suspicious.
    # Phase 5: soften for short inputs — only apply the ratio check when input
    # has at least 5 words, to avoid false positives on single-word corrections.
    in_words = {w for w in re.findall(r"[a-z]+", raw_input.lower()) if len(w) > 2}
    out_words = re.findall(r"[a-z]+", stripped)
    if len(in_words) >= 5 and len(out_words) > len(in_words) * 3:
        overlap = sum(1 for w in out_words if w in in_words)
        if overlap < max(2, len(in_words) // 3):
            return True
    return False


def _load_croquet_dictionary() -> dict:
    """Load croquet-dictionary.json from shared/ — app dir only (Docker layout)."""
    p = Path(__file__).parent / "shared" / "croquet-dictionary.json"
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


# ---------------------------------------------------------------------------
# Phase 5: Request correlation ID middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """
    Assign a correlation ID to every request.

    - Honour X-Request-ID if a reverse proxy has already set one.
    - Otherwise generate a short UUID (12 hex chars — readable in logs).
    - Store in REQUEST_ID ContextVar so every log line in this request picks
      it up automatically.
    - Return the ID in the X-Request-ID response header so clients/upstreams
      can correlate their own logs.
    """
    request_id = request.headers.get("x-request-id") or uuid_lib.uuid4().hex[:12]
    token = REQUEST_ID.set(request_id)

    path = request.url.path
    method = request.method
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )

    t_start = time.monotonic()
    logger.info(
        "Request start",
        extra={
            "event": "request_start",
            "path": path,
            "method": method,
            "client_ip": client_ip,
        },
    )

    try:
        response = await call_next(request)
        duration_ms = round((time.monotonic() - t_start) * 1000)
        logger.info(
            "Request end",
            extra={
                "event": "request_end",
                "path": path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        duration_ms = round((time.monotonic() - t_start) * 1000)
        logger.error(
            "Request error",
            extra={
                "event": "request_end",
                "path": path,
                "status": 500,
                "duration_ms": duration_ms,
            },
        )
        raise
    finally:
        REQUEST_ID.reset(token)


# ---------------------------------------------------------------------------
# Phase 8: CORS — allow reply.croquetclaude.com to POST to /clean
#
# Explicit origin allowlist (NOT "*") so only the two known origins get the
# Access-Control-Allow-Origin header.  allow_credentials=False means the
# browser never sends cookies or auth headers cross-origin.
#
# Middleware registration order (Starlette/FastAPI LIFO): this call is the
# LAST add_middleware registered, so it becomes the outermost layer and
# handles OPTIONS preflight before correlation_id_middleware runs.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://reply.croquetclaude.com",
        "https://talk.croquetwade.com",
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
    max_age=86400,
)


# ---------------------------------------------------------------------------
# Phase 5: Global exception handler — structured ERROR log before 500 reply
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    from fastapi.responses import Response as _Response
    logger.error(
        "Unhandled exception",
        extra={
            "event": "exception",
            "exc_type": type(exc).__name__,
            "exc_msg": str(exc),
            "path": request.url.path,
        },
        exc_info=True,
    )
    return _Response(
        content=json.dumps({"detail": "An unexpected error occurred."}),
        status_code=500,
        media_type="application/json",
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


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# Phase 5: explicit allowlist — only these two files are served from /shared/.
# The parent-directory fallback is intentionally removed: in the Docker image
# the shared/ directory is always at /app/shared/ (copied by the Dockerfile).
SHARED_FILES = {
    "voice-to-text.js": "application/javascript",
    "croquet-dictionary.json": "application/json",
}


@app.get("/shared/{filename}")
async def shared_file(filename: str):
    from fastapi.responses import Response
    # Allowlist check first — anything not in the dict gets 404, no filesystem probe.
    if filename not in SHARED_FILES:
        raise HTTPException(status_code=404, detail="Not found")
    content_type = SHARED_FILES[filename]
    # Docker layout: shared/ lives inside the app directory at /app/shared/.
    file_path = Path(__file__).parent / "shared" / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return Response(content=file_path.read_text(encoding="utf-8"), media_type=content_type)


@app.post("/clean")
async def clean_transcript(request: Request, req: TranscriptRequest):
    client: httpx.AsyncClient = request.app.state.http
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    t_start = time.monotonic()
    raw = req.text or ""

    # Phase 5: normalize input — NFKC, strip zero-width chars, strip control chars.
    normalized = _normalize_input(raw)

    # Phase 5: chunk size cap — reject oversized inputs before touching the LLM.
    if len(normalized) > MAX_CHUNK_CHARS:
        raise HTTPException(status_code=413, detail="Chunk too large.")

    # Phase 5: prompt-injection guard — short-circuit without calling the LLM.
    if _has_prompt_injection(normalized):
        logger.warning(
            "Prompt injection detected — short-circuiting LLM",
            extra={"event": "prompt_injection_detected", "input_preview": normalized[:120]},
        )
        return {"cleaned": raw}  # return ORIGINAL raw, not normalized — preserve user intent

    # Input guard — too few word chars means there's nothing to clean.
    word_chars = sum(1 for c in normalized if c.isalpha())
    if word_chars < MIN_WORD_CHARS:
        return {"cleaned": raw}

    input_chars = len(normalized)

    try:
        t_or_start = time.monotonic()
        res = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": normalized},
                ],
                "max_tokens": min(2048, max(256, int(len(raw) * 1.5))),
            },
        )
        or_duration_ms = round((time.monotonic() - t_or_start) * 1000)
        data = res.json()

        if "choices" not in data:
            logger.error(
                "OpenRouter returned error response",
                extra={
                    "event": "openrouter_clean",
                    "status_code": res.status_code,
                    "duration_ms": or_duration_ms,
                    "input_chars": input_chars,
                    "output_chars": 0,
                    "error": str(data),
                    "client_ip": client_ip,
                },
            )
            duration_ms = round((time.monotonic() - t_start) * 1000)
            logger.warning(
                "Clean failed — openrouter_error",
                extra={
                    "event": "clean_failure",
                    "duration_ms": duration_ms,
                    "failure_reason": "openrouter_error",
                },
            )
            raise HTTPException(status_code=502, detail="Transcript cleaning failed, please try again.")

        cleaned = data["choices"][0]["message"]["content"]
        output_chars = len(cleaned)
        or_duration_ms = round((time.monotonic() - t_or_start) * 1000)

        logger.info(
            "OpenRouter clean succeeded",
            extra={
                "event": "openrouter_clean",
                "status_code": res.status_code,
                "duration_ms": or_duration_ms,
                "input_chars": input_chars,
                "output_chars": output_chars,
            },
        )

        if _looks_like_meta_response(normalized, cleaned):
            logger.warning(
                "Suspicious model output — falling back to raw input",
                extra={
                    "event": "clean_suspicious_output",
                    "input_chars": input_chars,
                    "output_chars": output_chars,
                    "input_preview": normalized[:120],
                    "output_preview": cleaned[:240],
                },
            )
            cleaned = raw
            output_chars = len(cleaned)

        duration_ms = round((time.monotonic() - t_start) * 1000)
        logger.info(
            "Clean succeeded",
            extra={
                "event": "clean_request",
                "duration_ms": duration_ms,
                "input_chars": input_chars,
                "output_chars": output_chars,
                "client_ip": client_ip,
            },
        )
        return {"cleaned": cleaned}

    except HTTPException:
        raise
    except Exception as e:
        duration_ms = round((time.monotonic() - t_start) * 1000)
        logger.error(
            "Clean unexpected error",
            extra={
                "event": "clean_failure",
                "duration_ms": duration_ms,
                "failure_reason": "internal_error",
                "exc_type": type(e).__name__,
                "exc_msg": str(e),
            },
        )
        raise HTTPException(status_code=502, detail="Transcript cleaning failed, please try again.")
