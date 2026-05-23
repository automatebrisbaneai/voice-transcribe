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
# read=30s    : normal API responses (OpenCode JSON endpoints)
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


# docs_url/redoc_url/openapi_url all None: don't advertise the API surface
# to anyone who guesses the conventional FastAPI paths.
app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

OPENCODE_GO_KEY = os.environ.get("OPENCODE_GO_KEY", "")
MODEL = "deepseek-v4-flash"  # OpenCode Go bare slug
OPENCODE_GO_URL = "https://opencode.ai/zen/go/v1/chat/completions"

# Shared-secret header for /clean. Empty default keeps tests + dev unauthenticated;
# in Coolify the env var is set and every browser pulls the value via the rendered HTML.
CLEAN_SHARED_KEY = os.environ.get("CLEAN_SHARED_KEY", "")
CLEAN_SHARED_KEY_HEADER = "X-Talk-Key"

# Per-IP rate limit. Default high enough that the test suite never trips it;
# Coolify can tune via env without code changes.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
RATE_LIMIT_WINDOW_S = 60.0

MIN_WORD_CHARS = 3
MAX_CHUNK_CHARS = 2_000

# Hard cap on the raw request body BEFORE FastAPI/Pydantic parses it.
# MAX_CHUNK_CHARS=2000 is the application-layer limit on the cleaned text;
# the JSON envelope adds a small fixed overhead, so 8KB gives headroom without
# letting an attacker stream a 50MB body just to land in a 413 reply.
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", "8192"))

# ---------------------------------------------------------------------------
# Per-IP rate limit — sliding window in process memory.
#
# Single-instance Coolify deployment: an in-process dict is sufficient.
# If we ever scale horizontally, replace with Redis.
# ---------------------------------------------------------------------------
from collections import defaultdict, deque

_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_rate_request_count = 0  # for periodic empty-bucket purge


def _check_rate_limit(ip: str) -> bool:
    """Return True if request is under the limit, False if over.

    Sliding-window over RATE_LIMIT_WINDOW_S. time.monotonic() is immune to
    wall-clock adjustments (NTP corrections, DST) which would otherwise let
    a buggy clock either reset or block buckets unexpectedly.
    """
    global _rate_request_count
    now = time.monotonic()
    bucket = _rate_buckets[ip]
    cutoff = now - RATE_LIMIT_WINDOW_S
    while bucket and bucket[0] < cutoff:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        return False
    bucket.append(now)
    # Periodic global prune. Drop ANY bucket whose newest timestamp is older
    # than the window cutoff — covers both empty deques (handled by the same
    # condition since max() raises ValueError on empty, guarded below) and
    # one-shot IPs whose single old timestamp would otherwise leak forever.
    _rate_request_count += 1
    if _rate_request_count % 500 == 0:
        stale_keys = [
            k for k, b in _rate_buckets.items()
            if not b or b[-1] < cutoff
        ]
        for k in stale_keys:
            del _rate_buckets[k]
    return True


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


def _resolve_shared_dir() -> Path:
    """Find the shared/ directory wherever it actually lives.

    Docker layout: ./shared/ is inside the app dir (Dockerfile `COPY . .` from
    the deploy submodule which has its own shared/).
    Dev layout: shared/ is a sibling at apps/shared/ (the canonical source).
    Falls back to the Docker path so `shared_file` returns clean 404s rather
    than tracebacks when neither exists.
    """
    here = Path(__file__).parent
    for candidate in (here / "shared", here.parent / "shared"):
        if candidate.is_dir():
            return candidate
    return here / "shared"


SHARED_DIR = _resolve_shared_dir()


def _load_croquet_dictionary() -> dict:
    """Load croquet-dictionary.json from whichever shared/ directory resolved."""
    p = SHARED_DIR / "croquet-dictionary.json"
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
    # Pin on request.state too, so the global exception handler can read it
    # AFTER this middleware's finally has reset the ContextVar.
    request.state.request_id = request_id

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
# Body-size guard — rejects oversized requests BEFORE FastAPI/Pydantic parses
# the body. Without this, a hostile client can stream a 10MB JSON envelope and
# we pay the parse cost before the route handler's chunk-cap check runs.
# Runs AFTER correlation_id middleware (so logged with request_id) and BEFORE
# the route, courtesy of FastAPI's LIFO middleware stack and the registration
# order at this point in the file.
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_size_middleware(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_REQUEST_BYTES:
                logger.warning(
                    "Request rejected — body too large",
                    extra={
                        "event": "request_too_large",
                        "content_length": cl,
                        "path": request.url.path,
                    },
                )
                from fastapi.responses import Response as _Response
                return _Response(
                    content=json.dumps({"detail": "Payload too large"}),
                    status_code=413,
                    media_type="application/json",
                )
        except ValueError:
            # Bad Content-Length string → let downstream handle it.
            pass
    return await call_next(request)


# ---------------------------------------------------------------------------
# CORS — explicit origin allowlist so only the three known sites cross-fetch
# /clean. allow_credentials=False only blocks browser-managed credentials
# (cookies, HTTP auth); custom headers like X-Talk-Key still pass when listed
# in allow_headers. Without X-Talk-Key in the list, browser preflight blocks
# any cross-origin POST that tries to send it.
#
# Middleware registration order (Starlette/FastAPI LIFO): this is the LAST
# middleware registered, so it wraps everything else and handles OPTIONS
# preflight before correlation_id_middleware / request_size_middleware run.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://reply.croquetclaude.com",
        "https://talk.croquetwade.com",
        "https://table.croquetclaude.com",
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", CLEAN_SHARED_KEY_HEADER],
    allow_credentials=False,
    max_age=86400,
)


# ---------------------------------------------------------------------------
# Phase 5: Global exception handler — structured ERROR log before 500 reply
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    from fastapi.responses import Response as _Response
    # By the time this runs, the middleware's `finally` has already reset the
    # REQUEST_ID ContextVar — so we read the pinned-on-state copy instead.
    request_id = getattr(request.state, "request_id", "")
    logger.error(
        "Unhandled exception",
        extra={
            "event": "exception",
            "request_id": request_id,
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
    "You are a transcript cleaner. Your only job: convert spoken voice into clean "
    "typed text without changing meaning, voice, or word choice.\n"
    "\n"
    "CORE RULE: If a sentence already makes sense, leave it alone. You are not "
    "improving the writing. You are removing fillers, adding punctuation, and fixing "
    "obvious transcription errors. Nothing else.\n"
    "\n"
    "EDIT RULES (apply in order):\n"
    "\n"
    "1. Self-corrections — when the speaker corrects themselves with \"sorry / I mean "
    "/ scratch that / no wait / actually / strike that\", keep what they corrected "
    "to and drop what they corrected from.\n"
    "\n"
    "2. Filler removal — delete: um, uh, ah, er, \"you know\" (when used as filler).\n"
    "\n"
    "3. Spoken commands — convert to formatting:\n"
    "   \"new line\" → line break\n"
    "   \"new paragraph\" → blank line\n"
    "   \"period\" → .   \"comma\" → ,   \"question mark\" → ?\n"
    "\n"
    "4. Capitalisation and punctuation — add where missing.\n"
    "\n"
    "5. Numbers — above twelve become digits (500, 20%); 1–12 stay as words; keep "
    "years and proper names as-is.\n"
    "\n"
    "6. Spelling — British English (organise, colour, behaviour, recognise).\n"
    "\n"
    "7. Context fixes — repair obvious transcription errors when surrounding words "
    "make the intended word clear. If you cannot tell what was meant, leave the "
    "word alone. Examples:\n"
    "     \"the meating went well\" → \"meeting\"\n"
    "     \"develop a coquet strategy\" → \"croquet\"\n"
    "\n"
    + (("KNOWN VOCABULARY — " + _DICTIONARY_HINT + "\n\n") if _DICTIONARY_HINT else "")
    + "FORBIDDEN:\n"
    "- Rephrasing, reordering, or restructuring sentences.\n"
    "- Adding words, transitions, headers, or commentary.\n"
    "- Em dashes — use commas or full stops.\n"
    "- Converting prose into bullet lists unless the speaker is clearly enumerating.\n"
    "- Markdown formatting (bold, italics, headers) unless input contained it.\n"
    "\n"
    "OUTPUT FORMAT — read this carefully:\n"
    "Output the cleaned transcript directly. No preamble like \"Here is\" or \"Sure\". "
    "No quotes around the output. No explanation after it. Just the cleaned text. "
    "If the input is empty, whitespace, a single word, or otherwise not a usable "
    "transcript, return it verbatim.\n"
    "\n"
    "EXAMPLE:\n"
    "Input: um so I went to the meating yesterday sorry the meeting yesterday and we "
    "talked about you know the new croquet strategy for twenty twenty six\n"
    "Output: I went to the meeting yesterday and we talked about the new croquet "
    "strategy for 2026."
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


_INDEX_PATH = Path(__file__).parent / "index.html"
_INDEX_PLACEHOLDER = "__CLEAN_SHARED_KEY_HEADERS__"


@app.get("/")
async def root():
    """Serve index.html with the shared-secret headers rendered as a JS object literal.

    The placeholder in index.html sits where a JSON object literal goes:
        window.VTT_EXTRA_HEADERS = __CLEAN_SHARED_KEY_HEADERS__;
    We substitute with `{"X-Talk-Key": "<key>"}` when the secret is set, or `{}`
    when unset (dev mode — voice-to-text.js sends no extra header).

    Read on every request because Coolify hot-replaces the file on deploy and
    keeps the same process running between releases. File is tiny so the extra
    disk hit is irrelevant.
    """
    html = _INDEX_PATH.read_text(encoding="utf-8")
    headers_obj = (
        json.dumps({CLEAN_SHARED_KEY_HEADER: CLEAN_SHARED_KEY})
        if CLEAN_SHARED_KEY
        else "{}"
    )
    rendered = html.replace(_INDEX_PLACEHOLDER, headers_obj)
    return HTMLResponse(rendered)


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
    file_path = SHARED_DIR / filename
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

    # Shared-secret gate — only active when CLEAN_SHARED_KEY env is set.
    # Generic 403 (no key name in the response body) so probing tools learn nothing.
    if CLEAN_SHARED_KEY and request.headers.get(CLEAN_SHARED_KEY_HEADER) != CLEAN_SHARED_KEY:
        logger.warning(
            "Forbidden clean attempt (missing/invalid auth header)",
            extra={"event": "clean_forbidden", "client_ip": client_ip},
        )
        raise HTTPException(status_code=403, detail="Forbidden")

    # Per-IP rate limit — drops drive-by abuse before we hit the LLM. Trusted
    # browsers carrying the shared secret are still subject to the same limit
    # so a compromised key cannot rinse the budget from a single host.
    if not _check_rate_limit(client_ip):
        logger.warning(
            "Rate limit exceeded",
            extra={"event": "clean_rate_limited", "client_ip": client_ip},
        )
        raise HTTPException(status_code=429, detail="Too many requests")

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
            OPENCODE_GO_URL,
            headers={
                "Authorization": f"Bearer {OPENCODE_GO_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "croquetwade-worker/1.0",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": normalized},
                ],
                # DeepSeek V4 on OpenCode burns substantial hidden reasoning tokens
                # against max_tokens (OR's reasoning={"effort":"none"} extension is
                # silently ignored). 500-token floor was too tight — reasoning ate
                # every token, content came back empty. 4096 covers reasoning + actual
                # cleaned output even for the longest practical transcripts.
                "max_tokens": 4096,
            },
        )
        or_duration_ms = round((time.monotonic() - t_or_start) * 1000)
        data = res.json()

        if "choices" not in data:
            logger.error(
                "Upstream LLM returned error response",
                extra={
                    "event": "upstream_clean",
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
                "Clean failed — upstream_error",
                extra={
                    "event": "clean_failure",
                    "duration_ms": duration_ms,
                    "failure_reason": "upstream_error",
                },
            )
            raise HTTPException(status_code=502, detail="Transcript cleaning failed, please try again.")

        cleaned = data["choices"][0]["message"]["content"]
        output_chars = len(cleaned)
        or_duration_ms = round((time.monotonic() - t_or_start) * 1000)

        logger.info(
            "Upstream clean succeeded",
            extra={
                "event": "upstream_clean",
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
