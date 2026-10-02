"""Gemini API access for DocPilot AI.

Two small adapters sit on top of the official `google-genai` SDK so the
rest of the application keeps talking to a chat model and to an
embeddings object exactly as it did when both ran inside the Ollama
container. Nothing else in the project needs to know that the
generation layer moved to the Gemini API.

The API key is read from the environment on every use and is never
stored in a module level variable, printed, logged or returned in an
error message.
"""

import os
import re

from google import genai
from google.genai import errors, types
from langchain_core.embeddings import Embeddings


# ============================================================
# DEFAULTS
# ============================================================

# Current stable Gemini model that is usable on the free tier.
# Chosen over gemini-3.8-flash because it answers the same document
# questions just as accurately on the same grounding rules, while
# answering in 2 to 4 seconds instead of stalling on demand spikes.
# Override with the GEMINI_MODEL environment variable.
DEFAULT_CHAT_MODEL = "gemini-3.5-flash-lite"

# Text embedding model on the Gemini Developer API.
# Override with the GEMINI_EMBEDDING_MODEL environment variable.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# nomic-embed-text produced 768 dimensional vectors and the existing
# Atlas vector index is defined for numDimensions: 768 with cosine
# similarity, so the Gemini embedding is truncated to the same width.
# The index definition therefore stays valid and does not have to be
# rebuilt: only the stored vectors have to be produced again, which the
# upload path already does on every upload.
EMBEDDING_DIMENSIONS = 768

# Texts per embedding request. Batching keeps the number of round trips
# low, which matters on the free tier rate limits.
DEFAULT_EMBED_BATCH_SIZE = 64

# Retries for a retryable Gemini failure (429, 500, 502, 503, 504).
# The free tier returns 503 during demand spikes, so this is higher
# than the SDK default of 3.
DEFAULT_MAX_RETRIES = 5


# ============================================================
# ERRORS
# ============================================================

class GeminiConfigurationError(RuntimeError):
    """Raised when the Gemini API key or model name is not configured.

    Kept separate from GeminiError so the backend can answer with a
    clear "not configured" message instead of a generic failure.
    """


class GeminiError(RuntimeError):
    """Raised when a Gemini request fails.

    The message is already redacted, so it can be returned to the
    browser and printed in the logs as is.
    """


# ============================================================
# API KEY AND CLIENT
# ============================================================

# One client per key. The SDK client is stateless for our use, and
# rebuilding it per request would re-parse HTTP configuration every
# time. The key itself is deliberately not kept in a global.
_clients = {}


def api_key():
    """The Gemini API key from the environment.

    GEMINI_API_KEY is the documented name for this project.
    GOOGLE_API_KEY is also accepted because the SDK reads it, so a
    hosting platform that only exposes that variable still works.
    """
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()

    if not key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is not set. Add it to your .env file or "
            "to your hosting platform's environment variables, then "
            "restart the application."
        )

    return key


def is_configured():
    """True when an API key is available."""
    try:
        api_key()

    except GeminiConfigurationError:
        return False

    return True


def client():
    """A Gemini client for the current key."""
    key = api_key()

    if key not in _clients:
        _clients[key] = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                # The free tier regularly answers with a temporary 503
                # or 429 under load, and both the SDK and this retry
                # policy treat those as retryable. An upload embeds many
                # chunks, so a single transient failure should not lose
                # the whole document, and a chat answer should survive a
                # short demand spike instead of showing an error.
                retry_options=types.HttpRetryOptions(
                    attempts=max(
                        1,
                        int(
                            os.getenv("GEMINI_MAX_RETRIES")
                            or DEFAULT_MAX_RETRIES
                        )
                    )
                )
            )
        )

    return _clients[key]


# ============================================================
# ERROR REPORTS
# ============================================================

# Any spelling of a credential that could appear inside a message
# raised by the SDK or by an HTTP layer is masked before the text is
# logged or sent back to the browser.
_SECRET_PATTERNS = (
    re.compile(r"x-goog-api-key=([^&\s]+)", re.IGNORECASE),
    re.compile(r"\b(api[_-]?key|apikey|key)=([^&\s]+)", re.IGNORECASE),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{10,}"),
)


def redact(message):
    """Mask anything that looks like a credential in `message`."""
    text = str(message)

    for secret in (
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GOOGLE_API_KEY")
    ):
        if secret and secret.strip():
            text = text.replace(secret.strip(), "[redacted]")

    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: (
                match.group(0).split("=")[0] + "=[redacted]"
                if "=" in match.group(0)
                else "[redacted]"
            ),
            text
        )

    return text


def _friendly_error(exc):
    """A message that tells the user what to do about a failure.

    Rate limits and quota problems get their own wording because they
    are the two failures that are actually recoverable by the user on
    a free tier deployment.
    """
    text = redact(exc).strip()

    name = type(exc).__name__

    lowered = text.lower()

    rate_limited = (
        "resource_exhausted" in lowered
        or "429" in lowered
        or "quota" in lowered
        or "rate limit" in lowered
    )

    if rate_limited:
        return (
            "The Gemini API rate limit or quota was reached. Wait a "
            "moment and try again. " + text
        )

    if "permission_denied" in lowered or "api key not valid" in lowered or "unauthenticated" in lowered:
        return (
            "Gemini rejected the API key. Check that GEMINI_API_KEY "
            "is correct and enabled for the Gemini API. " + text
        )

    if "not found" in lowered and "model" in lowered:
        return (
            "The configured GEMINI_MODEL is not available for this API "
            "key. " + text
        )

    if name == "ServerError" or "503" in lowered or "overloaded" in lowered:
        return (
            "The Gemini API is temporarily unavailable. Please try "
            "again. " + text
        )

    return "Gemini request failed. " + text


# ============================================================
# PROMPT CONVERSION
# ============================================================

_ROLE_LABELS = {
    "system": "Instruction",
    "human": "User",
    "user": "User",
    "ai": "DocPilot AI",
    "assistant": "DocPilot AI"
}


def _content_text(content):
    """Flatten one message body into plain text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # Content blocks as used by the multimodal message types.
    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            else:
                parts.append(
                    getattr(block, "text", "") or ""
                )

        return "".join(parts)

    return str(content)


def to_prompt_text(messages):
    """Turn whatever the caller passed into one prompt string.

    Every prompt in this application is a single template rendered
    into one human message, so the common case is returned unchanged.
    Several messages are joined with their role, which keeps the
    adapter correct if history is ever passed as real messages.
    """
    if messages is None:
        return ""

    if isinstance(messages, str):
        return messages

    # A ChatPromptValue or any other prompt object.
    to_messages = getattr(messages, "to_messages", None)

    if callable(to_messages):
        messages = to_messages()

    elif not isinstance(messages, (list, tuple)):
        messages = [messages]

    if isinstance(messages, (list, tuple)) and len(messages) == 1:
        return _content_text(
            getattr(messages[0], "content", messages[0])
        )

    blocks = []

    for message in messages or []:
        text = _content_text(
            getattr(message, "content", message)
        )

        if not text.strip():
            continue

        kind = (
            getattr(message, "type", None)
            or getattr(message, "role", None)
            or ""
        )

        label = _ROLE_LABELS.get(str(kind).lower())

        blocks.append(
            f"{label}: {text}" if label else text
        )

    return "\n\n".join(blocks)


# ============================================================
# RESPONSES
# ============================================================

def _response_text(response):
    """The text of a generateContent response.

    `response.text` is None when the model returned only thought
    tokens or was blocked, so the parts are read directly as well.
    """
    text = getattr(response, "text", None)

    if text:
        return text

    parts = []

    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)

        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)

    return "".join(parts)


def _check_blocked(response):
    """Raise a useful error when the model produced no candidates.

    A safety block or an empty candidate list would otherwise look
    exactly like "the document does not contain the answer", which is
    a misleading message to show the user.
    """
    candidates = getattr(response, "candidates", None) or []

    if candidates:
        return

    feedback = getattr(response, "prompt_feedback", None)

    reason = getattr(feedback, "block_reason", None)

    if reason:
        raise GeminiError(
            "Gemini blocked the request before answering "
            f"(reason: {reason})."
        )

    raise GeminiError(
        "Gemini returned an empty response. Please try again."
    )


# ============================================================
# CHAT MODEL
# ============================================================

class GeminiChunk:
    """One streamed piece of an answer.

    Shaped like the chunk objects the application already reads from
    `astream()`, so the SSE code in app.py is unchanged.
    """

    __slots__ = ("content",)

    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return f"GeminiChunk({self.content!r})"


class GeminiResult:
    """One complete answer. Shaped like the old `.invoke()` result."""

    __slots__ = ("content",)

    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return f"GeminiResult({self.content!r})"


class GeminiChat:
    """Generation only, for every answer the application produces.

    Replaces ChatOllama. Retrieval, chunking, prompts, streaming and
    the fallback behaviour all stay in app.py and are untouched.
    """

    def __init__(self, model=None, thinking_level=None):
        self.model = (
            model
            or os.getenv("GEMINI_MODEL")
            or DEFAULT_CHAT_MODEL
        )

        level = (
            thinking_level
            if thinking_level is not None
            else os.getenv("GEMINI_THINKING_LEVEL")
        )

        self.thinking_level = (
            str(level).strip().upper() or None
            if level
            else None
        )

    def _config(self):
        """Request options.

        No temperature, top_p or top_k is sent: Gemini 3.x ignores
        them, so the deterministic setting the Gemma version used has
        no equivalent and must not be faked with a deprecated field.

        thinking_level is only sent when it is configured. It accepts
        low, medium and high, where low gives the fastest first token.
        """
        if not self.thinking_level:
            return None

        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=self.thinking_level
            )
        )

    def describe(self):
        """Model information safe to print in the logs."""
        return (
            f"{self.model}"
            f"{f' (thinking: {self.thinking_level})' if self.thinking_level else ''}"
        )

    def invoke(self, messages):
        """One complete answer. Used by the title and summary calls."""
        # Raises GeminiConfigurationError before any work is done.
        api_key()

        try:
            response = client().models.generate_content(
                model=self.model,
                contents=to_prompt_text(messages),
                config=self._config()
            )

        except errors.APIError as exc:
            raise GeminiError(_friendly_error(exc)) from None

        _check_blocked(response)

        return GeminiResult(_response_text(response).strip())

    async def astream(self, messages):
        """Yield the answer piece by piece, as the SSE route needs."""
        # Raises GeminiConfigurationError before the stream opens.
        api_key()

        stream = await client().aio.models.generate_content_stream(
            model=self.model,
            contents=to_prompt_text(messages),
            config=self._config()
        )

        async for response in stream:

            text = _response_text(response)

            if text:
                yield GeminiChunk(text)


# ============================================================
# EMBEDDINGS
# ============================================================

class GeminiEmbeddings(Embeddings):
    """Embeddings for chunking and for MongoDB Atlas Vector Search.

    Replaces OllamaEmbeddings(nomic-embed-text). The output width is
    held at 768 so the existing Atlas index definition keeps matching,
    and queries and documents use the retrieval task types so the
    stored vectors and the query vectors share one space.
    """

    def __init__(
        self,
        model=None,
        batch_size=None,
        dimensions=None
    ):
        self.model = (
            model
            or os.getenv("GEMINI_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )

        self.batch_size = max(
            1,
            int(
                os.getenv("EMBED_BATCH_SIZE")
                or batch_size
                or DEFAULT_EMBED_BATCH_SIZE
            )
        )

        self.dimensions = int(
            os.getenv("EMBEDDING_DIMENSIONS")
            or dimensions
            or EMBEDDING_DIMENSIONS
        )

    def describe(self):
        """Model information safe to print in the logs."""
        return f"{self.model} ({self.dimensions}d)"

    def _embed(self, texts, task_type):
        texts = list(texts)

        if not texts:
            return []

        api_key()

        vectors = []
        gemini = client()

        for start in range(0, len(texts), self.batch_size):

            batch = texts[start:start + self.batch_size]

            try:
                response = gemini.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimensions,
                        task_type=task_type
                    )
                )

            except errors.APIError as exc:
                raise GeminiError(
                    "Could not create the document embeddings. "
                    + _friendly_error(exc)
                ) from None

            vectors.extend(
                embedding.values
                for embedding in response.embeddings
            )

        if len(vectors) != len(texts):
            raise GeminiError(
                "Gemini returned "
                f"{len(vectors)} embedding(s) for "
                f"{len(texts)} text(s). Please upload the "
                "document again."
            )

        return vectors

    def embed_documents(self, texts):
        """Vectors stored in MongoDB."""
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        """Vector of the user's question, used for the search."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]