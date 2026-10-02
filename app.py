import json
import os
import shutil
import time
import uuid
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo import MongoClient

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader
)
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gemini_client import (
    DEFAULT_CHAT_MODEL,
    GeminiChat,
    GeminiEmbeddings,
    GeminiError,
    is_configured,
    redact
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

app = FastAPI(title="AI Document Assistant")


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


# ============================================================
# STATIC FILES
# ============================================================

static_path = Path(__file__).parent / "static"

if static_path.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(static_path)),
        name="static"
    )


# ============================================================
# MONGODB SETTINGS
# ============================================================

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")


if not all([
    MONGODB_URI,
    DB_NAME,
    COLLECTION_NAME,
    INDEX_NAME
]):
    raise RuntimeError(
        "MONGODB_URI, DB_NAME, COLLECTION_NAME and "
        "INDEX_NAME must be set in .env"
    )


# ============================================================
# MONGODB CONNECTION
# ============================================================

client = MongoClient(MONGODB_URI)

collection = client[DB_NAME][COLLECTION_NAME]


# ============================================================
# ATLAS SEARCH INDEX
# ============================================================
# $vectorSearch can only be restricted to a document when the
# document id is declared as a filter field on the index. Without
# it every search returns chunks from previously indexed
# documents as well, which is what made answers drift away from
# the uploaded file.
# ============================================================
# langchain-mongodb flattens Document.metadata into top level
# fields when it writes, so the document id is stored as
# "document_id" and not as "metadata.document_id". Querying
# "metadata.document_id" matches nothing, which is why chunks
# from earlier uploads were never cleaned up and were returned
# by every search.
# ============================================================

DOCUMENT_ID_FIELD = "document_id"
PAGE_FIELD = "page"
CHUNK_INDEX_FIELD = "chunk_index"


def read_chunk(document):
    """Read one stored chunk into a predictable shape.

    Chunks written by langchain-mongodb keep their metadata as
    top level fields. Accepting the nested layout as well keeps
    the rest of the code readable and unaffected if the write
    path ever changes.
    """
    metadata = document.get("metadata") or {}

    return {
        "id": document.get("_id"),
        "text": document.get("text", ""),
        "page": metadata.get(PAGE_FIELD, document.get(PAGE_FIELD)),
        "chunk_index": metadata.get(
            CHUNK_INDEX_FIELD,
            document.get(CHUNK_INDEX_FIELD)
        )
    }


def ensure_vector_index():
    """Add the document id filter field to the vector index.

    Safe to call on every start: the definition is only rewritten
    when the filter field is missing.
    """
    definition = None

    for index in collection.list_search_indexes(INDEX_NAME):
        if index.get("type") == "vectorSearch":
            definition = index.get("latestDefinition")
            break

    if not definition:
        print(
            "[index] vector index not found, "
            "MongoDB Atlas will create it on first write"
        )
        return False

    fields = [
        field for field in definition.get("fields", [])
        if not (
            field.get("type") == "filter"
            and field.get("path") != DOCUMENT_ID_FIELD
        )
    ]

    already_filterable = any(
        field.get("type") == "filter"
        and field.get("path") == DOCUMENT_ID_FIELD
        for field in fields
    )

    if already_filterable:
        print("[index] vector index already filters by document")
        return True

    fields.append({
        "type": "filter",
        "path": DOCUMENT_ID_FIELD
    })

    print("[index] adding document filter field to vector index")

    collection.update_search_index(
        name=INDEX_NAME,
        definition={"fields": fields}
    )

    return True


def clear_indexed_documents():
    """Remove every chunk previously indexed by this app.

    Only the app is allowed to keep one document at a time, so
    clearing everything it wrote is the correct behaviour. Both
    the current flat layout and the nested metadata layout are
    matched so nothing survives an upload.
    """
    result = collection.delete_many({
        "$or": [
            {DOCUMENT_ID_FIELD: {"$exists": True}},
            {"metadata.document_id": {"$exists": True}},
        ]
    })

    if result.deleted_count:
        print(
            f"[index] removed {result.deleted_count} "
            "chunk(s) from previously indexed documents"
        )

    return result.deleted_count


# ============================================================
# EMBEDDINGS
# ============================================================
# Generation moved to the Gemini API, so the embedding model that
# used to come from the Ollama container moves with it. Gemini
# embeddings are truncated to 768 dimensions, which is exactly what
# nomic-embed-text produced, so the Atlas vector index definition
# (numDimensions 768, cosine, document_id filter) stays valid and
# does not have to be rebuilt. The stored vectors themselves are
# produced again by the next upload, which already clears every chunk
# the application ever wrote.
# ============================================================

# Concurrent embedding requests during an upload. The free tier limits
# requests per minute, so this is deliberately modest and can be
# lowered further on a shared key.
EMBED_WORKERS = max(
    1,
    int(os.getenv("EMBED_WORKERS", "4"))
)

embeddings = GeminiEmbeddings()


def store_chunks(chunks):
    """Embed and store chunks, parallelizing embedding requests.

    Each worker sends its own batch of texts to the Gemini embedding
    endpoint. The worker count is capped because the free tier limits
    requests per minute, and several workers in flight is what used to
    make large documents index much faster than one request at a time.
    Results are identical either way.
    """
    workers = min(
        EMBED_WORKERS,
        max(1, os.cpu_count() or 2)
    )

    if len(chunks) <= 12 or workers < 2:
        vector_store.add_documents(chunks)
        return

    split = max(1, len(chunks) // workers)

    groups = [
        chunks[i:i+split]
        for i in range(0, len(chunks), split)
    ]

    with ThreadPoolExecutor(
        max_workers=min(workers, len(groups))
    ) as pool:
        list(pool.map(vector_store.add_documents, groups))


# ============================================================
# MONGODB ATLAS VECTOR STORE
# ============================================================

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=f"{DB_NAME}.{COLLECTION_NAME}",
    embedding=embeddings,
    index_name=INDEX_NAME,
)


# ============================================================
# LLM
# ============================================================
# Answer generation runs on the Gemini API. Retrieval, prompts,
# chunking, streaming and the fallback message below are unchanged,
# so this is still a document grounded RAG application and not a
# general purpose chat model.
#
# The model name comes from the environment, and the API key is only
# ever read from the environment as well. Nothing is hard coded, and
# no key is logged or returned to the browser.
# ============================================================

GEMINI_MODEL = (
    os.getenv("GEMINI_MODEL", "").strip()
    or DEFAULT_CHAT_MODEL
)

llm = GeminiChat(model=GEMINI_MODEL)


def require_gemini():
    """Stop with a clear error when the API key is missing.

    Called at the start of every route that generates text, so a
    missing key produces one clear message instead of a stack trace
    or a generic failure halfway through a stream.
    """
    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "The Gemini API key is not configured. Set "
                "GEMINI_API_KEY in your environment and restart "
                "the application."
            )
        )


# ============================================================
# RAG PROMPTS
# ============================================================
# Answers must stay grounded in the uploaded document, so the
# rules are spelled out for both the passage and the
# whole-document prompt.
# ============================================================

GROUNDING_RULES = """
Rules:
- The uploaded document is the source of truth. Answer the
  user's question using the Context below and nothing else.
- Answer the question that was actually asked. Do not turn every
  question into an overview or a summary, and do not simply
  repeat the passage that looks most similar to the question.
- Match the answer to the question: a direct explanation for
  "what is X", the listed items for "what are the advantages",
  the specific detail asked for, a summary only when a summary
  is requested, and key points only when key points are asked
  for.
- The user rarely uses the document's own words. When the
  wording differs, use the Context passage that explains the
  same idea instead of insisting on matching words.
- If several passages are relevant, combine them into one answer
  instead of listing them one by one.
- Never use outside or general knowledge and never guess.
- Do not add facts, examples, numbers, or details that are not
  in the Context.
- If the Context does not contain the answer, reply exactly:
  "I couldn't find that information in the uploaded document."
  Saying this is always better than inventing an answer.
- A passage that only shares words with the question is not an
  answer. Use the passage that actually states what was asked.
- Refer to a marked passage as "Page N" when the Context shows
  a page number.
- Read all of the Context before answering, not only the first
  passage.
- Keep answers clear, concise, and professional. Give a short
  direct answer for a specific question and a fuller answer only
  when the question asks for a summary or an overview.
"""

prompt = ChatPromptTemplate.from_template(
    """
You are DocPilot AI, a document question-answering assistant.

""" + GROUNDING_RULES + """
Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
)


document_prompt = ChatPromptTemplate.from_template(
    """
You are DocPilot AI, a document question-answering assistant.

The Context below holds excerpts taken from across the WHOLE
uploaded document, in document order and marked with the page
they came from. It is a coverage sample, not a single passage.

""" + GROUNDING_RULES + """
- Because the Context covers the whole document, describe the
  document as a whole rather than a single section.
- Follow the order in which the passages appear.

Context:
{context}

Question:
{question}

Answer:
"""
)


section_prompt = ChatPromptTemplate.from_template(
    """
You are DocPilot AI summarising one part of an uploaded
document so the parts can be combined into a whole-document
answer later.

Write only what this part of the Context actually states.
Do not add outside knowledge, and do not comment on the part
itself. Keep it to a few factual sentences.
"""
)


combine_prompt = ChatPromptTemplate.from_template(
    """
You are DocPilot AI, a document question-answering assistant.

Notes below were written from excerpts of the WHOLE uploaded
document, in document order.

""" + GROUNDING_RULES + """
- Use the notes to cover the document as a whole, not just the
  first note.
- Do not repeat the same point more than once.

Notes:
{context}

Question:
{question}

Answer:
"""
)


title_prompt = ChatPromptTemplate.from_template(
    """
You name chat conversations about one uploaded document.

Reply with the name only: no quotes, no full stop, no
explanation, no question mark, nothing else.

Rules:
- Use 2 to 5 words.
- Name the subject of the document, for example
  "Java Interview Preparation" or "Cloud Computing".
- Prefer the topic of the document over the wording of the
  user's question.
- Leave out file names, file extensions, and the words
  "document", "PDF" and "question".
- Never answer the question itself.

Document topic: {document}

Opening of the document: {excerpt}

First thing the user asked: {question}

Name:
"""
)


# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# ============================================================
# APPLICATION STATE
# ============================================================

chat_history = []

current_document_id = None
current_filename = None
current_file_type = None
current_page_count = None
current_unit = "pages"


# ============================================================
# CHAT REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


class TitleRequest(BaseModel):
    message: str = ""


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def on_startup():
    # Retrieval is only reliable when the index can filter by
    # document, so the check runs once when the app starts.
    try:
        ensure_vector_index()

    except Exception as exc:
        print(f"[index] could not verify vector index: {exc}")

    # Which model is in use is printed, the key never is. A missing
    # key is only a warning here: the rest of the application, the
    # upload path and the interface all still work, and the routes
    # that generate text report the problem themselves.
    print(
        f"[gemini] generation model: {llm.describe()} | "
        f"embedding model: {embeddings.describe()} | "
        f"api key: "
        f"{'configured' if is_configured() else 'MISSING'}"
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def root():

    index_file = static_path / "index.html"

    if not index_file.exists():
        return HTMLResponse(
            """
            <h1>AI Document Assistant</h1>
            <p>static/index.html was not found.</p>
            """
        )

    return HTMLResponse(
        index_file.read_text(encoding="utf-8")
    )


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    global current_document_id
    global current_filename
    global current_file_type
    global current_page_count
    global current_unit
    global chat_history

    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Please select a file."
        )


    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )


    # --------------------------------------------------------
    # Check the API key
    #
    # Indexing embeds every chunk, so a missing key is reported here
    # instead of half way through the upload.
    # --------------------------------------------------------

    require_gemini()


    # --------------------------------------------------------
    # Create document ID
    # --------------------------------------------------------

    document_id = str(uuid.uuid4())


    # --------------------------------------------------------
    # Create upload directory
    # --------------------------------------------------------

    upload_dir = Path(__file__).parent / "uploads"

    upload_dir.mkdir(exist_ok=True)


    # --------------------------------------------------------
    # Temporary file path
    # --------------------------------------------------------

    temp_path = upload_dir / f"{document_id}{extension}"


    try:

        t0 = time.time()

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        with temp_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        print(f"[upload] saved file: {time.time()-t0:.1f}s")


        # ----------------------------------------------------
        # Load document
        # ----------------------------------------------------

        t1 = time.time()

        if extension == ".pdf":
            loader = PyPDFLoader(str(temp_path))
        else:
            loader = Docx2txtLoader(str(temp_path))

        pages = loader.load()

        loaded_pages = len(pages)

        print(
            f"[upload] {extension} parsed: "
            f"{time.time()-t1:.1f}s"
        )

        # Everything the loader produced is reported so a page
        # that yields no text is visible instead of silently
        # disappearing from the index.
        page_report = []

        for position, loaded_page in enumerate(pages, start=1):

            page_report.append({
                "unit": position,
                "chars": len(
                    loaded_page.page_content.strip()
                )
            })

        extracted_chars = sum(
            entry["chars"] for entry in page_report
        )

        print(
            f"[ingest] pages loaded: {loaded_pages} | "
            f"chars per page: "
            f"{[e['chars'] for e in page_report]} | "
            f"total chars: {extracted_chars}"
        )


        # ----------------------------------------------------
        # Count pages or sections
        # ----------------------------------------------------
        # A Word loader returns the whole file as a single
        # document, so the blank-line separated blocks are
        # counted for the document panel. The document itself
        # is left intact so chunking still works on the full
        # text instead of one tiny chunk per paragraph.
        # ----------------------------------------------------

        if extension == ".pdf":

            unit = "pages"

            unit_count = len(pages)

        else:

            unit = "sections"

            unit_count = sum(
                1
                for loaded_page in pages
                for block in re.split(
                    r"\n\s*\n",
                    loaded_page.page_content
                )
                if block.strip()
            )

        unit_count = max(unit_count, 1)


        # ----------------------------------------------------
        # Clean PDF text
        # ----------------------------------------------------
        # Some PDFs extract characters with spaces between them,
        # e.g. "W o r k s b o t". Join consecutive single-
        # character tokens so names and URLs become searchable.
        # ----------------------------------------------------
        def clean_pdf_text(raw_text):
            cleaned_lines = []
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    cleaned_lines.append("")
                    continue

                tokens = line.split()
                output = []
                char_buffer = []

                def flush_buffer():
                    if char_buffer:
                        output.append("".join(char_buffer))
                        char_buffer.clear()

                for token in tokens:
                    if len(token) == 1:
                        char_buffer.append(token)
                    else:
                        flush_buffer()
                        output.append(token)

                flush_buffer()
                cleaned_lines.append(" ".join(output))

            cleaned = "\n".join(cleaned_lines)
            cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
            cleaned = re.sub(r"([({[])\s+", r"\1", cleaned)
            cleaned = re.sub(r"\s+([)}])", r"\1", cleaned)
            return cleaned

        for page in pages:
            if extension == ".pdf":
                page.page_content = clean_pdf_text(
                    page.page_content
                )
            else:
                page.page_content = re.sub(
                    r"[ \t]+",
                    " ",
                    page.page_content
                )

        # Remove pages that carry no text at all. A scanned or
        # image-only page has nothing to embed, so it is reported
        # back to the caller rather than being counted as
        # indexed.
        pages = [
            page for page in pages
            if len(page.page_content.strip()) > 2
        ]

        empty_units = [
            entry["unit"] for entry in page_report
            if entry["chars"] <= 2
        ]

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="The document does not contain readable text."
            )

        print(f"[upload] text cleaned: {time.time()-t1:.1f}s")


        # ----------------------------------------------------
        # Split document into chunks
        # ----------------------------------------------------

        chunks = text_splitter.split_documents(
            pages
        )

        print(
            f"[upload] split into {len(chunks)} chunks: "
            f"{time.time()-t1:.1f}s"
        )


        # ----------------------------------------------------
        # Add metadata
        # ----------------------------------------------------
        # document_id scopes every later search to this file,
        # page keeps the provenance of each chunk, and
        # chunk_index restores document order when the whole
        # document is read back for a summary.
        # ----------------------------------------------------

        for index, chunk in enumerate(chunks):

            chunk.metadata["document_id"] = document_id

            chunk.metadata["filename"] = file.filename

            chunk.metadata["file_type"] = extension.lstrip(".")

            chunk.metadata["chunk_index"] = index

            # PyPDFLoader numbers pages from 0 and that number
            # is kept even when an empty page is dropped, so the
            # page shown to the user is the real one in the file.
            source_page = chunk.metadata.get("page")

            if isinstance(source_page, int):
                chunk.metadata["page"] = source_page + 1
                chunk.metadata["total_pages"] = unit_count
            else:
                chunk.metadata["page"] = None
                chunk.metadata.pop("total_pages", None)

        # Chunk level diagnostics: which page every chunk came
        # from, so a missing page is obvious in the log instead of
        # only showing up as a wrong answer later.
        chunks_per_page = {}

        for chunk in chunks:
            page = chunk.metadata.get("page")
            key = page if page is not None else "none"
            chunks_per_page[key] = chunks_per_page.get(key, 0) + 1

        indexed_pages = sorted(
            page for page in chunks_per_page
            if page != "none"
        )

        print(
            f"[ingest] chunks: {len(chunks)} | "
            f"per page: {chunks_per_page} | "
            f"pages with chunks: {indexed_pages} | "
            f"pages without text: {empty_units or 'none'}"
        )


        # ----------------------------------------------------
        # Delete previous uploaded documents
        #
        # Only one document is kept at a time, and chunks left
        # over from earlier uploads are removed as well so a new
        # document never answers with the previous one.
        # ----------------------------------------------------

        clear_indexed_documents()


        # ----------------------------------------------------
        # Create embeddings and store chunks
        # ----------------------------------------------------

        store_chunks(chunks)

        print(
            f"[upload] embedded + stored: {time.time()-t1:.1f}s"
        )

        # The write is verified against the collection, so a
        # partially indexed document is reported instead of
        # looking ready.
        stored_count = collection.count_documents({
            DOCUMENT_ID_FIELD: document_id
        })

        print(
            f"[ingest] vectors inserted: {stored_count} "
            f"of {len(chunks)} chunk(s)"
        )

        if stored_count != len(chunks):

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Only {stored_count} of {len(chunks)} chunks "
                    "were indexed. Please upload the document again."
                )
            )


        # ----------------------------------------------------
        # Set current document
        # ----------------------------------------------------

        current_document_id = document_id

        current_filename = file.filename

        current_file_type = extension.lstrip(".")

        current_page_count = unit_count

        current_unit = unit


        # ----------------------------------------------------
        # Clear previous conversation
        # ----------------------------------------------------

        chat_history.clear()


        return {
            "status": "success",
            "filename": file.filename,
            "file_type": extension.lstrip("."),
            "pages": unit_count,
            "unit": unit,
            "chunks": len(chunks),
            "indexed_chunks": stored_count,
            "indexed_pages": len(indexed_pages),
            "empty_pages": empty_units,
            "document_id": document_id,
            "title": derive_file_title(file.filename)
        }


    except HTTPException:
        raise


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {redact(exc)}"
        )


    finally:

        # ----------------------------------------------------
        # Delete temporary file
        # ----------------------------------------------------

        if temp_path.exists():

            temp_path.unlink()


# ============================================================
# GET CURRENT DOCUMENT
# ============================================================

@app.get("/document")
async def document_info():

    return {
        "filename": current_filename,
        "document_id": current_document_id,
        "file_type": current_file_type,
        "pages": current_page_count,
        "unit": current_unit
    }


# ============================================================
# DELETE CURRENT DOCUMENT
# ============================================================

@app.delete("/document")
async def delete_document():

    global current_document_id
    global current_filename
    global current_file_type
    global current_page_count
    global chat_history

    try:
        # Everything this app ever indexed is removed, including
        # chunks stored by earlier versions of the app.
        clear_indexed_documents()

        current_document_id = None
        current_filename = None
        current_file_type = None
        current_page_count = None
        chat_history.clear()

        return {
            "status": "success",
            "message": "Document deleted successfully."
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {exc}"
        )


# ============================================================
# RETRIEVAL
# ============================================================
# Three things made the old retrieval answer from the wrong part
# of the document:
#
#   1. $vectorSearch was not filtered by document id, so every
#      search also returned chunks of documents uploaded earlier.
#   2. The keyword fallback ran a $regex over the whole
#      collection with no document filter at all.
#   3. A document-level question such as "give me an overview"
#      was answered from the same top similarity matches as any
#      other question, which are usually all from the first
#      pages.
#
# The helpers below fix each one.
# ============================================================

# Candidate pool for vector search. Kept deliberately modest: the
# pool only gets better once it is restricted to one document.
VECTOR_SEARCH_K = 20

# Chunks handed to the model for a specific question.
PASSAGE_CHUNK_BUDGET = 8

# Chunks sampled across the document for a document-level
# question. A chunk is ~500 characters, so this stays inside the
# 8192 token window once the prompt and history are added.
OVERVIEW_CHUNK_BUDGET = 16

# A document bigger than this is summarised part by part and the
# parts are combined, so the model never has to hold the whole
# document at once.
MAP_REDUCE_CHUNK_LIMIT = 48

# How many parts a large document is split into.
DOCUMENT_SUMMARY_PARTS = 4

# Hard character ceiling for the assembled context.
CONTEXT_CHAR_BUDGET = 12000

# Keywords scanned when vector search finds nothing useful.
KEYWORD_FALLBACK_LIMIT = 6

# Atlas reports a cosine score between 0 and 1. Anything below
# this is not really about the question, so it is only used when
# nothing better was found.
MIN_RELEVANT_SCORE = 0.55


# Questions that are about the document as a whole rather than
# about one passage.
DOCUMENT_LEVEL_PATTERNS = (
    r"\boverview\b",
    r"\bsummar",
    r"\bkey\s*points?\b",
    r"\bmain\s*points?\b",
    r"\btakeaways?\b",
    r"\bhighlights?\b",
    r"\bin\s+short\b",
    r"\bin\s+brief\b",
    r"\bin\s+summary\b",
    r"\bwhole\s+document\b",
    r"\bentire\s+(document|pdf|file|report)\b",
    r"\ball\s+(the\s+)?(pages?|sections?)\b",
    r"\bwhat\s+is\s+this\s+(document|pdf|file)\b",
    r"\bwhat\s+is\s+this\s+document\s+about\b",
    r"\bwhat\s+does\s+this\s+(document|pdf|file)\b",
    r"\btell\s+me\s+about\s+this\b",
    r"\bcontents?\b",
    r"\btable\s+of\s+contents\b",
    r"\bstructure\b",
    r"\bpurpose\s+of\s+this\b"
)

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an",
    "who", "where", "when", "why", "how",
    "does", "do", "can", "could", "tell",
    "me", "about", "of", "for", "to",
    "in", "on", "and", "with", "this",
    "that", "from", "give", "please",
    "document", "pdf", "file", "uploaded",
    "short", "brief", "summary", "point",
    "points", "key", "main"
}


def is_document_level(question):
    """True when the question is about the document as a whole."""
    text = question.lower()

    return any(
        re.search(pattern, text)
        for pattern in DOCUMENT_LEVEL_PATTERNS
    )


def page_in_question(question):
    """The page a question explicitly asks about, if any.

    "What is the definition on page 3?" should read page 3, so the
    number is returned and that page is searched first. Questions
    that do not name a page return None and are answered with
    normal semantic search across the whole document.
    """
    text = question.lower()

    match = re.search(
        r"\b(?:page|pages|pg|p)\s*\.?\s*(\d{1,3})\b",
        text
    )

    if not match:
        return None

    page = int(match.group(1))

    if page < 1:
        return None

    return page


def expand_follow_up(question):
    """Give a referential follow-up the context it needs.

    "Explain that in simple words" cannot be embedded on its own,
    so the previous question is added and the same part of the
    document is found again.

    This is only done for questions that actually point back at
    something. A short but self contained question such as
    "What is cloud?" must not have the previous question mixed
    into it, because that pulls unrelated passages into the search
    and is what made specific questions return the wrong text.
    """
    if not is_referential(question):
        return question

    previous = ""

    for message in reversed(chat_history):

        # The history holds LangChain message objects, which are
        # not dictionaries and have no .get(). A plain dictionary
        # entry is still accepted so the shape does not matter.
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content", "")

        else:
            role = getattr(message, "type", None)
            content = getattr(message, "content", "")

        if role in ("user", "human"):
            previous = content
            break

    if not previous:
        return question

    return f"{question} {previous}"


def is_referential(question):
    """True when the question points back at earlier turns.

    A question only counts as a follow-up when it is short and
    contains a word that refers to something already discussed,
    or when it is an ellipsis such as "and the second one".
    """
    text = question.lower().strip()

    if len(text.split()) > 12:
        return False

    markers = (
        r"\bthat\b", r"\bthis\b", r"\bthese\b", r"\bthose\b",
        r"\bit\b", r"\bthey\b", r"\bthem\b", r"\bhe\b", r"\bshe\b",
        r"\babove\b", r"\bprevious", r"\bearlier\b", r"\bsame\b",
        r"\bthe (first|second|third|fourth|fifth|last|next|former|latter)\b",
        r"^\s*(and|also|what about|how about)\b",
        r"\bin (simple|short|plain) (words|terms|language)\b",
        r"^\s*(more|why|how)\b"
    )

    return any(
        re.search(pattern, text)
        for pattern in markers
    )


def spread(items, limit):
    """Take `limit` items spread evenly across the whole list.

    Taking the first N chunks would only ever cover the start of
    the document, so the sample is distributed over the full
    range instead. Lists shorter than the limit are kept whole.
    """
    total = len(items)

    if total <= limit:
        return list(items)

    if limit <= 1:
        return [items[0]]

    step = (total - 1) / (limit - 1)

    return [
        items[round(position * step)]
        for position in range(limit)
    ]


def stored_chunks():
    """Every stored chunk of the current document, in reading order.

    Reading the chunks back in order is what makes a
    whole-document answer possible: the vector index only returns
    the most similar matches, which for a broad question are
    almost always the first pages.
    """
    cursor = collection.find(
        {
            DOCUMENT_ID_FIELD: current_document_id
        },
        {
            "text": 1,
            PAGE_FIELD: 1,
            CHUNK_INDEX_FIELD: 1
        }
    ).sort([
        (PAGE_FIELD, 1),
        (CHUNK_INDEX_FIELD, 1)
    ])

    return [
        read_chunk(document) for document in cursor
    ]


def search_queries(question):
    """Build the queries used to search the document.

    A short question such as "What is cloud?" is a poor search
    string on its own: the embedding of three words sits far away
    from a sentence that defines the term. A second, intent based
    query is therefore added for short and definitional questions,
    which pulls the explaining passage into the candidate pool.

    The extra queries are generic. Nothing about any particular
    subject is built in, so this works for any document.
    """
    queries = [question]

    text = question.lower().strip()
    words = text.split()

    definitional = bool(re.match(
        r"^\s*(what|who|when|where|why|how)\b",
        text
    )) or bool(re.search(
        r"\b(define|definition|meaning|means|explain|"
        r"tell me about|what is meant)\b",
        text
    ))

    if definitional or len(words) <= 5:

        subject = re.sub(
            r"^\s*(what|who|when|where|why|how)\s+"
            r"(is|are|was|were|does|do|did|can you|could you|"
            r"please)\s+",
            "",
            text
        )

        subject = re.sub(
            r"^\s*(define|definition of|meaning of|means|"
            r"explain|tell me about)\s+",
            "",
            subject
        )

        subject = subject.strip(" ?.!,")

        if len(subject.split()) >= 1:

            queries.append(
                f"{subject} definition meaning explained"
            )

    return queries


def merge_hits(*hit_lists):
    """Merge several result lists, keeping the best score per chunk."""
    best = {}
    order = []

    for hits in hit_lists:

        for doc, score in hits:

            key = (
                doc.metadata.get(CHUNK_INDEX_FIELD),
                doc.page_content[:80]
            )

            if key not in best:
                order.append(key)

            if score > best.get(key, (0, None))[0]:
                best[key] = (score, doc)

    return [
        (best[key][1], best[key][0])
        for key in order
    ]


def log_hits(question, hits, fallback=False):
    """Print exactly what the search returned for a question.

    This is the layer that decides which text the model sees, so
    it prints the question, then every retrieved chunk with its
    page and score. A wrong answer can then be traced either to
    retrieval (wrong chunks) or to the prompt (right chunks, bad
    answer) without guessing.
    """
    print(
        f"\n[retrieval] {'UNFILTERED ' if fallback else ''}"
        f"USER QUESTION: {question}"
    )

    if not hits:
        print("[retrieval] RETRIEVED CHUNKS: none")
        return

    for position, (doc, score) in enumerate(hits[:6], start=1):

        preview = " ".join(
            doc.page_content.split()
        )[:160]

        print(
            f"[retrieval]   CHUNK {position} | "
            f"PAGE: {doc.metadata.get(PAGE_FIELD)} | "
            f"SCORE: {score:.3f} | {preview}"
        )


def vector_search(question, k=VECTOR_SEARCH_K):
    """Vector search restricted to the uploaded document.

    The pre_filter relies on the filter field added by
    ensure_vector_index(). If the index is still being rebuilt the
    search falls back to unfiltered results rather than failing,
    and the post filter then keeps only this document's chunks.
    """
    try:
        hits = vector_store.similarity_search_with_score(
            question,
            k=k,
            pre_filter={
                DOCUMENT_ID_FIELD: current_document_id
            }
        )

        log_hits(question, hits)

        return hits

    except Exception as exc:
        print(f"[chat] filtered vector search failed: {exc}")

    results = []

    try:
        results = vector_store.similarity_search_with_score(
            question,
            k=k
        )

        log_hits(question, results, fallback=True)

    except Exception as exc:
        print(f"[chat] vector search error: {exc}")

    try:
        allowed = {
            chunk["id"] for chunk in stored_chunks()
        }

    except Exception as exc:
        print(f"[chat] could not read stored chunks: {exc}")
        return []

    return [
        (doc, score)
        for doc, score in results
        if getattr(doc, "id", None) in allowed
    ]


def keyword_search(question):
    """Exact-word lookup, limited to the uploaded document.

    Used only when vector search did not return enough, because a
    $regex scan matches any chunk that shares a word with the
    question, including irrelevant ones.
    """
    words = [
        word.strip(".,!?;:()[]{}\"'")
        for word in question.lower().split()
    ]

    keywords = [
        word
        for word in words
        if len(word) >= 3 and word not in STOP_WORDS
    ]

    found = []

    for keyword in keywords[:KEYWORD_FALLBACK_LIMIT]:
        try:
            cursor = collection.find(
                {
                    DOCUMENT_ID_FIELD: current_document_id,
                    "text": {
                        "$regex": re.escape(keyword),
                        "$options": "i"
                    }
                },
                {
                    "text": 1,
                    PAGE_FIELD: 1
                }
            ).limit(2)

            for item in cursor:
                chunk = read_chunk(item)

                found.append((chunk["text"], chunk["page"]))

        except Exception as exc:
            print(f"[chat] keyword search error: {exc}")

    return found


def select_passages(question):
    """Choose the passages that answer a specific question."""
    selected = []
    seen = set()

    def add(text, page):
        cleaned = (text or "").strip()

        if not cleaned or cleaned in seen:
            return False

        seen.add(cleaned)
        selected.append({
            "text": cleaned,
            "page": page
        })

        return True

    # A question that names a page is answered from that page, so
    # "the definition on page 3" cannot be satisfied by a similar
    # looking passage on page 2.
    asked_page = page_in_question(question)

    if asked_page is not None:

        print(
            f"[retrieval] question names page "
            f"{asked_page}, reading that page first"
        )

        for chunk in stored_chunks():

            if chunk["page"] != asked_page:
                continue

            add(chunk["text"], chunk["page"])

            if len(selected) >= PASSAGE_CHUNK_BUDGET:
                return selected

    hits = merge_hits(
        *[
            vector_search(query)
            for query in search_queries(
                expand_follow_up(question)
            )
        ]
    )

    # Rank by score, so the closest match is always first and the
    # prompt reads the best passage before the weaker ones.
    hits = sorted(
        hits,
        key=lambda hit: hit[1],
        reverse=True
    )

    log_hits(question, hits)

    # A hit below the relevance floor is kept only if nothing
    # better turned up, so one weak match cannot become the whole
    # context.
    strong = [
        (doc, score) for doc, score in hits
        if score >= MIN_RELEVANT_SCORE
    ]

    if strong:
        hits = strong

    for doc, _score in hits:
        add(doc.page_content, doc.metadata.get(PAGE_FIELD))

        if len(selected) >= PASSAGE_CHUNK_BUDGET:
            return selected

    # The page the question asked for is still missing, so its
    # chunks are added even when the search ranked them low.
    if (
        asked_page is not None
        and all(
            passage["page"] != asked_page
            for passage in selected
        )
    ):

        for chunk in stored_chunks():

            if chunk["page"] != asked_page:
                continue

            add(chunk["text"], chunk["page"])

            if len(selected) >= PASSAGE_CHUNK_BUDGET:
                break

    # Not enough similarity matches, so the chunks sitting next to
    # the best match are added. A question about one page is
    # usually split across neighbouring chunks, and without them
    # the same few passages get returned every time.
    if len(selected) < PASSAGE_CHUNK_BUDGET and selected:
        anchor_page = selected[0]["page"]

        for chunk in stored_chunks():
            if len(selected) >= PASSAGE_CHUNK_BUDGET:
                break

            if chunk["page"] != anchor_page:
                continue

            add(chunk["text"], chunk["page"])

    if len(selected) < PASSAGE_CHUNK_BUDGET:
        for text, page in keyword_search(question):
            if len(selected) >= PASSAGE_CHUNK_BUDGET:
                break

            add(text, page)

    if not selected:
        # Atlas indexes new vectors asynchronously, so a question
        # asked right after an upload can briefly return nothing.
        # A spread sample of the document is better than an empty
        # context.
        for chunk in spread(
            stored_chunks(),
            PASSAGE_CHUNK_BUDGET
        ):
            add(chunk["text"], chunk["page"])

    return selected



def select_overview_passages():
    """Cover the whole document instead of the first few pages.

    Chunks are spread evenly over the document in reading order,
    so an overview is built from the first page, the middle and
    the last page rather than from one section.
    """
    passages = []
    seen = set()

    for chunk in spread(
        stored_chunks(),
        OVERVIEW_CHUNK_BUDGET
    ):
        cleaned = (chunk["text"] or "").strip()

        if not cleaned or cleaned in seen:
            continue

        seen.add(cleaned)

        passages.append({
            "text": cleaned,
            "page": chunk["page"]
        })

    return passages


def render_context(passages):
    """Join passages into a context block, keeping page numbers."""
    parts = []
    used = 0

    for position, passage in enumerate(passages, start=1):
        text = passage["text"]
        page = passage.get("page")

        label = f"[Passage {position}"

        if isinstance(page, int):
            label += f" - Page {page}"

        label += "]"

        block = f"{label}\n{text}"

        if used + len(block) > CONTEXT_CHAR_BUDGET:
            remaining = CONTEXT_CHAR_BUDGET - used

            if remaining > 400:
                parts.append(block[:remaining])
                used = CONTEXT_CHAR_BUDGET

            break

        parts.append(block)
        used += len(block) + 2

    return "\n\n".join(parts)


def summarise_part(text):
    """Summarise one slice of a large document."""
    try:
        result = llm.invoke(
            section_prompt.invoke({"context": text})
        )

        return (result.content or "").strip()

    except Exception as exc:
        print(f"[chat] part summary failed: {redact(exc)}")

        return ""


def summarise_large_document():
    """Summarise a large document part by part.

    Returns one passage per part, or None when the document is
    small enough to be sampled in a single pass. Each part is
    written up separately and the notes are combined afterwards,
    so a document bigger than the context window is still answered
    from all of its parts.
    """
    chunks = stored_chunks()

    if len(chunks) <= MAP_REDUCE_CHUNK_LIMIT:
        return None

    print(
        f"[chat] document has {len(chunks)} chunks, "
        "summarising it in parts"
    )

    groups = spread(
        list(enumerate(chunks)),
        DOCUMENT_SUMMARY_PARTS
    )

    notes = []
    covered_pages = set()

    for position, group in enumerate(groups, start=1):

        start = group[0][0]
        end = group[-1][0] + 1

        part_passages = []

        for _index, chunk in chunks[start:end]:
            page = chunk["page"]

            if page is not None:
                covered_pages.add(page)

            part_passages.append({
                "text": chunk["text"],
                "page": page
            })

        part_text = render_context(part_passages)

        if not part_text.strip():
            continue

        note = summarise_part(part_text)

        if note:
            notes.append({
                "text": note,
                "page": None
            })

    print(
        "[chat] summarised "
        f"{len(notes)} part(s) covering pages "
        f"{sorted(covered_pages)}"
    )

    return notes or None


def build_history():
    """Recent turns, trimmed so they cannot crowd out the context."""
    turns = []

    for message in chat_history[-6:]:
        role = (
            "User"
            if isinstance(message, HumanMessage)
            else "DocPilot AI"
        )

        content = (message.content or "").strip()

        if not content:
            continue

        if len(content) > 400:
            content = content[:400] + "..."

        turns.append(f"{role}: {content}")

    return "\n".join(turns)


# ============================================================
# CONVERSATION TITLE
# ============================================================
# A conversation used to be listed under the first question the
# user typed, so the history read as a list of questions
# ("What are the eligibility criteria?"). The name is now a
# short title about the subject of the conversation, taken from
# the uploaded file and the conversation itself, and nothing
# about any particular file is hard coded.
# ============================================================

# Words that describe the file rather than its subject. They are
# dropped so "Worksbot Company Internship Hiring.pdf" is named
# "Worksbot Internship Hiring" instead of repeating the file
# name back.
TITLE_NOISE_WORDS = {
    "doc", "docs", "document", "documents", "file", "files",
    "final", "copy", "draft", "report", "reports", "note",
    "notes", "slides", "presentation", "pdf", "docx", "txt",
    "page", "pages", "chapter", "appendix", "part", "section",
    "company", "organisation", "organization", "group",
    "new", "old", "updated", "revised", "revision", "version",
    "the", "a", "an", "and", "or", "of", "for", "to", "in",
    "on", "at", "with", "from", "by"
}

# Grammatical filler. Only dropped when a file name is built
# entirely out of noise words, so an ordinary name keeps its
# original words.
TITLE_FILLER_WORDS = {
    "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "as", "all", "any",
    "some", "no", "not", "we", "you", "they", "them", "their",
    "our", "your", "my", "he", "she", "his", "her", "can",
    "could", "should", "would", "will", "shall", "do", "does",
    "did", "done", "have", "has", "had", "if", "then", "than",
    "so", "but", "also", "just", "more", "most", "other",
    "into", "over", "under", "per", "via", "about", "up", "out"
}

# A conversation name is a label, not a sentence.
TITLE_MAX_WORDS = 4
TITLE_MAX_CHARS = 46

# Used only when a conversation has no document to be named
# after, because answers are always about an uploaded file.
TITLE_STOP_WORDS = {
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will", "shall", "please", "tell", "give",
    "show", "list", "explain", "describe", "summarise", "summarize",
    "me", "my", "you", "your", "we", "our", "about", "this", "that",
    "these", "those", "there", "document", "pdf", "file", "uploaded",
    "the", "and", "for", "from", "with", "have", "has", "had",
    "know", "need", "want", "many", "much", "any", "some"
}


def title_words(text):
    return re.findall(r"[A-Za-z][A-Za-z0-9'&]*", text or "")


def normalise(text):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        (text or "").lower()
    ).strip()


def read_title_words(words):
    """Join the words of a title, tidied up for display.

    Capitalising only the longer words keeps initials such as
    "PR" or "ML" as they were written while a plain lower case
    file name still reads like a title.
    """
    return " ".join(
        (
            word[:1].upper() + word[1:]
            if len(word) >= 3
            else word
        )
        for word in words
    )


def derive_file_title(filename):
    """Short topic title taken from an uploaded file name.

    The extension, the numbers and the words that only describe
    the file are removed, so any document ends up with a short
    readable name without anything being written for one
    particular file.
    """
    base = re.sub(
        r"\.[A-Za-z0-9]+$",
        "",
        (filename or "").strip()
    )

    words = [
        word for word in title_words(base)
        if not word.isdigit() and not re.fullmatch(
            r"v\d+", word, flags=re.IGNORECASE
        )
    ]

    meaningful = [
        word for word in words
        if word.lower() not in TITLE_NOISE_WORDS
    ]

    if not meaningful:
        meaningful = [
            word for word in words
            if word.lower() not in TITLE_FILLER_WORDS
        ]

    if not meaningful:
        meaningful = words

    if not meaningful:
        return ""

    return read_title_words(
        meaningful[:TITLE_MAX_WORDS]
    )


def derive_question_title(question):
    """Topic title taken from a question alone."""
    words = [
        word for word in title_words((question or "").lower())
        if word.lower() not in TITLE_STOP_WORDS
        and not word.isdigit()
    ]

    if len(words) < 2:
        return ""

    return read_title_words(words[:3])


def clean_title(raw, fallback="", question=""):
    """Normalise a generated name and reject anything unusable.

    The model is asked for a name only, but a small model still
    replies with quotes, a trailing full stop, a "Title:" prefix
    or the whole question now and then. Anything that is not a
    short readable name is dropped in favour of the document
    topic, so a conversation always ends up with a name.
    """
    text = ""

    # A question is not a name, so a model that repeats the
    # question back is treated as having given no name. This is
    # read from the raw reply, before the surrounding punctuation
    # is cleaned off.
    asked_back = "?" in (raw or "")

    for line in (raw or "").splitlines():

        candidate = re.sub(
            r"^[^A-Za-z0-9]+",
            "",
            line.strip()
        )

        candidate = re.sub(
            r"[^A-Za-z0-9]+$",
            "",
            candidate
        )

        if candidate:
            text = candidate
            break

    text = re.sub(
        r"^(title|name|topic)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # A question is not a name, so a model that repeats the
    # question back is treated as having given no name. This is
    # checked before the trailing question mark is cleaned off.
    if asked_back:
        text = ""
    # A file extension and a trailing full stop are never part of
    # a conversation name.
    text = re.sub(
        r"\.[A-Za-z0-9]{1,5}$",
        "",
        text
    )

    text = re.sub(
        r"[.,;:!?]+$",
        "",
        text
    ).strip()

    if asked_back:
        text = ""

    # "Internship Document" is the topic with a word that says
    # nothing, so that word is dropped instead of the name.
    text = re.sub(
        r"\s+(document|documents|pdf|docx|file|files|"
        r"question|questions)$",
        "",
        text,
        flags=re.IGNORECASE
    )

    if text:
        text = " ".join(
            text.split()[:TITLE_MAX_WORDS + 1]
        )

    if text and len(text) > TITLE_MAX_CHARS:
        text = " ".join(
            text.split()[:TITLE_MAX_WORDS]
        )

    if text and len(text) > TITLE_MAX_CHARS:
        text = ""

    if text and question:
        if normalise(text) == normalise(question):
            text = ""

    return text or fallback


def document_excerpt(limit=400):
    """The opening of the uploaded document, used for naming.

    A single chunk is read straight from the collection. This is
    only extra material for the name, so retrieval and answer
    generation are not involved.
    """
    if not current_document_id:
        return ""

    document = collection.find_one(
        {DOCUMENT_ID_FIELD: current_document_id},
        {"text": 1}
    )

    if not document:
        return ""

    return " ".join(
        (document.get("text") or "").split()
    )[:limit]


def generate_conversation_title(message):
    """A short, topic based name for a conversation.

    The file name and the opening of the document give the
    subject, and the first question shows what the conversation
    is actually about. This is a small separate call whose only
    job is the name shown in the history list, so answer
    generation is left alone.
    """
    question = (message or "").strip()

    subject = derive_file_title(current_filename)

    fallback = (
        subject
        or derive_question_title(question)
    )

    if not subject and not question:
        return fallback

    try:
        excerpt = document_excerpt()

    except Exception as exc:
        print(f"[title] could not read the document: {exc}")
        excerpt = ""

    generated = ""

    try:
        result = llm.invoke(
            title_prompt.invoke({
                "document": (
                    subject
                    or "no document uploaded"
                ),
                "excerpt": (
                    excerpt
                    or "no text available"
                ),
                "question": (
                    question
                    or "the user has not asked anything yet"
                )
            })
        )

        generated = (result.content or "").strip()

    except Exception as exc:
        print(
            f"[title] could not name the conversation: "
            f"{redact(exc)}"
        )

    title = clean_title(
        generated,
        fallback,
        question
    )

    print(
        f"[title] subject: {subject or '—'} | "
        f"question: {question or '—'} | "
        f"model: {generated or '—'} | title: {title or '—'}"
    )

    return title


# ============================================================
# CHAT / RAG
# ============================================================

NOT_FOUND_ANSWER = (
    "I couldn't find this information in the uploaded document."
)


def has_answer(text):
    """True when the model actually produced an answer.

    An empty or near empty reply must never reach the user as a
    blank bubble, so the fallback message is sent instead.
    """
    cleaned = (text or "").strip()

    if len(cleaned) < 2:
        return False

    # A reply that only says it cannot find something counts as an
    # answer, because that is the correct grounded response.
    return True


@app.post("/chat")
async def chat(request: ChatRequest):
    global chat_history

    question = request.message.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # Answered with a clear message before any retrieval is done, so
    # a missing key never leaves the user looking at an empty bubble.
    require_gemini()

    if not current_document_id:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF before asking a question."
        )

    document_level = is_document_level(question)

    if document_level:

        # A document that does not fit the context window is
        # written up part by part first.
        part_notes = summarise_large_document()

        if part_notes:
            passages = part_notes
            active_prompt = combine_prompt
        else:
            passages = select_overview_passages()
            active_prompt = document_prompt

        prompt_values = {
            "context": render_context(passages),
            "question": question
        }
    else:
        passages = select_passages(question)
        active_prompt = prompt
        prompt_values = {
            "history": build_history(),
            "context": render_context(passages),
            "question": question
        }

    context = prompt_values["context"]

    # Nothing was retrieved for this question, so the model is not
    # called at all and the user is told the document does not
    # cover it.
    if not context.strip():

        async def empty_context():

            yield (
                "data: "
                + json.dumps({
                    'type': 'context',
                    'pages': [],
                    'document_level': document_level
                })
                + "\n\n"
            )

            yield (
                f"data: "
                f"{json.dumps({'token': NOT_FOUND_ANSWER})}"
                f"\n\n"
            )

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            empty_context(),
            media_type="text/event-stream"
        )

    pages_used = sorted({
        passage["page"]
        for passage in passages
        if isinstance(passage.get("page"), int)
    })

    messages = active_prompt.invoke(prompt_values)

    async def generate():
        answer = ""

        try:
            # Sent first so the interface can report what the
            # answer was built from.
            yield (
                "data: "
                + json.dumps({
                    'type': 'context',
                    'pages': pages_used,
                    'document_level': document_level
                })
                + "\n\n"
            )

            async for chunk in llm.astream(messages):
                if chunk.content:
                    answer += chunk.content

                    yield (
                        f"data: "
                        f"{json.dumps({'token': chunk.content})}"
                        f"\n\n"
                    )

            # The model produced nothing usable, so the grounded
            # fallback is streamed instead of leaving an empty
            # bubble in the conversation.
            if not has_answer(answer):

                print(
                    "[chat] model returned no usable answer, "
                    "sending the not found message"
                )

                answer = NOT_FOUND_ANSWER

                yield (
                    f"data: "
                    f"{json.dumps({'token': answer})}"
                    f"\n\n"
                )

            chat_history.append(
                HumanMessage(content=question)
            )

            chat_history.append(
                AIMessage(content=answer)
            )

            yield "data: [DONE]\n\n"

        except GeminiError as exc:

            # The Gemini adapter already redacts the key and turns a
            # raw SDK failure into something actionable, and the
            # frontend renders this event exactly as before.
            print("[chat] gemini error:", redact(exc))

            yield (
                f"data: "
                f"{json.dumps({'error': redact(exc)})}"
                f"\n\n"
            )

            yield "data: [DONE]\n\n"

        except Exception as exc:

            # redact() is applied here too, so an unexpected failure
            # can never send a credential to the browser.
            print("LLM error:", redact(exc))

            yield (
                f"data: "
                f"{json.dumps({'error': redact(exc)})}"
                f"\n\n"
            )

            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.delete("/history")
async def clear_history():

    chat_history.clear()

    return {
        "status": "ok"
    }


# ============================================================
# CONVERSATION TITLE
# ============================================================

@app.post("/conversation-title")
def conversation_title(request: TitleRequest):
    """A short name for a conversation, based on its subject.

    Declared as a plain def endpoint so FastAPI runs it in its
    worker thread pool: the extra model call must never block the
    chat stream.
    """
    # The name is a nicety, so a missing key or a failed call falls
    # back to the name built from the file and the question instead
    # of failing the request.
    if not is_configured():
        return {
            "title": derive_file_title(current_filename)
        }

    return {
        "title": generate_conversation_title(request.message)
    }