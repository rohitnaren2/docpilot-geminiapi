# RAG Chatbot using MongoDB Atlas, Ollama, LangChain, and FastAPI  [![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-db61a2?style=flat&logo=github-sponsors)](https://github.com/sponsors/santhoshh-maax) [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/N4N51YFMS9)

A Retrieval-Augmented Generation (RAG) chatbot built using **MongoDB Atlas Vector Search**, **Ollama**, **Gemma 3**, **LangChain**, and **FastAPI**. The application retrieves relevant information from a vector database and generates context-aware responses using a locally hosted Large Language Model (LLM).

The backend performs semantic search with MongoDB Atlas Vector Search and LangChain, while the frontend provides a modern web-based chat interface with real-time streaming responses.
---

# Features

- 📄 Load PDF documents
- ✂️ Split documents into chunks
- 🧠 Generate embeddings locally using Ollama
- 💾 Store embeddings in MongoDB Atlas
- 🔍 Perform semantic search with MongoDB Vector Search
- 🤖 Generate answers using Gemma 3
- 💬 **Web-based chatbot UI** with streaming responses
- 🌐 **FastAPI backend** with REST API
- 🆓 Completely free (No OpenAI API required)

---

# Running Modes

This project supports two ways of running the application.

## Option 1 — Local Installation

- Install Python
- Install Ollama
- Pull the required models manually
- Run FastAPI locally

Recommended for development.

---

## Option 2 — Docker (Recommended)

- No need to install Ollama on your computer.
- Docker automatically downloads and configures Ollama.
- Required models (`gemma3:4b` and `nomic-embed-text`) are downloaded automatically on first run.
- Everything runs inside Docker containers.

Recommended for first-time users and deployment.

---

# Tech Stack

- Python 3.11
- LangChain
- MongoDB Atlas + Vector Search
- Ollama (Gemma 3:4b + nomic-embed-text)
- FastAPI + Uvicorn
- HTML / CSS / JavaScript (vanilla)
- PyPDF

---

# Project Structure

```
RAG Application/
│
├── app.py                  # FastAPI web server (API + frontend)
├── rag.py                  # Original CLI chatbot
├── load_data.py            # PDF ingestion script
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env                    # MongoDB connection string
|--.env.template            #use as env file template
├── requirements.txt        # Python dependencies
│
├── docker/
│   └── ollama-init/
│       └── Dockerfile
│
├── scripts/
│   └── init-ollama.sh
│
├── static/
│     └── index.html        # Chatbot frontend UI
│
├── sample_files/
│     └── singpore_pr_details.pdf
│
└── README.md
```

---

# Choose Your Setup

| Setup | Ollama Installation | Python Installation | Recommended |
|---------|--------------------|---------------------|-------------|
| Local | ✅ Required          | ✅ Required        | For Development |
| Docker | ❌ Not Required     | ❌ Not Required    | ⭐ Recommended |

---

# Local Installation

- Python 3.11+
- MongoDB Atlas Account
- Ollama

---

# Step 1 - Install Ollama

Download and install Ollama

https://ollama.com/download

Verify Installation

```bash
ollama --version
```

---

## Ollama Configuration

The Ollama connection differs depending on how you run the project.

### Local Installation

Use the default Ollama configuration.

```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)
```

---

### Docker Installation

When running inside Docker, FastAPI communicates with the Ollama container through Docker's internal network.

Use:

```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama:11434"
)

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
    base_url="http://ollama:11434"
)
```

> **Important:** `localhost` refers to the current container. Therefore, Docker containers must communicate using the service name (`ollama`) instead of `localhost`.

---

# Step 2 - Download Gemma 3

```bash
ollama pull gemma3:4b
```

Verify

```bash
ollama list
```

Expected Output

```
gemma3:4b
```

---

# Step 3 - Download Embedding Model

```bash
ollama pull nomic-embed-text
```

Verify

```bash
ollama list
```

Expected Output

```
gemma3:4b
nomic-embed-text
```

---

# Step 4 - Install Python Packages

```bash
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-ollama langchain-mongodb pymongo pypdf fastapi uvicorn pydantic
```

---

# Step 5 - Create MongoDB Atlas Cluster

1. Create a MongoDB Atlas account.
2. Create an M0 Free Cluster.
3. Create a Database User.
4. Allow Network Access (`0.0.0.0/0`).
5. Copy the MongoDB Connection String.

Example

```
mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
```

---

# Step 6 - Create .env file

```python
MONGODB_URI = YOUR_MONGODB_CONNECTION_STRING
```

---

# Step 7 - Add PDF

Create a folder

```
sample_files/
```

Place your PDF inside.

Example

```
sample_files/
    singpore_pr_details.pdf
```

---

# Step 8 - Load Data into MongoDB

Run

```bash
python load_data.py
```

This will

- Read the PDF
- Remove empty pages
- Split into chunks
- Generate embeddings using `nomic-embed-text`
- Store vectors in MongoDB Atlas

Expected Output

```
Old documents deleted.
Total pages loaded.
Pages cleaned.
Chunks created.
Creating embeddings...
Data successfully stored.
```

---

# Step 9 - Create Vector Search Index

Open

```
MongoDB Atlas
```

Go to

```
Database
    ↓
Search
    ↓
Create Search Index
```

Choose

```
Bring your own embeddings
```

Index Name

```
vector_index
```

Database

```
singapore_pr_chunks
```

Collection

```
chunked_data
```

Vector Field

```
embedding
```

Dimensions

```
768
```

Similarity

```
Cosine
```

Wait until the index status becomes

```
READY
```

---

# Step 10 - Run the Web Application

Make sure Ollama is running in the background, then start the server:

```bash
python -m uvicorn app:app --port 8000
```

Open your browser and go to

```
http://localhost:8000
```

You will see a chatbot interface. Type your questions and get instant, streamed answers.

### API Endpoints

| Method | Endpoint       | Description                |
|--------|---------------|----------------------------|
| GET    | `/`           | Chatbot frontend UI        |
| POST   | `/chat`       | Send message (SSE stream)  |
| GET    | `/history`    | Get chat history           |
| DELETE | `/history`    | Clear chat history         |

---

# Step 11 - (Alternative) Run the CLI Version

```bash
python rag.py
```

Example

```
==============================
 Singapore PR RAG Assistant
==============================

You: Who is eligible for Singapore PR?
Assistant: A foreigner granted permanent residence status, allowing indefinite stay in Singapore.

You: exit
```

---

# Docker Installation (Recommended)

## Prerequisites

- Docker Desktop
- Docker Compose

Verify installation:

```bash
docker --version
docker compose version
```

---

## Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git

cd YOUR_REPO
```

---

## Create the `.env` File

```env
MONGODB_URI=YOUR_MONGODB_CONNECTION_STRING

DB_NAME=singapore_pr_chunks

COLLECTION_NAME=chunked_data

INDEX_NAME=vector_index
```

---

## Build and Run

```bash
docker compose up --build
```

On the first run Docker will:

- Download the Ollama Docker image.
- Build the FastAPI image.
- Build the Ollama initializer image.
- Automatically download:
  - `gemma3:4b`
  - `nomic-embed-text`
- Start the FastAPI application.

No manual Ollama installation is required.

---

Open:

```
http://localhost:8000
```

---

## Stop Containers

```bash
docker compose down
```

---

## Start Again

```bash
docker compose up
```

The downloaded models are stored in a Docker volume, so they will **not** be downloaded again.

---

## Remove Everything

```bash
docker compose down -v
```

This removes:

- Containers
- Networks
- Ollama models

The next run will download the models again.

---

# RAG Workflow

```
                PDF
                 │
                 ▼
        PyPDFLoader
                 │
                 ▼
      Recursive Chunking
                 │
                 ▼
      Ollama Embeddings
    (nomic-embed-text)
                 │
                 ▼
       MongoDB Atlas
      Vector Database
                 │
                 ▼
      Vector Search Index
                 │
                 ▼
       User Question
                 │
                 ▼
 Question Embedding
                 │
                 ▼
 Semantic Search
                 │
                 ▼
 Top Relevant Chunks
                 │
                 ▼
      Gemma 3 (LLM)
                 │
                 ▼
      Final Response
```

---

# Commands Summary

## Install Ollama

```bash
ollama --version
```

## Download Models

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

## View Installed Models

```bash
ollama list
```

## Load PDF

```bash
python load_data.py
```

## Run Web Chatbot

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

## Run CLI Chatbot

```bash
python rag.py
```

## Docker

Build and Run

```bash
docker compose up --build
```

Start Existing Containers

```bash
docker compose up
```

Stop Containers

```bash
docker compose down
```

Remove Containers and Models

```bash
docker compose down -v
```

---

# Future Improvements

- Multi-PDF Support
- PDF Upload via Web UI
- Chat History Persistence
- Source Citation in Responses
- User Authentication
- Multiple Collections
- Hybrid Search
- Metadata Filtering

---

# 👨‍💻 Author

**Santhosh P**

B.E. Computer Science and Engineering

Mount Zion College of Engineering and Technology

👉 **[Sponsor me on GitHub](https://github.com/sponsors/santhoshh-maax)**

---

# ⭐ Support

If you found this project useful, please consider giving this repository a **⭐ Star**.

---

# 📜 License

This project is developed for educational and learning purposes.
