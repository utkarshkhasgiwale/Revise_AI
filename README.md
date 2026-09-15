# Revise AI

A RAG (Retrieval Augmented Generation) system for YouTube DSA lecture playlists. Ask questions and get answers with exact timestamps.

## Features

- **Hybrid Search**: Combines keyword (true corpus-wide BM25) and semantic (vector) search with score fusion
- **Conversation Memory**: Remembers context and resolves query references across turns
- **Timestamp Search (Show Me Where)**: Find exact moments in lectures instantly without LLM generation
- **Grounding**: Refuses to answer when information isn't in the lectures
- **Auto-indexing**: Detects and indexes new lectures from playlists automatically

## Quick Start

```bash
# Install dependencies
pip install qdrant-client groq sentence-transformers fastapi uvicorn pydantic python-dotenv typer rich yt-dlp faster-whisper

# Set up environment
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Run server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

## Usage

### Web Interface

Two modes available:
- **Ask**: Get AI-generated answers with grounded citations
- **Show Me Where**: See all moments a topic is mentioned (instant, no LLM)

### CLI

```bash
# Find timestamps for a topic
python -m ytrag features show-where "binary search" --k 6

# Evaluate retrieval methods against golden set
python -m ytrag eval --hybrid

# Detect new lectures in playlist
python -m ytrag features detect-new --playlist "PLAYLIST_URL"
```

## API Endpoints

- `POST /ask` - Ask a question with conversation memory and hybrid search
- `POST /show-where` - Get timestamps where topic is discussed
- `GET /health` - Service status
- `GET /stats` - Index statistics

## Tech Stack

- **Backend**: FastAPI
- **Vector DB**: Qdrant (local persistent or hosted cluster)
- **LLM**: Groq
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Search**: Independent Hybrid Retrieval (BM25 + Vector)

## Configuration

Environment variables in `.env`:

```bash
GROQ_API_KEY=your_key_here          # Required: Get from console.groq.com
YTRAG_TOP_K=6                       # Results per query
YTRAG_MAX_DISTANCE=0.6              # Coarse vector similarity pre-filter
YTRAG_BM25_WEIGHT=0.3               # Hybrid search: keyword weight
YTRAG_VECTOR_WEIGHT=0.7             # Hybrid search: semantic weight
```

## Project Structure

```
revise_ai/
├── ytrag/              # Core library (CLI package name)
│   ├── answer.py       # LLM answer generation & guards
│   ├── retrieval.py    # True independent hybrid search & BM25
│   ├── conversation.py # Context management & query rewriting
│   ├── index.py        # Qdrant client & chunk management
│   ├── detection.py    # New lecture monitor & auto-indexing
│   └── ...
├── api/
│   ├── main.py         # FastAPI server
│   └── static/         # Web UI
├── eval/               # Test datasets (golden sets)
└── transcripts/        # Cached YouTube transcripts
```
