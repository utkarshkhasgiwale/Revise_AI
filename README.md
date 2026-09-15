# Revise AI

**Live Demo:** https://revise-ai-zcxc.onrender.com/

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