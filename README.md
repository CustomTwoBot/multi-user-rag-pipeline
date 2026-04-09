#------------README---------------

# Multi-User RAG Pipeline

A local Retrieval-Augmented Generation (RAG) API built with FastAPI, ChromaDB, and Ollama. Users can submit personal profiles which are embedded and stored in a vector database, then ask natural language questions and receive context-grounded AI responses — all running locally with no cloud dependencies or API costs.

## Features
- Semantic search over stored documents using text embeddings
- Multi-user support with ChromaDB metadata filtering
- Optional per-user query filtering via URL parameter
- Fully local — no OpenAI key or cloud services required

## Tech Stack
- Python, FastAPI, ChromaDB, Ollama, Pydantic, Uvicorn

## How to Run

**1. Install dependencies**
pip install -r requirements.txt

**2. Start Ollama and pull models**
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text

**3. Run the API**
uvicorn main:app --reload

**4. Open Swagger UI**
http://127.0.0.1:8000/docs
