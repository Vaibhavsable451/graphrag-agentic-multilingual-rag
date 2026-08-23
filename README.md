# graphrag-agentic-multilingual-rag
# Agentic Multi-RAG Platform

An agentic AI system that combines three RAG strategies behind one LangGraph
orchestrator, exposes itself as an MCP tool server, and ships with a React
chat + knowledge-graph frontend.

## What makes it "agentic"

Instead of one fixed retrieve → generate pipeline, a LangGraph `StateGraph`
decides what to do at each step:

```
detect_language → route → retrieve → grade → (retry retrieve | generate) → localize
```

- **route**: an LLM call classifies the question as needing plain vector
  search, graph traversal, or both.
- **grade**: checks whether retrieval actually returned usable context; if
  not, it loops back and widens the search (up to 2 retries) instead of
  answering blind.
- **localize**: translates the final answer back into the question's
  original language.

## The three RAG strategies

| Strategy | File | What it does |
|---|---|---|
| **Agentic RAG** | `backend/app/agents/graph_workflow.py` | The routing/grading/retry loop above — the agent chooses its own retrieval path. |
| **GraphRAG** | `backend/app/rag/graph_rag.py` | Extracts entities/relations from ingested text into a `networkx` graph, answers multi-hop "how is X connected to Y" questions by traversal, and pushes entity nodes into Pinecone under a `graph` namespace so they're also semantically searchable. |
| **Multilingual RAG** | `backend/app/rag/multilingual_rag.py` | Detects the query language, translates to English for retrieval when needed, and translates the answer back — backed by a multilingual sentence-transformer embedding model shared across languages. |

## MCP integration

`backend/app/mcp/mcp_server.py` exposes the same pipeline as MCP tools
(`ask_rag`, `ingest_text`, `get_knowledge_graph`), so any MCP client — Claude
Desktop, another agent, etc. — can call this project directly, not just the
React frontend.

```bash
python -m app.mcp.mcp_server
```

## Multi-provider LLMs

`backend/app/core/llm_router.py` tries Groq first (fast + cheap), then falls
back to OpenRouter, then Gemini — so a single missing/rate-limited key
doesn't take the whole agent down. Swap the default with
`PRIMARY_LLM_PROVIDER` in `.env`.

## Stack

- **Backend**: FastAPI, LangChain, LangGraph, Pinecone, sentence-transformers, MCP SDK
- **Frontend**: React (Vite), react-force-graph-2d for the knowledge-graph view
- **LLMs**: Groq, OpenRouter, Gemini (auto-failover)
- **Vector DB**: Pinecone (serverless)

## Project structure

```
backend/
  app/
    core/          # llm_router.py, embeddings.py
    rag/            # graph_rag.py, multilingual_rag.py
    agents/          # graph_workflow.py (the LangGraph agent)
    mcp/             # mcp_server.py
    vectorstore/     # pinecone_client.py
    api/             # routes.py, schemas.py
    main.py
frontend/
  src/
    components/      # ChatWindow, MessageBubble, Sidebar, GraphView
    api/client.js
docker-compose.yml
```

## Setup

**Backend**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, PINECONE_API_KEY
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

Or run both with `docker compose up`.

## API

- `POST /api/chat` — `{ "question": "..." }` → answer + language + route + sources + graph facts
- `POST /api/ingest` — `{ "text": "...", "source": "..." }` → chunks into vector store + graph
- `GET /api/graph` — nodes/edges for the graph view
- `GET /api/health`

## Notes / what to harden before treating this as production

- Add auth on the API (currently CORS is wide open for local dev).
- Entity extraction in `graph_rag.py` is a single LLM call per document —
  fine for a portfolio demo, but batch/chunk it for larger documents.
- Add a proper document loader (PDF/DOCX chunking) ahead of `/ingest` for
  real files instead of raw pasted text.
- Persist the `networkx` graph (currently in-memory) to disk or a graph DB
  if you need it to survive a restart.
