# Josh's RAG Pipeline

A RAG (Retrieval-Augmented Generation) recommendation system for laptop recommendations powered by Josh's content.

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Start Docker services
docker-compose up -d

# 3. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 4. Setup database
python cli.py setup --all

# 5. Sync data
python cli.py sync --all

# 6. Ingest content and generate embeddings
python cli.py ingest --all

# 7. Create vector indexes
python cli.py setup --indexes

# 8. Start API
python cli.py api
```

API is now running at `http://localhost:8000`

---

## Documentation

- **[SETUP.md](SETUP.md)** - Complete setup guide with troubleshooting
- **[TECHNICAL.md](TECHNICAL.md)** - Architecture, API reference, and technical details
- **[RAG_IMPLEMENTATION_GUIDE.md](RAG_IMPLEMENTATION_GUIDE.md)** - Implementation strategy and philosophy

---

## Features

- ✅ **Natural Language Input** - Use free-text prompts or structured quiz (LLM intent extraction)
- ✅ **Semantic Search** - Vector-based content retrieval using e5-base-v2 embeddings
- ✅ **Multi-Source Content** - Unified search across blog articles, YouTube transcripts, and test data
- ✅ **Smart Ranking** - Josh's context (60%) + Spec matching (25%) + Test data (15%)
- ✅ **Performance Benchmarks** - Real test data from 229 laptop configurations
- ✅ **Spec Fallback** - Pure spec matching for low confidence queries
- ✅ **API Authentication** - Optional Bearer token authentication
- ✅ **Query Logging** - Monitor all recommendation requests
- ✅ **Real-Time Sync** - Webhooks (Sanity, YouTube) + Periodic polling (Products every 5 min)
- ✅ **Incremental Sync** - Cursor-based syncing (only fetch new/updated content)
- ✅ **Date Filtering** - Only use recent, relevant content

---

## API Usage

### Get Recommendations (Natural Language)

**NEW: Use natural language prompts!**

```bash
curl -X POST http://localhost:8000/recommend/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I need a laptop for college programming and gaming, budget around $1500",
    "top_k": 5
  }'
```

### Get Recommendations (Structured Quiz)

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "quiz_response": {
      "profession": ["student"],
      "use_case": ["programming"],
      "budget": ["value"],
      "portability": "light"
    },
    "top_k": 3
  }'
```

### With Authentication

```bash
# Set API_KEY in .env first
curl -X POST http://localhost:8000/recommend \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"quiz_response": {...}}'
```

---

## Project Structure

```
jj-rag-pipeline/
├── src/
│   ├── data_pipeline/      # Content extraction & embeddings
│   ├── rag/                # Retriever, ranker, fallback
│   └── api/                # FastAPI service (main production code)
├── scripts/                # Unified CLI tools (local operations)
│   ├── setup.py           # Database setup
│   ├── ingest.py          # Content ingestion
│   ├── sync.py            # External data sync
│   └── youtube.py         # YouTube operations
├── tests/                  # Evaluation suite
├── docs/                   # Production schema reference
├── SETUP.md               # Setup guide
├── TECHNICAL.md           # Technical documentation
└── RAG_IMPLEMENTATION_GUIDE.md  # Implementation strategy
```

---

## Testing

```bash
# Run full pipeline evaluation
uv run python tests/test_rag_evaluation.py

# Results saved to: tests/rag_evaluation_results.json
```

---

## Common Tasks

**Update content from Sanity:**
```bash
python cli.py sync --sanity --all
```

**Fetch YouTube transcripts:**
```bash
python cli.py youtube --fetch --count=20
python cli.py ingest --youtube --embeddings
```

**Update product configs:**
```bash
python cli.py sync --products
```

**Sync test data (performance benchmarks):**
```bash
python cli.py sync --test-data
```

**Start API server:**
```bash
python cli.py api              # With auto-reload
python cli.py api --no-reload  # Production mode
```

**Monitor queries:**
```bash
docker exec -it jj-rag-pipeline-postgres-1 psql -U postgres -d josh_rag
SELECT * FROM rag_query_logs ORDER BY created_at DESC LIMIT 10;
```

---

## Tech Stack

- **Python 3.12+** with uv package manager
- **FastAPI** for REST API
- **PostgreSQL 15** with pgvector for vector search
- **Redis** for caching
- **Sentence Transformers** (e5-base-v2) for embeddings
- **OpenAI GPT-3.5-turbo** for intent extraction
- **Docker** for local development
- **AWS S3** for test data storage

---

## Support

For detailed information, see:
- Setup instructions: [SETUP.md](SETUP.md)
- Technical details: [TECHNICAL.md](TECHNICAL.md)
- Implementation guide: [RAG_IMPLEMENTATION_GUIDE.md](RAG_IMPLEMENTATION_GUIDE.md)
