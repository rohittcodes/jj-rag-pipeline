# Setup Guide

Complete setup guide for Josh's RAG Pipeline - a laptop recommendation system powered by semantic search.

---

## Prerequisites

- **Python 3.12+**
- **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
- **uv** - Python package manager: `pip install uv`

---

## Quick Start

### 1. Clone and Install

```bash
git clone <repository-url>
cd jj-rag-pipeline
uv sync
```

### 2. Start Docker Services

```bash
docker-compose up -d
```

This starts:
- PostgreSQL (port 5432) - Vector database with pgvector
- Redis (port 6379) - Caching layer

### 3. Configure Environment

```bash
cp .env.example .env
```

**Required variables:**
```bash
# Local Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=josh_rag
DB_USER=postgres
DB_PASSWORD=postgres

# Sanity CMS
SANITY_PROJECT_ID=your_project_id
SANITY_API_TOKEN=your_read_token
SANITY_API_VERSION=v2021-10-21
```

**Optional variables:**
```bash
# API Authentication (leave empty to disable)
API_KEY=

# RAG Configuration
RAG_CONFIDENCE_THRESHOLD=0.75
CONTENT_MAX_AGE_YEARS=1

# Production Database (for real product specs)
PROD_DB_HOST=your_rds_host.rds.amazonaws.com
PROD_DB_PORT=5432
PROD_DB_NAME=your_prod_db_name
PROD_DB_USER=postgres
PROD_DB_PASSWORD=your_prod_password
```

### 4. Initialize Database

```bash
# Setup all database schemas
python cli.py setup --all
```

This creates:
- `josh_content` & `content_chunks` - Blog content with embeddings
- `youtube_content` & `youtube_chunks` - YouTube transcripts with embeddings
- `test_data_chunks` - Performance benchmark data with embeddings
- `configs` - Product configurations
- `rag_query_logs` - API query logs

### 5. Sync Data from External Sources

```bash
# Sync product configs from production database
python cli.py sync --products

# Sync all articles from Sanity CMS
python cli.py sync --sanity --all

# Or sync everything at once
python cli.py sync --all
```

### 6. Ingest YouTube & Local content

**Ingest transcripts from `raw/youtube/`:**
```bash
python cli.py ingest --youtube
```

**Local JSON Ingestion (Optional):**
If you have local blog exports in `raw/blogs/`, you can ingest them manually:
```bash
python cli.py ingest --blogs
```

**Generate missing embeddings:**
If any content is missing embeddings, run:
```bash
python cli.py ingest --embeddings
```

### 7. Create Vector Indexes

```bash
# Create HNSW indexes for fast similarity search
python cli.py setup --indexes
```

This creates optimized vector indexes for semantic search. The `intfloat/e5-base-v2` model (~400MB) is downloaded automatically when generating embeddings.

### 8. Start API Server

```bash
python cli.py api
```

API is now running at `http://localhost:8000`

---

## Verify Installation

### Check Database

```bash
# Connect to PostgreSQL
docker exec -it jj-rag-pipeline-postgres-1 psql -U postgres -d josh_rag

# Check content
SELECT COUNT(*) FROM josh_content;
SELECT COUNT(*) FROM content_chunks WHERE embedding IS NOT NULL;
SELECT COUNT(*) FROM configs;

# Exit
\q
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations (Structured)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"quiz_response": {"profession": ["student"]}, "top_k": 3}'

# Test Streaming API
uv run python scripts/test_streaming_api.py
```

---

## Project Structure

```
jj-rag-pipeline/
├── src/
│   ├── data_pipeline/           # Content extraction & embeddings
│   │   ├── content_extractor.py # Extract from Sanity JSON & Raw files
│   │   ├── embedding_generator.py # Gemini/SentenceTransformer embeddings
│   │   ├── sanity_client.py     # Sanity API client
│   │   └── youtube_script_parser.py # YouTube transcript parser
│   ├── rag/                     # RAG components
│   │   ├── retriever.py         # Semantic search
│   │   ├── ranker.py            # Josh + spec scoring
│   │   ├── product_client.py    # Config database access
│   │   └── spec_fallback.py     # Spec-only fallback
│   └── api/                     # FastAPI service
│       ├── main.py              # API endpoints
│       ├── webhooks.py          # Sanity webhooks
│       └── ingestion_service.py # Background ingestion
├── scripts/                     # Unified CLI tools
│   ├── setup.py                 # Database schema setup
│   ├── ingest.py                # Content ingestion & embeddings
│   ├── sync.py                  # Sync from external sources
│   └── youtube.py               # YouTube transcript fetching
├── tests/                       # Test suite
│   └── test_rag_evaluation.py   # Full pipeline evaluation
├── raw/                         # Local content data
│   ├── blogs/                   # Blog JSON files
│   └── youtube/                 # YouTube transcripts
├── docker-compose.yml           # PostgreSQL + Redis
├── pyproject.toml               # Dependencies
├── SETUP.md                     # This file
└── TECHNICAL.md                 # Technical documentation
```

---

## Common Tasks

### Update Content from Sanity

```bash
# Sync all articles from Sanity CMS
python cli.py sync --sanity --all

# Sync specific article
python cli.py sync --sanity --id=<article_id>
```

### Fetch YouTube Transcripts

```bash
# Test with 1 video
python cli.py youtube --test

# Fetch 20 videos
python cli.py youtube --fetch --count=20

# Then ingest and generate embeddings
python cli.py ingest --youtube
python cli.py ingest --embeddings --youtube-only
```

### Update Product Configs

```bash
# Sync from production database
python cli.py sync --products
```

### Sync Test Data (Performance Benchmarks)

```bash
# Sync all available test data PDFs from S3
python cli.py sync --test-data

# Sync specific config
python cli.py sync --test-data --config-id=15

# Sync with limit (useful for testing)
python cli.py sync --test-data --limit=10
```

**What is Test Data?**
- Performance benchmarks for specific laptop configurations
- Examples: gaming FPS, video rendering times, battery life tests
- Stored as PDFs in AWS S3, parsed and embedded for semantic search
- Currently synced: 229 configs with 2,126 benchmark chunks

### Re-generate Embeddings

```bash
# For all content
python cli.py ingest --embeddings --regenerate

# For blogs only
python cli.py ingest --embeddings --blogs-only --regenerate

# For YouTube only
python cli.py ingest --embeddings --youtube-only --regenerate
```

### Enable API Authentication

```bash
# 1. Set API key in .env
API_KEY=your_secure_api_key_here

# 2. Restart API server
# Press Ctrl+C to stop, then:
uv run uvicorn src.api.main:app --reload

# 3. Test with auth
curl -X POST http://localhost:8000/recommend \
  -H "Authorization: Bearer your_secure_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"quiz_response": {...}}'
```

### Monitor Query Logs

```bash
docker exec -it jj-rag-pipeline-postgres-1 psql -U postgres -d josh_rag

SELECT 
  query_text,
  results->>'source' as source,
  response_time_ms,
  created_at
FROM rag_query_logs
ORDER BY created_at DESC
LIMIT 20;
```

### Docker Commands

```bash
# Stop containers
docker-compose down

# View logs
docker-compose logs -f postgres
docker-compose logs -f redis

# Restart containers
docker-compose restart

# Remove all data (WARNING: deletes database)
docker-compose down -v
```

---

## Troubleshooting

### "Connection refused" when connecting to database

```bash
# Check if containers are running
docker ps

# If not running, start them
docker-compose up -d

# Check logs
docker-compose logs postgres
```

### "Model not found" error

The embedding model downloads automatically on first use (~400MB). Ensure you have internet connection.

### "No results found" from API

```bash
# Check if embeddings exist
docker exec -it jj-rag-pipeline-postgres-1 psql -U postgres -d josh_rag -c \
  "SELECT COUNT(*) FROM content_chunks WHERE embedding IS NOT NULL;"

# If 0, generate embeddings
uv run python scripts/generate_embeddings.py
```

### API returns 401 Unauthorized

If `API_KEY` is set in `.env`, you must include it in requests:
```bash
-H "Authorization: Bearer YOUR_API_KEY"
```

To disable auth, remove or empty the `API_KEY` variable.

---

## Next Steps

1. Read [TECHNICAL.md](TECHNICAL.md) for architecture details
2. See [RAG_IMPLEMENTATION_GUIDE.md](RAG_IMPLEMENTATION_GUIDE.md) for implementation strategy
3. Run evaluation: `uv run python tests/test_rag_evaluation.py`
4. Integrate with your frontend/backend

---

## Support

For issues or questions, refer to:
- Technical documentation: `TECHNICAL.md`
- Implementation guide: `RAG_IMPLEMENTATION_GUIDE.md`
- Database schema: `docs/prod-schema/`
