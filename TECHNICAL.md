# Technical Documentation

Complete technical reference for Josh's RAG Pipeline.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Pipeline](#data-pipeline)
3. [RAG Components](#rag-components)
4. [API Reference](#api-reference)
5. [Database Schema](#database-schema)
6. [Configuration](#configuration)
7. [Performance](#performance)

---

## Architecture Overview

### System Flow

**Option 1: Natural Language Prompt**
```
User Prompt → LLM Intent Extraction → Structured Quiz
                                           ↓
                                    API → Retriever → Ranker → Recommendations
                                           ↓
                                    (if low confidence)
                                           ↓
                                      Spec Fallback → Recommendations
```

**Option 2: Structured Quiz (Legacy)**
```
User Quiz → API → Retriever → Ranker → Recommendations
                      ↓
                  (if low confidence)
                      ↓
                 Spec Fallback → Recommendations
```

### Components

1. **Intent Extractor** - LLM-based extraction of structured intent from natural language
2. **Data Pipeline** - Unified sync from multiple sources with auto-embedding
3. **Retriever** - Semantic search using vector similarity across all sources
4. **Ranker** - Multi-signal scoring (Josh context + Specs + Test data)
5. **Spec Fallback** - Pure spec matching for low confidence queries
6. **API** - FastAPI service with webhooks, polling, authentication, and logging

### Technology Stack

- **Language:** Python 3.12+
- **Framework:** FastAPI
- **Database:** PostgreSQL 15 with pgvector
- **Cache:** Redis
- **Embeddings:** Sentence Transformers (e5-base-v2)
- **Package Manager:** uv

---

## Data Pipeline

### Content Sources

**1. Sanity CMS (Blogs, Reviews, Guides)**

**Process:**
1. Fetch articles via Sanity API
2. Extract metadata (title, URL, publish date, tags)
3. Parse content sections
4. Store in `josh_content` table

**Key Fields:**
- `title`, `content_type`, `url`, `publish_date`
- `tags` (use case tags: gaming, programming, etc.)
- `author`, `updated_at`

**2. YouTube Transcripts**

**Process:**
1. Fetch transcripts via YouTube API
2. Extract metadata (title, video ID, publish date)
3. Store in `youtube_content` table
4. Chunk and embed transcripts

**3. Test Data (Performance Benchmarks)**

**Source:** AWS S3 (PDF files)

**Process:**
1. Query `configs` table for `test_data_pdf_key`
2. Download PDFs from S3 using boto3
3. Parse PDFs to extract benchmark sections
4. Store in `test_data_chunks` table with `config_id` foreign key

**Benchmark Categories:**
- Gaming (FPS, settings, games tested)
- Rendering (DaVinci Resolve, Premiere Pro)
- Battery life (video playback, web browsing)
- Synthetic benchmarks (Geekbench, Cinebench, 3DMark)

**Current Status:** 229 configs synced with 2,126 benchmark chunks

### Chunking Strategy

**Algorithm:** Sliding window with overlap

**Parameters:**
- Chunk size: 512 tokens
- Overlap: 50 tokens
- Preserves: Section titles, metadata

**Metadata Extraction:**
- Product mentions (regex-based)
- Config IDs (from embedded data)
- Rankings, prices, specs

**Storage:** `content_chunks` table

### Embedding Generation

**Model:** `intfloat/e5-base-v2`
- Dimensions: 768
- Max tokens: 512
- Trained on: Diverse text corpus

**Process:**
1. Prefix query: "query: " for search queries
2. Prefix passage: "passage: " for content chunks
3. Generate embeddings via Sentence Transformers
4. Store as pgvector in PostgreSQL

**Indexing:** HNSW index for fast similarity search

---

## RAG Components

### 1. Retriever (`src/rag/retriever.py`)

**Purpose:** Find relevant content chunks via semantic search

**Algorithm:**
```python
1. Convert quiz → natural language query
2. Generate query embedding (768d)
3. Vector similarity search (cosine distance)
4. Filter by date threshold (optional)
5. Return top K chunks with metadata
```

**Query Construction:**
```python
def construct_query(quiz_response: Dict) -> str:
    parts = []
    if profession: parts.append(f"for {profession}")
    if use_case: parts.append(f"for {use_case}")
    if budget: parts.append(f"{budget} budget")
    if portability: parts.append(f"{portability} laptop")
    return f"laptop recommendation {' '.join(parts)}"
```

**SQL Query (Multi-Source UNION):**
```sql
-- Blogs
SELECT ... FROM content_chunks cc
JOIN josh_content jc ON cc.content_id = jc.id
WHERE cc.embedding IS NOT NULL

UNION ALL

-- YouTube
SELECT ... FROM youtube_chunks yc
JOIN youtube_content yt ON yc.content_id = yt.id
WHERE yc.embedding IS NOT NULL

UNION ALL

-- Test Data
SELECT ... FROM test_data_chunks td
JOIN configs c ON td.config_id = c.config_id
WHERE td.embedding IS NOT NULL

ORDER BY similarity DESC
LIMIT %s;
```

**Output:** `List[RetrievalResult]` with similarity scores

### 2. Ranker (`src/rag/ranker.py`)

**Purpose:** Score and rank products from retrieved chunks

**Scoring Formula:**
```
Total Score = (Josh Score × 0.60) + (Spec Score × 0.25) + (Test Data Score × 0.15)
```

**Score Components:**

1. **Josh Score (60%)** - Semantic relevance from blog/YouTube content
2. **Spec Score (25%)** - Matching user requirements (RAM, storage, GPU, etc.)
3. **Test Data Score (15%)** - Presence of performance benchmarks

**Josh Score Components:**
- Ranking position: 0-1 (1st place = 1.0, 10th = 0.1)
- Similarity: Average chunk similarity
- Mention frequency: Number of mentions across chunks

**Spec Score Components:**
- Budget match: Price range alignment
- Use case match: GPU, RAM, CPU requirements
- Portability match: Weight and screen size
- Screen size match: User preference alignment

**Product Extraction:**
1. **Primary:** Group by `config_id` from chunk metadata
2. **Fallback:** Extract product names via regex, map to configs

**Config Resolution:**
When multiple configs match a product name:
```
Score = (Josh Context × 0.5) + (User Requirements × 0.3) + (Rating × 0.2)
```

**Explanation Generation:**
```python
def generate_explanation(product: Dict) -> str:
    # Extract from chunks:
    # - Josh's ranking/opinion
    # - Specific specs mentioned
    # - Pros/cons
    # - Price mentions
    # - Direct quotes
    return formatted_explanation
```

### 3. Spec Fallback (`src/rag/spec_fallback.py`)

**Trigger:** RAG confidence < 0.75 (configurable)

**Algorithm:**
```python
1. Fetch all configs from database
2. Score each config:
   - Budget match (0-1)
   - Use case match (0-1)
   - Portability match (0-1)
   - Screen size match (0-1)
3. Average scores
4. Return top K
```

**No Josh Context:** Explanations are spec-based only

---

## API Reference

### Base URL

```
http://localhost:8000
```

### Authentication

**Optional:** Set `API_KEY` in `.env` to enable

**Format:**
```
Authorization: Bearer YOUR_API_KEY
```

### Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "retriever": "ready",
  "ranker": "ready"
}
```

#### `POST /recommend`

Get laptop recommendations based on quiz response.

**Request:**
```json
{
  "quiz_response": {
    "profession": ["student", "developer"],
    "use_case": ["programming", "video_editing"],
    "budget": ["value", "premium"],
    "portability": "light",
    "screen_size": ["14 inch", "15-16 inch"]
  },
  "top_k": 5
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "product_name": "MacBook Air 13 Apple",
      "config_id": 26,
      "confidence_score": 0.89,
      "josh_score": 0.92,
      "spec_score": 0.83,
      "ranking": 1,
      "source_article": "Best Laptops for Programmers",
      "source_url": "https://...",
      "explanation": "Josh ranks this #1 for programmers..."
    }
  ],
  "query": "laptop recommendation for student developer...",
  "total_results": 5
}
```

**Status Codes:**
- `200` - Success
- `401` - Unauthorized (if API_KEY is set)
- `404` - No recommendations found
- `500` - Server error

#### `POST /search`

Direct semantic search in Josh's content.

**Request:**
```json
{
  "query": "best laptop for gaming under $1000",
  "top_k": 10
}
```

**Response:**
```json
{
  "query": "best laptop for gaming under $1000",
  "results": [
    {
      "chunk_id": 123,
      "content_title": "Best Gaming Laptops for $600",
      "chunk_text": "...",
      "similarity": 0.87,
      "section_title": "Budget Gaming Options",
      "url": "https://..."
    }
  ],
  "total_results": 10
}
```

#### `POST /webhook/sanity`

Webhook for Sanity CMS content updates.

**Headers:**
```
X-Sanity-Webhook-Signature: <signature>
```

**Payload:** Sanity webhook payload

**Response:**
```json
{
  "status": "processing",
  "message": "Content ingestion started in background"
}
```

---

## Database Schema

### `josh_content`

Content metadata from Sanity CMS.

```sql
CREATE TABLE josh_content (
    id SERIAL PRIMARY KEY,
    sanity_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content_type VARCHAR(50),
    url TEXT,
    publish_date DATE,
    updated_at TIMESTAMP,
    tags TEXT[],
    author VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_josh_content_sanity_id ON josh_content(sanity_id);
CREATE INDEX idx_josh_content_publish_date ON josh_content(publish_date);
CREATE INDEX idx_josh_content_updated_at ON josh_content(updated_at);
```

### `content_chunks`

Text chunks with embeddings and metadata.

```sql
CREATE TABLE content_chunks (
    id SERIAL PRIMARY KEY,
    content_id INTEGER REFERENCES josh_content(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    section_title TEXT,
    chunk_index INTEGER,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_content_chunks_content_id ON content_chunks(content_id);
CREATE INDEX idx_content_chunks_embedding ON content_chunks 
    USING hnsw (embedding vector_cosine_ops);
```

**Metadata JSONB:**
```json
{
  "config_id": 123,
  "product_mentions": ["MacBook Air", "Dell XPS"],
  "ranking": 1,
  "price": "$1299",
  "specs": {...}
}
```

### `configs`

Product configurations synced from production.

```sql
CREATE TABLE configs (
    config_id INTEGER PRIMARY KEY,
    product_id INTEGER,
    product_name VARCHAR(255) NOT NULL,
    brand VARCHAR(100),
    model VARCHAR(255),
    specs JSONB,
    price NUMERIC(10,2),
    final_rating NUMERIC(3,2),
    test_data JSONB,
    last_synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_configs_product_name ON configs(product_name);
CREATE INDEX idx_configs_brand ON configs(brand);
CREATE INDEX idx_configs_price ON configs(price);
```

**Specs JSONB:**
```json
{
  "Processor": "Apple M3",
  "Memory Amount": "16GB",
  "Storage Amount": "512GB SSD",
  "Display Size": "13.6 inches",
  "Weight (lbs)": "2.7",
  "Dedicated Graphics (Yes/No)": "No"
}
```

### `rag_query_logs`

API query logs for monitoring.

```sql
CREATE TABLE rag_query_logs (
    id SERIAL PRIMARY KEY,
    query_text TEXT,
    top_k INTEGER,
    results JSONB,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rag_query_logs_created_at ON rag_query_logs(created_at);
```

**Results JSONB:**
```json
{
  "quiz": {...},
  "recommendations": [...],
  "source": "josh_rag_primary",
  "confidence": 0.89
}
```

---

## Configuration

### Environment Variables

**Database:**
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=josh_rag
DB_USER=postgres
DB_PASSWORD=postgres
```

**Sanity CMS:**
```bash
SANITY_PROJECT_ID=your_project_id
SANITY_DATASET=production
SANITY_API_TOKEN=your_read_token
SANITY_API_VERSION=v2021-10-21
SANITY_WEBHOOK_SECRET=optional
```

**RAG Configuration:**
```bash
EMBEDDING_MODEL=intfloat/e5-base-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
RAG_CONFIDENCE_THRESHOLD=0.75
```

**Content Filtering:**
```bash
# Option 1: Explicit date
CONTENT_MIN_PUBLISH_DATE=2024-01-01

# Option 2: Relative age
CONTENT_MAX_AGE_YEARS=1
```

**API:**
```bash
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your_secure_api_key  # Leave empty to disable auth
```

**Production Database (optional):**
```bash
PROD_DB_HOST=your_rds_host.rds.amazonaws.com
PROD_DB_PORT=5432
PROD_DB_NAME=your_prod_db_name
PROD_DB_USER=postgres
PROD_DB_PASSWORD=your_prod_password
```

### Tuning Parameters

**Retrieval:**
- `TOP_K_RESULTS`: Number of chunks to retrieve (default: 5)
- `CONFIDENCE_THRESHOLD`: Min similarity for RAG (default: 0.75)

**Ranking:**
- Josh weight: 70% (hardcoded in `ranker.py`)
- Spec weight: 30% (hardcoded in `ranker.py`)

**Config Selection:**
- Josh context: 50% (hardcoded in `product_client.py`)
- User requirements: 30%
- Rating: 20%

---

## Performance

### Benchmarks

**Retrieval:**
- Embedding generation: ~50-100ms
- Vector search: ~50-100ms
- Total: ~100-200ms

**Ranking:**
- Product extraction: ~20-50ms
- Scoring: ~20-50ms
- Total: ~50-100ms

**API Latency:**
- RAG path: ~200-400ms
- Fallback path: ~50-100ms
- Target: <500ms P95

### Optimization

**Database:**
- HNSW index for vector search
- Indexes on foreign keys and date columns
- JSONB for flexible metadata

**Caching:**
- Redis for query results (not yet implemented)
- Model loaded once at startup

**Scaling:**
- Stateless API (horizontal scaling)
- Read replicas for database
- CDN for static content

---

## Development

### Adding New Features

1. **New Scoring Factor:**
   - Update `ranker.py` → `_calculate_spec_score()`
   - Adjust weights in scoring formula
   - Re-run evaluation

2. **New Data Source:**
   - Create extractor in `data_pipeline/`
   - Add to ingestion pipeline
   - Re-generate embeddings

3. **New API Endpoint:**
   - Add to `src/api/main.py`
   - Add authentication if needed
   - Update API documentation

### Testing

```bash
# Full pipeline evaluation
uv run python tests/test_rag_evaluation.py

# Output: tests/rag_evaluation_results.json
```

**Evaluation Metrics:**
- Config ID coverage
- Similarity scores
- Ranking consistency
- Explanation quality

### Debugging

**Enable verbose mode:**
```python
retriever = RAGRetriever(verbose=True)
ranker = RAGRanker(verbose=True)
```

**Check embeddings:**
```sql
SELECT 
    cc.id,
    cc.chunk_text,
    cc.embedding IS NOT NULL as has_embedding
FROM content_chunks cc
LIMIT 10;
```

**Monitor queries:**
```sql
SELECT 
    query_text,
    results->>'source' as source,
    response_time_ms,
    created_at
FROM rag_query_logs
ORDER BY created_at DESC
LIMIT 20;
```

---

## Deployment

### Production Checklist

- [ ] Set strong `API_KEY`
- [ ] Configure CORS properly
- [ ] Enable Redis caching
- [ ] Set up monitoring (logs, metrics)
- [ ] Configure database backups
- [ ] Set up SSL/TLS
- [ ] Rate limiting
- [ ] Error tracking (Sentry, etc.)

### Docker Deployment

```bash
# Build image
docker build -t jj-rag-api .

# Run container
docker run -d \
  --name jj-rag-api \
  -p 8000:8000 \
  --env-file .env \
  jj-rag-api
```

### Environment-Specific Configs

**Development:**
- `API_RELOAD=true`
- `ENVIRONMENT=development`
- Verbose logging

**Production:**
- `API_RELOAD=false`
- `ENVIRONMENT=production`
- Error logging only
- Enable caching
- Database read replicas

---

## Troubleshooting

### Low Confidence Scores

**Cause:** Content doesn't match user query

**Solutions:**
1. Add more content from Sanity
2. Adjust date threshold (allow older content)
3. Lower `RAG_CONFIDENCE_THRESHOLD`
4. Improve query construction

### Null Config IDs

**Cause:** Product name mismatch between content and database

**Solutions:**
1. Re-sync configs: `uv run python scripts/sync_products.py --full`
2. Check product name format in database
3. Update fuzzy matching in `product_client.py`

### Slow API Response

**Cause:** Model loading, database query, or network

**Solutions:**
1. Pre-load model at startup (already done)
2. Add database indexes
3. Enable Redis caching
4. Use database connection pooling

### Memory Issues

**Cause:** Large model or many embeddings

**Solutions:**
1. Increase Docker memory limit
2. Use smaller embedding model
3. Batch processing for embeddings
4. Database query pagination

---

## References

- **Sentence Transformers:** https://www.sbert.net/
- **pgvector:** https://github.com/pgvector/pgvector
- **FastAPI:** https://fastapi.tiangolo.com/
- **Sanity CMS:** https://www.sanity.io/docs
