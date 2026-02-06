# Future Improvements & Next Steps

This document outlines potential enhancements and next steps for the JustJosh RAG Pipeline.

---

## Current Status

### ✅ Completed
- Basic RAG pipeline (retriever + ranker)
- Multi-source content (Sanity blogs + YouTube videos)
- Product config syncing from production DB
- Date-based content filtering
- Spec-based fallback for low confidence
- RAG query logging
- API key authentication
- FastAPI service with webhooks (Sanity)
- Consolidated CLI (`setup`, `sync`, `api`)

### 🚧 In Progress (Current Plan)
- **Unified Sync Architecture Overhaul**
  - Cursor-based incremental syncing for all sources
  - YouTube webhook (PubSubHubbub)
  - Products polling (every 5 min)
  - Eliminate `raw/` folder
  - Auto-embedding after sync
  - Consolidate scripts (`ingest.py` → `sync.py`, `youtube.py` → `sync.py`)
- **Improved Product Extraction**
  - Multi-signal scoring (title, frequency, position, context)
  - Distinguish main product from passing mentions

---

## Immediate Next Steps

### 1. Test Data Ingestion & Integration

**Problem:** Test data (performance benchmarks for each config) is not yet ingested into the RAG pipeline.

**What is Test Data?**
- Performance benchmarks for specific laptop configurations
- Examples: gaming FPS, video rendering times, battery life tests
- Stored in production DB: `configs.test_data` (JSONB column)
- Already synced to local DB but not yet used in recommendations

**Implementation Plan:**

#### A. Database Schema Enhancement
```sql
-- Already exists in configs table:
-- test_data JSONB (contains benchmark results)

-- Create dedicated table for test data chunks (optional, for better search)
CREATE TABLE test_data_chunks (
  id SERIAL PRIMARY KEY,
  config_id INT REFERENCES configs(id),
  test_type VARCHAR(100),  -- 'gaming', 'rendering', 'battery', etc.
  test_description TEXT,
  benchmark_results JSONB,
  chunk_text TEXT,  -- Natural language description for embedding
  embedding vector(768),
  created_at TIMESTAMP DEFAULT NOW()
);
```

#### B. Test Data Extraction & Chunking
```python
# In sync.py or new test_data_processor.py

def process_test_data(config_id: int, test_data: dict):
    """
    Extract test data and create searchable chunks.
    
    Example test_data:
    {
      "gaming": {
        "cyberpunk_2077": {"fps": 85, "settings": "high"},
        "fortnite": {"fps": 144, "settings": "epic"}
      },
      "rendering": {
        "davinci_resolve": {"4k_timeline": "2.3x realtime"}
      },
      "battery": {
        "video_playback": "12.5 hours",
        "web_browsing": "15 hours"
      }
    }
    """
    chunks = []
    
    for category, tests in test_data.items():
        # Create natural language description
        chunk_text = f"Performance tests for {category}:\n"
        
        for test_name, results in tests.items():
            chunk_text += f"- {test_name}: {format_results(results)}\n"
        
        chunks.append({
            'config_id': config_id,
            'test_type': category,
            'chunk_text': chunk_text,
            'benchmark_results': tests
        })
    
    return chunks
```

#### C. Integration with Retriever
```python
# Option 1: Add test_data_chunks to UNION query in retriever.py
SELECT ... FROM content_chunks WHERE ...
UNION ALL
SELECT ... FROM youtube_chunks WHERE ...
UNION ALL
SELECT ... FROM test_data_chunks WHERE ...

# Option 2: Use test_data in ranker scoring
# Boost configs that have strong test results for user's use case
if user_use_case == 'gaming':
    test_score = calculate_gaming_performance_score(config.test_data)
    final_score += test_score * 0.2  # 20% weight
```

#### D. Recommendation Enhancement
- Show actual benchmark numbers in explanations
- "This laptop achieves 85 FPS in Cyberpunk 2077 at high settings"
- "Battery lasts 12.5 hours for video playback"
- Compare test results between recommendations

**Priority:** High (adds concrete, measurable data to recommendations)

---

## Phase 2: Performance & Scalability

### 2. Redis Caching Layer

**Why:** Reduce database load and improve API response time

**Implementation:**
```python
# Cache structure
cache_key = f"rec:{hash(quiz_response)}"
ttl = 3600  # 1 hour

# Cache recommendations
redis.setex(cache_key, ttl, json.dumps(recommendations))

# Cache embeddings
embedding_key = f"emb:{content_hash}"
redis.setex(embedding_key, 604800, embedding.tobytes())  # 7 days
```

**What to Cache:**
- Frequently requested quiz combinations
- Product embeddings (avoid regenerating)
- Popular recommendations
- Config details (specs, prices)

**Estimated Impact:** 50-80% reduction in response time for cached queries

---

### 3. Database Connection Pooling

**Current Issue:** Creates new DB connection for each request

**Solution:**
```python
from psycopg2.pool import ThreadedConnectionPool

# Initialize pool at startup
db_pool = ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host=..., port=..., database=...
)

# Use pool in components
conn = db_pool.getconn()
try:
    # ... use connection
finally:
    db_pool.putconn(conn)
```

**Estimated Impact:** 20-30% reduction in query latency

---

## Phase 3: Monitoring & Analytics

### 4. Query Analytics Dashboard

**Track:**
- Most common quiz combinations
- RAG vs. fallback usage ratio (target: 80%+ RAG)
- Low confidence queries (investigate why)
- Recommendation click-through rates
- Average response time (P50/P95/P99)
- Content source distribution (blog vs YouTube vs test data)

**Implementation:**
- Use existing `rag_query_logs` table
- Create simple dashboard (Grafana, Metabase, or custom)
- Add daily/weekly email reports

**Queries to Monitor:**
```sql
-- RAG vs Fallback ratio
SELECT 
  results->>'source' as source,
  COUNT(*) as count,
  AVG(response_time_ms) as avg_latency
FROM rag_query_logs
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY results->>'source';

-- Low confidence queries
SELECT 
  query_text,
  results->>'confidence' as confidence,
  response_time_ms
FROM rag_query_logs
WHERE (results->>'confidence')::float < 0.75
ORDER BY created_at DESC
LIMIT 50;
```

---

### 5. Content Freshness Monitoring

**Alerts:**
- Content older than 6 months (flag for review)
- Sync failures (Sanity, YouTube, Products)
- Missing embeddings
- Broken config_id references

**Implementation:**
```sql
-- Old content check
SELECT 
  title,
  publish_date,
  AGE(NOW(), publish_date) as age
FROM josh_content
WHERE publish_date < NOW() - INTERVAL '6 months'
ORDER BY publish_date;

-- Missing embeddings
SELECT COUNT(*) 
FROM content_chunks 
WHERE embedding IS NULL;
```

---

## Phase 4: Content Quality

### 6. Automated Content Validation

**Checks:**
- Verify all mentioned products exist in `configs` table
- Flag outdated specs (e.g., "Intel 10th gen" when 14th gen exists)
- Detect broken config_id references
- Identify duplicate/similar chunks

**Implementation:**
```python
def validate_content():
    """Run nightly validation checks."""
    issues = []
    
    # Check 1: Orphaned product mentions
    for chunk in get_all_chunks():
        products = extract_product_mentions(chunk.text)
        for product in products:
            if not config_exists(product):
                issues.append(f"Unknown product: {product} in chunk {chunk.id}")
    
    # Check 2: Outdated specs
    for chunk in get_all_chunks():
        if contains_outdated_specs(chunk.text):
            issues.append(f"Outdated specs in chunk {chunk.id}")
    
    # Check 3: Duplicate content
    duplicates = find_duplicate_chunks(similarity_threshold=0.95)
    
    return issues
```

---

### 7. Content Coverage Analysis

**Goal:** Identify gaps in Josh's content

**Analysis:**
```python
# Find underserved query patterns
SELECT 
  query_text,
  COUNT(*) as frequency,
  AVG((results->>'confidence')::float) as avg_confidence
FROM rag_query_logs
WHERE (results->>'confidence')::float < 0.75
GROUP BY query_text
ORDER BY frequency DESC
LIMIT 20;

# Suggest content topics
# "Many users ask about 'budget gaming laptops' but confidence is low"
# → Suggest Josh create content on this topic
```

---

## Phase 5: Recommendation Quality

### 8. A/B Testing Framework

**Test Variations:**
- Josh score weight: 70% vs 80% vs 60%
- Different embedding models (e5-base-v2 vs e5-large)
- RAG vs spec-only recommendations
- Different chunking strategies (512 vs 768 tokens)

**Implementation:**
```python
# Assign users to test groups
test_group = hash(session_id) % 100

if test_group < 50:
    # Control: Josh 70%
    josh_weight = 0.7
else:
    # Treatment: Josh 80%
    josh_weight = 0.8

# Log which group user is in
log_query(..., test_group=test_group)

# Analyze results
compare_metrics(control_group, treatment_group)
```

---

### 9. User Feedback Loop

**Collect Feedback:**
```python
@app.post("/feedback")
async def submit_feedback(
    query_id: int,
    helpful: bool,
    clicked_config_id: Optional[int] = None,
    comment: Optional[str] = None
):
    """Track which recommendations users find helpful."""
    save_feedback(query_id, helpful, clicked_config_id, comment)
```

**Use Feedback:**
- Identify poorly performing recommendations
- Adjust scoring weights based on click patterns
- Retrain/fine-tune embedding model (advanced)

---

## Phase 6: Advanced Features

### 10. Personalization

**Track User History:**
```sql
CREATE TABLE user_sessions (
  session_id VARCHAR(255) PRIMARY KEY,
  quiz_history JSONB[],
  clicked_configs INT[],
  created_at TIMESTAMP DEFAULT NOW()
);
```

**Personalized Recommendations:**
- "You previously looked at gaming laptops, here are similar options"
- Learn user preferences over time
- Adjust scoring based on past behavior

---

### 11. Contextual Recommendations

**Features:**
- "If you liked X, you'll love Y" (similar configs)
- "Alternatives to X" (same use case, different brand/price)
- "Upgrade from X" (better specs, higher price)

**Implementation:**
```python
@app.get("/similar/{config_id}")
async def get_similar_products(config_id: int, top_k: int = 5):
    """Find similar products using embedding similarity."""
    config = get_config(config_id)
    
    # Use config specs as query
    similar = find_similar_configs(
        specs=config.specs,
        use_cases=config.use_cases,
        exclude_id=config_id,
        top_k=top_k
    )
    
    return similar
```

---

### 12. Price Tracking & Alerts

**Features:**
- Track price history for recommended products
- Alert users when products go on sale
- Show "best time to buy" based on historical data

**Implementation:**
```sql
CREATE TABLE price_history (
  id SERIAL PRIMARY KEY,
  config_id INT REFERENCES configs(id),
  price DECIMAL(10,2),
  recorded_at TIMESTAMP DEFAULT NOW()
);

-- Track daily
INSERT INTO price_history (config_id, price)
SELECT id, current_price FROM configs;
```

---

## Phase 7: DevOps & Infrastructure

### 13. Docker Containerization

**Current:** docker-compose.yml exists but could be enhanced

**Improvements:**
- Multi-stage builds (smaller images)
- Separate containers for API, sync workers, Redis
- Health checks for all services
- Volume management for embeddings

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  sync-worker:
    build: .
    command: python cli.py sync --all
    depends_on:
      - postgres
    # Run every hour
    restart: unless-stopped
  
  postgres:
    image: ankane/pgvector:latest
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
```

---

### 14. CI/CD Pipeline

**Automated Tests:**
```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          uv sync
          uv run pytest tests/
      - name: Check lints
        run: uv run ruff check .
```

**Auto-Deploy:**
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          ssh user@server 'cd /app && git pull && docker-compose up -d'
```

---

### 15. Monitoring & Alerting

**Tools:**
- **Sentry** for error tracking
- **Prometheus + Grafana** for metrics
- **Uptime monitoring** (UptimeRobot, Pingdom)

**Alerts:**
- API response time > 1s
- Error rate > 1%
- Sync failures
- Database connection issues
- Low disk space

---

## Phase 8: Testing

### 16. Comprehensive Test Suite

**Unit Tests:**
```python
# tests/test_retriever.py
def test_query_construction():
    retriever = RAGRetriever()
    query = retriever.construct_query({
        'profession': ['student'],
        'use_case': ['gaming']
    })
    assert 'gaming' in query.lower()
    assert 'student' in query.lower()
```

**Integration Tests:**
```python
# tests/test_api.py
def test_recommend_endpoint():
    response = client.post("/recommend", json={
        "profession": ["student"],
        "use_case": ["gaming"],
        "budget": ["value"]
    })
    assert response.status_code == 200
    assert len(response.json()['recommendations']) > 0
```

**Load Tests:**
```python
# tests/load_test.py
from locust import HttpUser, task

class RAGUser(HttpUser):
    @task
    def get_recommendations(self):
        self.client.post("/recommend", json={...})

# Run: locust -f tests/load_test.py --users 100 --spawn-rate 10
```

---

## Priority Recommendations

### 🔥 High Priority (Do First)
1. **Test Data Ingestion** - Adds concrete performance data to recommendations
2. **Unified Sync Architecture** - Already planned, critical for maintainability
3. **Redis Caching** - Significant performance improvement
4. **Query Analytics Dashboard** - Understand system usage

### 🎯 Medium Priority (After MVP Launch)
5. **A/B Testing Framework** - Validate improvements
6. **User Feedback Loop** - Learn from real usage
7. **Docker Containerization** - Easier deployment
8. **Monitoring & Alerting** - Production readiness

### 💡 Low Priority (Nice to Have)
9. **Personalization** - Requires significant user data
10. **Price Tracking** - Additional complexity
11. **Advanced Content Analysis** - Diminishing returns

---

## Estimated Timeline

### Sprint 1 (Current) - 1 week
- ✅ Unified sync architecture
- ✅ Webhooks (Sanity + YouTube)
- ✅ Improved product extraction

### Sprint 2 - 1 week
- Test data ingestion & integration
- Redis caching layer
- Database connection pooling

### Sprint 3 - 1 week
- Query analytics dashboard
- Content validation automation
- Basic monitoring

### Sprint 4 - 1 week
- A/B testing framework
- User feedback endpoints
- Docker containerization

### Sprint 5+ - Ongoing
- Advanced features (personalization, price tracking)
- Continuous optimization
- Scale as needed

---

## Success Metrics

### Current Baseline
- RAG confidence rate: ~70% (target: 85%+)
- API latency: ~200ms (target: <100ms)
- Recommendation accuracy: Manual validation needed

### Target Metrics (3 months)
- RAG confidence rate: 85%+
- API latency P95: <100ms
- User satisfaction: 4.5+/5.0
- Click-through rate: 30%+
- Spec fallback usage: <15%

---

## Notes

- Focus on incremental improvements
- Measure impact of each change
- Don't over-engineer early
- User feedback is critical
- Keep it simple and maintainable

---

**Last Updated:** 2026-02-04
**Next Review:** After Sprint 1 completion
