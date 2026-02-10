# RAG IMPLEMENTATION GUIDE

> [!CAUTION]
> **UPDATE NOTICE (2026-02-10):** This guide contains historical implementation steps. For the most up-to-date setup instructions, authentication details, and API usage, please refer to:
> 1. **[README.md](README.md)** (Quick Start & Features)
> 2. **[SETUP.md](SETUP.md)** (Complete Installation & Troubleshooting)
> 3. **[TECHNICAL.md](TECHNICAL.md)** (Architecture & API Reference)

---
> 

---

## 🎯 **Implementation Philosophy**

```
Build → Test → Validate → Deploy → Measure → Iterate
```

**Key Principles:**

1. ✅ **Test retrieval quality at every step** (manual review required)
2. ✅ **Add safety rails** (confidence thresholds, fallbacks)
3. ✅ **A/B test gradually** (10% → 50% → 100%)
4. ✅ **Monitor metrics obsessively** (CTR, conversion, satisfaction)
5. ✅ **Keep it simple initially** (complexity kills MVPs)

---

## 📋 **Phase 1: RAG-Only MVP (2-3 Weeks)**

**Goal:** Validate Josh's content can power 70-85% of recommendations with high accuracy

**Success Criteria:**

- ✅ Top-5 retrieval accuracy >85% (manual review)
- ✅ RAG confidence >0.75 for 70%+ of queries
- ✅ CTR improvement >50% vs tag-based system
- ✅ <200ms P95 latency

---

## Week 1: Content Ingestion & Setup

### **Day 1-2: Environment Setup & Database Preparation**

### **Step 1.1: Create Project Structure**

```bash
# Create new Python project
mkdir jj-recommendation-engine
cd jj-recommendation-engine

# Initialize Git
git init
git remote add origin <your-repo-url>

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Create directory structure
mkdir -p {data/{raw,processed,test},src/{data_pipeline,rag,api,utils},tests,notebooks,scripts}

```

**Project Structure:**

```
jj-recommendation-engine/
├── data/
│   ├── raw/                    # Josh's content exports
│   │   ├── blogs/
│   │   └── test_queries.json  # For validation
│   ├── processed/
│   │   ├── content_chunks.jsonl
│   │   └── embeddings/
│   └── test/                   # Test datasets
│       ├── quiz_samples.json
│       └── expected_results.json
├── src/
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── content_extractor.py
│   │   ├── content_chunker.py
│   │   └── embedding_generator.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── ranker.py
│   │   └── explainer.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   └── routes.py
│   └── utils/
│       ├── __init__.py
│       ├── db_config.py
│       └── redis_client.py
├── tests/
│   ├── test_retrieval.py
│   ├── test_accuracy.py
│   └── test_api.py
├── notebooks/
│   └── 01_content_exploration.ipynb
├── scripts/
│   ├── setup_database.py
│   ├── ingest_content.py
│   └── validate_rag.py
├── requirements.txt
├── .env.example
└── README.md

```

**Validation:** ✅ Directory structure created

---

### **Step 1.2: Install Dependencies**

```bash
# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core RAG & NLP
sentence-transformers==2.3.1    # e5-base-v2 embeddings (768d)
langchain==0.1.0                # RAG framework
langchain-community==0.0.13     # Community integrations

# Vector DB & Database
psycopg2-binary==2.9.9          # PostgreSQL
pgvector==0.2.4                 # Vector extension
sqlalchemy==2.0.25              # ORM

# API Framework
fastapi==0.109.0                # API server
uvicorn[standard]==0.27.0       # ASGI server
pydantic==2.5.3                 # Data validation

# Caching
redis==5.0.1                    # Redis client

# Data Processing
pandas==2.1.4
numpy==1.26.3
beautifulsoup4==4.12.3          # HTML parsing
lxml==5.1.0                     # XML/HTML parser

# Testing & Validation
pytest==7.4.4
pytest-asyncio==0.23.3
httpx==0.26.0                   # Async HTTP client

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
tqdm==4.66.1
loguru==0.7.2                   # Better logging

# Development
ipython==8.20.0
jupyter==1.0.0
black==24.1.1                   # Code formatter
flake8==7.0.0                   # Linter
EOF

# Install dependencies
pip install -r requirements.txt

```

**Validation:** ✅ Run `pip list | grep sentence-transformers`

---

### **Step 1.3: Setup Database Tables**

```bash
# Create database setup script
cat > scripts/setup_database.py << 'EOF'
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection
conn = psycopg2.connect(
    host=os.getenv("DATABASE_HOST"),
    port=os.getenv("DATABASE_PORT"),
    database=os.getenv("DATABASE_NAME"),
    user=os.getenv("DATABASE_USER"),
    password=os.getenv("DATABASE_PASSWORD")
)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = conn.cursor()

print("📦 Setting up database tables...")

# 1. Enable pgvector extension
print("1. Enabling pgvector extension...")
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
print("✅ pgvector enabled")

# 2. Create josh_content table
print("2. Creating josh_content table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS josh_content (
  id SERIAL PRIMARY KEY,
  content_type VARCHAR(50) NOT NULL,  -- 'blog', 'video', 'review'
  title TEXT NOT NULL,
  raw_content TEXT NOT NULL,
  published_date DATE,
  url TEXT,
  use_case_tags TEXT[],  -- {'video_editing', 'gaming', 'student'}
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_josh_content_type ON josh_content(content_type);
CREATE INDEX IF NOT EXISTS idx_josh_content_date ON josh_content(published_date);
CREATE INDEX IF NOT EXISTS idx_josh_content_tags ON josh_content USING gin(use_case_tags);
""")
print("✅ josh_content table created")

# 3. Create content_chunks table
print("3. Creating content_chunks table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS content_chunks (
  id SERIAL PRIMARY KEY,
  content_id INT REFERENCES josh_content(id) ON DELETE CASCADE,
  chunk_text TEXT NOT NULL,
  chunk_index INT NOT NULL,  -- Order within content
  embedding_vector vector(768),  -- e5-base-v2 embeddings (768 dimensions)

  -- Store extracted data as JSONB for flexibility
  extracted_insights JSONB DEFAULT '{}',
  /* Example extracted_insights:
  {
    "products_mentioned": [
      {"name": "MacBook Pro 14", "config_id": 432, "sentiment": "positive"}
    ],
    "ranking": 1,
    "recommendation_type": "top_pick",
    "quotes": ["The only 14-inch laptop we recommend..."],
    "pros": ["Powerful", "Great display"],
    "cons": ["Expensive"],
    "who_is_this_for": "Ideal travel companion for editors"
  }
  */

  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_content_chunks_content ON content_chunks(content_id);
CREATE INDEX IF NOT EXISTS idx_extracted_insights ON content_chunks USING gin(extracted_insights);
""")
print("✅ content_chunks table created")

# 4. Create ivfflat index for vector search (after data is loaded)
print("4. Note: ivfflat index will be created after embedding ingestion")
print("   Run: CREATE INDEX idx_content_chunks_embedding ON content_chunks")
print("        USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);")

# 5. Create rag_query_logs table (for monitoring)
print("5. Creating rag_query_logs table...")
cursor.execute("""
CREATE TABLE IF NOT EXISTS rag_query_logs (
  id SERIAL PRIMARY KEY,
  quiz_session_id VARCHAR(255),
  query_text TEXT NOT NULL,
  query_embedding vector(768),
  top_results JSONB,  -- Top 5 retrieved chunks
  confidence_score FLOAT,
  recommendation_source VARCHAR(50),  -- 'josh_rag', 'spec_fallback', 'ml_fallback'
  latency_ms INT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_logs_session ON rag_query_logs(quiz_session_id);
CREATE INDEX IF NOT EXISTS idx_rag_logs_created ON rag_query_logs(created_at);
""")
print("✅ rag_query_logs table created")

print("\\n✨ Database setup complete!")
print("\\nNext steps:")
print("1. Ingest Josh's content: python scripts/ingest_content.py")
print("2. Generate embeddings: python scripts/generate_embeddings.py")
print("3. Create ivfflat index after embeddings are loaded")

cursor.close()
conn.close()
EOF

# Run setup
python scripts/setup_database.py

```

**Validation:**

```sql
-- Check tables created
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('josh_content', 'content_chunks', 'rag_query_logs');

```

**Expected:** 3 tables listed

---

### **Step 1.4: Configure Environment Variables**

```bash
# Create .env file
cat > .env << 'EOF'
# Database (same as Node.js backend)
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=justjosh
DATABASE_USER=your_db_user
DATABASE_PASSWORD=your_db_password

# Redis (for caching)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-api-key-here

# RAG Configuration
EMBEDDING_MODEL=intfloat/e5-base-v2
EMBEDDING_DIMENSION=768
RAG_TOP_K=10
RAG_CONFIDENCE_THRESHOLD=0.75

# Cache TTLs (seconds)
CACHE_TTL_RAG_RESULTS=3600      # 1 hour
CACHE_TTL_EMBEDDINGS=604800     # 7 days

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag_service.log
EOF

# Create .env.example (for repo)
cp .env .env.example
# Edit .env.example to remove sensitive values

```

**Validation:** ✅ `.env` file created with correct values

---

### **Day 3-4: Content Extraction & Ingestion**

### **Step 2.1: Export Josh's Blog Content**

**Manual Task:** Export all blog posts from your CMS/website

**Expected format:** HTML or Markdown files

```bash
# Create data/raw/blogs/ directory
mkdir -p data/raw/blogs

# Example blog post structure:
# data/raw/blogs/best-laptops-video-editing-2025.md

```

**Example blog post:**

```markdown
---
title: Best Laptops for Video Editing 2025
date: 2025-01-15
url: <https://bestlaptop.deals/blog/best-laptops-video-editing-2025>
use_case_tags: [video_editing, content_creation]
---

# Best Laptops for Video Editing 2025

After testing over 50 laptops, here are our top picks...

## #1 MacBook Pro 14 (M4 Max)

**Who is this for?** Ideal travel companion for editors who value performance on the go.

**Why we recommend it:** The only 14-inch laptop we recommend for serious editing. Nearly identical experience to the 16-inch model but extremely portable.

**Pros:**
- Excellent keyboard and trackpad
- Extremely powerful
- Long battery life (15 hours)
- Fantastic display
- Minimal fan noise

**Cons:**
- Expensive ($2,099)
- No Wi-Fi 7

**Our verdict:** If you need portability without sacrificing performance, this is the one.

---

## #2 MacBook Pro 16 (M4 Max)

...

```

**Validation:** ✅ 20-50 blog posts exported to `data/raw/blogs/`

---

### **Step 2.2: Build Content Extractor**

```python
# src/data_pipeline/content_extractor.py
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import yaml
from loguru import logger

class ContentExtractor:
    """Extract structured data from Josh's blog posts"""

    def __init__(self):
        self.product_patterns = [
            r"MacBook Pro (?:14|16)(?: \\([^)]+\\))?",
            r"Legion \\d+[A-Za-z]*",
            r"Dell XPS \\d+",
            r"ThinkPad [A-Za-z0-9]+",
            # Add more patterns based on your product catalog
        ]

    def extract_from_markdown(self, filepath: Path) -> Dict:
        """Extract content from Markdown blog post"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split frontmatter and content
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                body = parts[2].strip()
            else:
                frontmatter = {}
                body = content
        else:
            frontmatter = {}
            body = content

        # Extract product sections
        sections = self._extract_sections(body)

        return {
            'title': frontmatter.get('title', ''),
            'published_date': frontmatter.get('date', None),
            'url': frontmatter.get('url', ''),
            'use_case_tags': frontmatter.get('use_case_tags', []),
            'content_type': 'blog',
            'raw_content': body,
            'sections': sections,
            'products_mentioned': self._extract_products(body)
        }

    def _extract_sections(self, content: str) -> List[Dict]:
        """Split content into semantic sections"""
        sections = []
        current_section = {'title': '', 'content': '', 'rank': None}

        lines = content.split('\\n')
        for line in lines:
            # Check for ranking headers (## #1 MacBook Pro, etc.)
            rank_match = re.match(r'##\\s*#(\\d+)\\s+(.+)', line)
            if rank_match:
                # Save previous section
                if current_section['content']:
                    sections.append(current_section)

                # Start new section
                rank = int(rank_match.group(1))
                title = rank_match.group(2).strip()
                current_section = {
                    'title': title,
                    'content': '',
                    'rank': rank,
                    'type': 'product_recommendation'
                }
            else:
                current_section['content'] += line + '\\n'

        # Save last section
        if current_section['content']:
            sections.append(current_section)

        return sections

    def _extract_products(self, content: str) -> List[str]:
        """Extract product names mentioned"""
        products = set()
        for pattern in self.product_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            products.update(matches)
        return list(products)

    def extract_structured_insights(self, section: Dict) -> Dict:
        """Extract pros, cons, quotes, etc."""
        content = section['content']

        insights = {
            'ranking': section.get('rank'),
            'recommendation_type': 'top_pick' if section.get('rank') == 1 else 'recommended',
            'quotes': [],
            'pros': [],
            'cons': [],
            'who_is_this_for': None
        }

        # Extract "Who is this for?"
        who_match = re.search(r'\\*\\*Who is this for\\?\\*\\*\\s*(.+?)(?:\\n\\n|\\*\\*)', content, re.DOTALL)
        if who_match:
            insights['who_is_this_for'] = who_match.group(1).strip()

        # Extract pros
        pros_match = re.search(r'\\*\\*Pros:\\*\\*\\s*\\n((?:- .+\\n)+)', content)
        if pros_match:
            insights['pros'] = [
                line.strip('- ').strip()
                for line in pros_match.group(1).split('\\n')
                if line.strip()
            ]

        # Extract cons
        cons_match = re.search(r'\\*\\*Cons:\\*\\*\\s*\\n((?:- .+\\n)+)', content)
        if cons_match:
            insights['cons'] = [
                line.strip('- ').strip()
                for line in cons_match.group(1).split('\\n')
                if line.strip()
            ]

        # Extract quotes (text in quotes or after "Why we recommend it:")
        quote_match = re.search(r'\\*\\*Why we recommend it:\\*\\*\\s*(.+?)(?:\\n\\n|\\*\\*)', content, re.DOTALL)
        if quote_match:
            insights['quotes'].append(quote_match.group(1).strip())

        return insights

# Test the extractor
if __name__ == '__main__':
    extractor = ContentExtractor()

    # Test on first blog post
    blog_files = list(Path('data/raw/blogs').glob('*.md'))
    if blog_files:
        result = extractor.extract_from_markdown(blog_files[0])
        print(f"✅ Extracted: {result['title']}")
        print(f"   Sections: {len(result['sections'])}")
        print(f"   Products: {result['products_mentioned']}")
    else:
        print("❌ No blog posts found in data/raw/blogs/")

```

**Validation:**

```bash
python src/data_pipeline/content_extractor.py

```

**Expected output:** ✅ Blog post extracted with sections, products, and metadata

---

### **Step 2.3: Ingest Content into Database**

```python
# scripts/ingest_content.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import psycopg2
from dotenv import load_dotenv
import os
from src.data_pipeline.content_extractor import ContentExtractor
from loguru import logger
from tqdm import tqdm

load_dotenv()

def ingest_blog_posts():
    """Ingest all blog posts into josh_content table"""

    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv("DATABASE_HOST"),
        port=os.getenv("DATABASE_PORT"),
        database=os.getenv("DATABASE_NAME"),
        user=os.getenv("DATABASE_USER"),
        password=os.getenv("DATABASE_PASSWORD")
    )
    cursor = conn.cursor()

    extractor = ContentExtractor()
    blog_dir = Path('data/raw/blogs')
    blog_files = list(blog_dir.glob('*.md'))

    logger.info(f"📚 Found {len(blog_files)} blog posts")

    ingested = 0
    skipped = 0

    for filepath in tqdm(blog_files, desc="Ingesting blogs"):
        try:
            # Extract content
            data = extractor.extract_from_markdown(filepath)

            # Check if already exists (by URL)
            cursor.execute(
                "SELECT id FROM josh_content WHERE url = %s",
                (data['url'],)
            )
            existing = cursor.fetchone()

            if existing:
                logger.warning(f"⏭️  Skipping (already exists): {data['title']}")
                skipped += 1
                continue

            # Insert into josh_content
            cursor.execute("""
                INSERT INTO josh_content
                (content_type, title, raw_content, published_date, url, use_case_tags)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                data['content_type'],
                data['title'],
                data['raw_content'],
                data['published_date'],
                data['url'],
                data['use_case_tags']
            ))

            content_id = cursor.fetchone()[0]

            # Insert sections as chunks (we'll add embeddings later)
            for idx, section in enumerate(data['sections']):
                # Extract structured insights
                insights = extractor.extract_structured_insights(section)

                # Add products mentioned
                insights['products_mentioned'] = [
                    {'name': prod, 'sentiment': 'positive'}
                    for prod in data['products_mentioned']
                ]

                cursor.execute("""
                    INSERT INTO content_chunks
                    (content_id, chunk_text, chunk_index, extracted_insights)
                    VALUES (%s, %s, %s, %s)
                """, (
                    content_id,
                    section['content'],
                    idx,
                    psycopg2.extras.Json(insights)
                ))

            conn.commit()
            ingested += 1
            logger.success(f"✅ Ingested: {data['title']} ({len(data['sections'])} chunks)")

        except Exception as e:
            logger.error(f"❌ Error ingesting {filepath.name}: {e}")
            conn.rollback()
            continue

    logger.info(f"\\n📊 Ingestion complete:")
    logger.info(f"   ✅ Ingested: {ingested} blog posts")
    logger.info(f"   ⏭️  Skipped: {skipped} (already existed)")

    cursor.close()
    conn.close()

if __name__ == '__main__':
    ingest_blog_posts()

```

**Run ingestion:**

```bash
python scripts/ingest_content.py

```

**Validation:**

```sql
-- Check ingested content
SELECT
  COUNT(*) as total_blogs,
  COUNT(DISTINCT id) as unique_blogs
FROM josh_content;

-- Check chunks
SELECT
  c.title,
  COUNT(ch.id) as num_chunks
FROM josh_content c
LEFT JOIN content_chunks ch ON c.id = ch.content_id
GROUP BY c.id, c.title
ORDER BY c.published_date DESC
LIMIT 10;

```

**Expected:** 20-50 blogs ingested, 500-2000 chunks created

---

### **Day 5: Embedding Generation & Vector Index**

### **Step 3.1: Generate Embeddings**

```python
# src/data_pipeline/embedding_generator.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm
import numpy as np

load_dotenv()

class EmbeddingGenerator:
    """Generate embeddings for content chunks using e5-base-v2"""

    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        logger.info(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = 768  # e5-base-v2 dimension
        logger.success(f"✅ Model loaded (dimension: {self.dimension})")

    def generate_batch(self, texts: list) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        # e5 models require "query: " or "passage: " prefix
        # For content chunks, use "passage: " prefix
        prefixed_texts = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        # For queries, use "query: " prefix
        prefixed_query = f"query: {query}"
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True
        )
        return embedding

def generate_embeddings_for_all_chunks():
    """Generate embeddings for all content chunks in database"""

    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv("DATABASE_HOST"),
        port=os.getenv("DATABASE_PORT"),
        database=os.getenv("DATABASE_NAME"),
        user=os.getenv("DATABASE_USER"),
        password=os.getenv("DATABASE_PASSWORD")
    )
    cursor = conn.cursor()

    # Initialize embedding generator
    generator = EmbeddingGenerator()

    # Fetch all chunks without embeddings
    cursor.execute("""
        SELECT id, chunk_text
        FROM content_chunks
        WHERE embedding_vector IS NULL
        ORDER BY id
    """)

    chunks = cursor.fetchall()
    logger.info(f"📊 Found {len(chunks)} chunks to embed")

    if len(chunks) == 0:
        logger.warning("⚠️  No chunks found. Run ingest_content.py first.")
        return

    # Process in batches
    BATCH_SIZE = 32
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    processed = 0
    failed = 0

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), total=total_batches, desc="Generating embeddings"):
        batch = chunks[i:i+BATCH_SIZE]

        try:
            # Extract IDs and texts
            chunk_ids = [chunk[0] for chunk in batch]
            texts = [chunk[1] for chunk in batch]

            # Generate embeddings
            embeddings = generator.generate_batch(texts)

            # Update database
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                cursor.execute("""
                    UPDATE content_chunks
                    SET embedding_vector = %s
                    WHERE id = %s
                """, (embedding.tolist(), chunk_id))

            conn.commit()
            processed += len(batch)

        except Exception as e:
            logger.error(f"❌ Error processing batch {i}: {e}")
            conn.rollback()
            failed += len(batch)
            continue

    logger.info(f"\\n✨ Embedding generation complete:")
    logger.info(f"   ✅ Processed: {processed} chunks")
    logger.info(f"   ❌ Failed: {failed} chunks")

    # Create ivfflat index for fast similarity search
    logger.info("\\n📊 Creating vector similarity index...")

    try:
        # Determine number of lists (rule of thumb: sqrt(num_rows))
        num_lists = max(10, int(np.sqrt(len(chunks))))

        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_content_chunks_embedding
            ON content_chunks
            USING ivfflat (embedding_vector vector_cosine_ops)
            WITH (lists = {num_lists});
        """)
        conn.commit()
        logger.success(f"✅ Vector index created (lists={num_lists})")
    except Exception as e:
        logger.error(f"❌ Error creating index: {e}")

    cursor.close()
    conn.close()

if __name__ == '__main__':
    generate_embeddings_for_all_chunks()

```

**Run embedding generation:**

```bash
python scripts/generate_embeddings.py

```

**Validation:**

```sql
-- Check embeddings generated
SELECT
  COUNT(*) as total_chunks,
  COUNT(embedding_vector) as chunks_with_embeddings,
  COUNT(*) - COUNT(embedding_vector) as missing_embeddings
FROM content_chunks;

-- Check index created
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'content_chunks'
AND indexname = 'idx_content_chunks_embedding';

```

**Expected:**

- ✅ All chunks have embeddings
- ✅ `idx_content_chunks_embedding` index exists

---

## Week 2: RAG Implementation & Testing

### **Day 1-2: Build RAG Retriever**

### **Step 4.1: Implement RAG Retriever**

```python
# src/rag/retriever.py
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from loguru import logger
from src.data_pipeline.embedding_generator import EmbeddingGenerator

load_dotenv()

@dataclass
class RetrievalResult:
    """Single retrieval result"""
    chunk_id: int
    content_id: int
    chunk_text: str
    similarity: float
    extracted_insights: Dict
    metadata: Dict

class RAGRetriever:
    """Retrieve relevant content chunks using vector similarity"""

    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.top_k = int(os.getenv("RAG_TOP_K", 10))
        self.confidence_threshold = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", 0.75))

        # Database connection
        self.conn = psycopg2.connect(
            host=os.getenv("DATABASE_HOST"),
            port=os.getenv("DATABASE_PORT"),
            database=os.getenv("DATABASE_NAME"),
            user=os.getenv("DATABASE_USER"),
            password=os.getenv("DATABASE_PASSWORD")
        )

    def construct_query(self, quiz_response: Dict) -> str:
        """Construct semantic search query from quiz response"""

        # Extract quiz fields
        professions = quiz_response.get('profession', [])
        budgets = quiz_response.get('budget', [])
        portability = quiz_response.get('portability', '')
        use_cases = quiz_response.get('use_case', [])
        screen_sizes = quiz_response.get('screen_size', [])

        # Build natural language query
        query_parts = ["What laptops does Josh recommend for"]

        if professions:
            query_parts.append(f"{', '.join(professions)}")

        if use_cases:
            query_parts.append(f"who need {', '.join(use_cases)}")

        if budgets:
            query_parts.append(f"with {', '.join(budgets)} budget")

        if portability:
            portability_map = {
                'light': 'lightweight and portable',
                'somewhat': 'moderate portability',
                'performance': 'prioritizing performance over portability'
            }
            query_parts.append(f"and prefers {portability_map.get(portability, portability)}")

        if screen_sizes:
            query_parts.append(f"with {' or '.join(screen_sizes)} screen")

        query = " ".join(query_parts) + "?"

        logger.info(f"🔍 Query: {query}")
        return query

    def retrieve(
        self,
        quiz_response: Dict,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant content chunks"""

        # Construct query
        query_text = self.construct_query(quiz_response)

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query_text)

        # Search database using pgvector
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        # Extract use case tags for filtering
        use_case_tags = quiz_response.get('use_case', [])

        if use_case_tags:
            # Filter by use case tags + similarity
            cursor.execute("""
                SELECT
                    ch.id as chunk_id,
                    ch.content_id,
                    ch.chunk_text,
                    ch.extracted_insights,
                    c.title as content_title,
                    c.url as content_url,
                    c.use_case_tags,
                    1 - (ch.embedding_vector <=> %s::vector) AS similarity
                FROM content_chunks ch
                JOIN josh_content c ON ch.content_id = c.id
                WHERE c.use_case_tags && %s  -- Overlapping array
                ORDER BY ch.embedding_vector <=> %s::vector
                LIMIT %s;
            """, (
                query_embedding.tolist(),
                use_case_tags,
                query_embedding.tolist(),
                top_k or self.top_k
            ))
        else:
            # No use case filter, just similarity
            cursor.execute("""
                SELECT
                    ch.id as chunk_id,
                    ch.content_id,
                    ch.chunk_text,
                    ch.extracted_insights,
                    c.title as content_title,
                    c.url as content_url,
                    c.use_case_tags,
                    1 - (ch.embedding_vector <=> %s::vector) AS similarity
                FROM content_chunks ch
                JOIN josh_content c ON ch.content_id = c.id
                ORDER BY ch.embedding_vector <=> %s::vector
                LIMIT %s;
            """, (
                query_embedding.tolist(),
                query_embedding.tolist(),
                top_k or self.top_k
            ))

        rows = cursor.fetchall()
        cursor.close()

        # Convert to RetrievalResult objects
        results = [
            RetrievalResult(
                chunk_id=row['chunk_id'],
                content_id=row['content_id'],
                chunk_text=row['chunk_text'],
                similarity=float(row['similarity']),
                extracted_insights=row['extracted_insights'] or {},
                metadata={
                    'content_title': row['content_title'],
                    'content_url': row['content_url'],
                    'use_case_tags': row['use_case_tags']
                }
            )
            for row in rows
        ]

        # Log retrieval stats
        if results:
            logger.info(f"📊 Retrieved {len(results)} chunks")
            logger.info(f"   Top similarity: {results[0].similarity:.3f}")
            logger.info(f"   Lowest similarity: {results[-1].similarity:.3f}")

            # Check confidence
            if results[0].similarity < self.confidence_threshold:
                logger.warning(f"⚠️  Low confidence ({results[0].similarity:.3f} < {self.confidence_threshold})")
        else:
            logger.warning("⚠️  No results found")

        return results

    def is_confident(self, results: List[RetrievalResult]) -> bool:
        """Check if retrieval is confident enough"""
        if not results:
            return False
        return results[0].similarity >= self.confidence_threshold

# Test retrieval
if __name__ == '__main__':
    retriever = RAGRetriever()

    # Test query
    quiz_response = {
        'profession': ['Student', 'Professional'],
        'use_case': ['video_editing'],
        'budget': ['value'],
        'portability': 'light'
    }

    results = retriever.retrieve(quiz_response)

    print(f"\\n🔍 Query Results:")
    for i, result in enumerate(results[:5], 1):
        print(f"\\n{i}. Similarity: {result.similarity:.3f}")
        print(f"   Title: {result.metadata['content_title']}")
        print(f"   Insights: {result.extracted_insights}")
        print(f"   Text preview: {result.chunk_text[:200]}...")

    print(f"\\n✅ Confident: {retriever.is_confident(results)}")

```

**Test retrieval:**

```bash
python src/rag/retriever.py

```

**Expected output:**

- ✅ Query constructed correctly
- ✅ 10 results retrieved
- ✅ Top similarity >0.75
- ✅ Relevant content chunks returned

---

### **Step 4.2: Create Test Dataset for Validation**

```python
# scripts/create_test_dataset.py
import json
from pathlib import Path

# Create test queries with expected results
test_dataset = [
    {
        "test_id": 1,
        "description": "Student video editing on budget",
        "quiz_response": {
            "profession": ["Student"],
            "use_case": ["video_editing"],
            "budget": ["budget"],
            "portability": "light"
        },
        "expected_products": ["MacBook Air M2", "Dell XPS 13"],
        "expected_ranking": 1,  # Should find Josh's #1 pick
        "min_similarity": 0.80
    },
    {
        "test_id": 2,
        "description": "Professional gaming high-end",
        "quiz_response": {
            "profession": ["Professional", "Gamer"],
            "use_case": ["gaming", "content_creation"],
            "budget": ["premium"],
            "portability": "performance"
        },
        "expected_products": ["Legion 7i", "ROG Zephyrus"],
        "expected_ranking": 1,
        "min_similarity": 0.75
    },
    {
        "test_id": 3,
        "description": "Developer lightweight portable",
        "quiz_response": {
            "profession": ["Developer"],
            "use_case": ["programming"],
            "budget": ["value"],
            "portability": "light"
        },
        "expected_products": ["MacBook Pro 14", "ThinkPad X1 Carbon"],
        "expected_ranking": 1,
        "min_similarity": 0.75
    },
    # Add 20-50 more test cases covering different scenarios
]

# Save test dataset
test_file = Path('data/test/quiz_samples.json')
test_file.parent.mkdir(parents=True, exist_ok=True)

with open(test_file, 'w') as f:
    json.dump(test_dataset, f, indent=2)

print(f"✅ Created test dataset with {len(test_dataset)} test cases")
print(f"   Saved to: {test_file}")

```

**Run:**

```bash
python scripts/create_test_dataset.py

```

**Validation:** ✅ `data/test/quiz_samples.json` created with 20-50 test cases

---

### **Step 4.3: Validate Retrieval Accuracy**

```python
# tests/test_retrieval.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
from src.rag.retriever import RAGRetriever
from loguru import logger
from tabulate import tabulate

def validate_retrieval_accuracy():
    """Validate RAG retrieval accuracy on test dataset"""

    # Load test dataset
    with open('data/test/quiz_samples.json', 'r') as f:
        test_cases = json.load(f)

    retriever = RAGRetriever()

    results = []
    passed = 0
    failed = 0

    for test_case in test_cases:
        logger.info(f"\\n{'='*60}")
        logger.info(f"Test #{test_case['test_id']}: {test_case['description']}")

        # Run retrieval
        retrieval_results = retriever.retrieve(test_case['quiz_response'], top_k=5)

        if not retrieval_results:
            logger.error("❌ No results retrieved")
            failed += 1
            results.append({
                'test_id': test_case['test_id'],
                'passed': False,
                'top_similarity': 0.0,
                'found_expected_product': False,
                'notes': 'No results'
            })
            continue

        # Check top similarity
        top_similarity = retrieval_results[0].similarity
        similarity_check = top_similarity >= test_case['min_similarity']

        # Check if expected products found
        found_products = []
        for result in retrieval_results:
            for expected_product in test_case['expected_products']:
                if expected_product.lower() in result.chunk_text.lower():
                    found_products.append(expected_product)

        product_check = len(found_products) > 0

        # Check if expected ranking found
        ranking_check = False
        if retrieval_results[0].extracted_insights:
            ranking = retrieval_results[0].extracted_insights.get('ranking')
            ranking_check = ranking == test_case['expected_ranking']

        # Overall pass/fail
        test_passed = similarity_check and product_check

        if test_passed:
            logger.success(f"✅ PASSED")
            passed += 1
        else:
            logger.error(f"❌ FAILED")
            failed += 1

        logger.info(f"   Similarity: {top_similarity:.3f} (threshold: {test_case['min_similarity']})")
        logger.info(f"   Expected products: {test_case['expected_products']}")
        logger.info(f"   Found products: {found_products}")
        logger.info(f"   Ranking check: {ranking_check}")

        # Show top result
        logger.info(f"\\n   Top result:")
        logger.info(f"   Title: {retrieval_results[0].metadata['content_title']}")
        logger.info(f"   Text preview: {retrieval_results[0].chunk_text[:200]}...")

        results.append({
            'test_id': test_case['test_id'],
            'description': test_case['description'],
            'passed': test_passed,
            'top_similarity': top_similarity,
            'found_products': found_products,
            'ranking_correct': ranking_check
        })

    # Summary
    logger.info(f"\\n{'='*60}")
    logger.info(f"📊 VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total tests: {len(test_cases)}")
    logger.info(f"✅ Passed: {passed} ({passed/len(test_cases)*100:.1f}%)")
    logger.info(f"❌ Failed: {failed} ({failed/len(test_cases)*100:.1f}%)")

    # Print detailed results table
    table_data = [
        [
            r['test_id'],
            r['description'][:40],
            '✅' if r['passed'] else '❌',
            f"{r['top_similarity']:.3f}",
            ', '.join(r['found_products'][:2]) if r['found_products'] else 'None'
        ]
        for r in results
    ]

    print("\\n" + tabulate(
        table_data,
        headers=['ID', 'Description', 'Pass', 'Similarity', 'Found Products'],
        tablefmt='grid'
    ))

    # Calculate metrics
    avg_similarity = sum(r['top_similarity'] for r in results) / len(results)
    logger.info(f"\\n📈 Metrics:")
    logger.info(f"   Average top similarity: {avg_similarity:.3f}")
    logger.info(f"   Test accuracy: {passed/len(test_cases)*100:.1f}%")

    # Success criteria
    if passed / len(test_cases) >= 0.85 and avg_similarity >= 0.80:
        logger.success("\\n🎉 VALIDATION PASSED! Ready to proceed.")
        return True
    else:
        logger.warning("\\n⚠️  VALIDATION NEEDS IMPROVEMENT")
        logger.warning("   Target: 85%+ accuracy, 0.80+ avg similarity")
        logger.warning("\\n   Next steps:")
        logger.warning("   1. Review failed test cases")
        logger.warning("   2. Improve content chunking")
        logger.warning("   3. Add more blog content")
        logger.warning("   4. Adjust similarity thresholds")
        return False

if __name__ == '__main__':
    validate_retrieval_accuracy()

```

**Run validation:**

```bash
python tests/test_retrieval.py

```

**Expected output:**

```
✅ Passed: 17/20 (85.0%)
Average top similarity: 0.82
🎉 VALIDATION PASSED!

```

**🚨 CRITICAL CHECKPOINT:**

- **DO NOT proceed to Week 3 unless:**
    - ✅ Test accuracy >85%
    - ✅ Average similarity >0.80
    - ✅ Manual review confirms results are relevant

**If validation fails:**

1. Review failed test cases manually
2. Improve content extraction (Step 2.2)
3. Add more detailed blog content
4. Adjust chunking strategy
5. Re-run validation

---

## Week 2 (continued) - Day 3-5: Implement Ranking & API

### **Day 3: Implement Simple Scoring System**

### **Step 5.1: Build Simple Ranker (Phase 1 - No ML)**

```python
# src/rag/ranker.py
from typing import List, Dict
from dataclasses import dataclass
from loguru import logger
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RankedConfig:
    """Ranked configuration with score and reasoning"""
    config_id: int
    product_id: int
    product_title: str
    confidence_score: float
    josh_score: float
    spec_score: float
    explanation: str
    josh_quote: str
    josh_ranking: int
    pros: List[str]
    cons: List[str]
    who_is_this_for: str

class SimpleRanker:
    """Phase 1: Simple rule-based ranking (Josh 70% + Specs 30%)"""

    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv("DATABASE_HOST"),
            port=os.getenv("DATABASE_PORT"),
            database=os.getenv("DATABASE_NAME"),
            user=os.getenv("DATABASE_USER"),
            password=os.getenv("DATABASE_PASSWORD")
        )

    def rank_configs(
        self,
        retrieval_results: List,
        quiz_response: Dict,
        top_k: int = 5
    ) -> List[RankedConfig]:
        """Rank configs using simple scoring: Josh 70% + Specs 30%"""

        # Extract config IDs from retrieval results
        config_candidates = self._extract_config_ids(retrieval_results)

        if not config_candidates:
            logger.warning("⚠️  No configs found in retrieval results")
            return []

        # Fetch config details from database
        configs = self._fetch_config_details(config_candidates)

        # Score each config
        ranked_configs = []
        for config in configs:
            # 1. Josh score (70% weight)
            josh_score = self._calculate_josh_score(config, retrieval_results)

            # 2. Spec match score (30% weight)
            spec_score = self._calculate_spec_score(config, quiz_response)

            # 3. Final score
            final_score = (josh_score * 0.70) + (spec_score * 0.30)

            # 4. Get explanation
            explanation = self._generate_explanation(config, retrieval_results, quiz_response)

            ranked_configs.append(RankedConfig(
                config_id=config['id'],
                product_id=config['product_id'],
                product_title=config['product_title'],
                confidence_score=final_score,
                josh_score=josh_score,
                spec_score=spec_score,
                explanation=explanation,
                josh_quote=config.get('josh_quote', ''),
                josh_ranking=config.get('josh_ranking', 0),
                pros=config.get('pros', []),
                cons=config.get('cons', []),
                who_is_this_for=config.get('who_is_this_for', '')
            ))

        # Sort by final score
        ranked_configs.sort(key=lambda x: x.confidence_score, reverse=True)

        logger.info(f"📊 Ranked {len(ranked_configs)} configs")
        if ranked_configs:
            logger.info(f"   Top config: {ranked_configs[0].product_title} (score: {ranked_configs[0].confidence_score:.3f})")

        return ranked_configs[:top_k]

    def _extract_config_ids(self, retrieval_results: List) -> List[Dict]:
        """Extract config IDs and Josh's opinions from retrieval results"""
        config_data = {}

        for result in retrieval_results:
            insights = result.extracted_insights
            products_mentioned = insights.get('products_mentioned', [])

            for product in products_mentioned:
                product_name = product['name']

                # Try to match product name to config in database
                config_id = self._match_product_to_config(product_name)

                if config_id and config_id not in config_data:
                    # Store Josh's opinion data
                    config_data[config_id] = {
                        'config_id': config_id,
                        'josh_ranking': insights.get('ranking', 0),
                        'josh_quote': insights.get('quotes', [''])[0] if insights.get('quotes') else '',
                        'similarity': result.similarity,
                        'pros': insights.get('pros', []),
                        'cons': insights.get('cons', []),
                        'who_is_this_for': insights.get('who_is_this_for', ''),
                        'recommendation_type': insights.get('recommendation_type', '')
                    }

        return list(config_data.values())

    def _match_product_to_config(self, product_name: str) -> int:
        """Match product name to config ID in database"""
        cursor = self.conn.cursor()

        # Fuzzy match product name
        cursor.execute("""
            SELECT c.id
            FROM configs c
            JOIN products p ON c.product_id = p.id
            WHERE
                p.title ILIKE %s
                AND c.is_active = true
                AND p.is_active = true
            ORDER BY p.created_at DESC
            LIMIT 1
        """, (f"%{product_name}%",))

        result = cursor.fetchone()
        cursor.close()

        return result[0] if result else None

    def _fetch_config_details(self, config_candidates: List[Dict]) -> List[Dict]:
        """Fetch full config details from database"""
        config_ids = [c['config_id'] for c in config_candidates]

        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute("""
            SELECT
                c.id,
                c.product_id,
                c.public_id,
                p.title as product_title,
                c.current_price,
                c.fallback_msrp,
                c.classification,
                p.weight,
                p.screen_size,
                p.use_cases
            FROM configs c
            JOIN products p ON c.product_id = p.id
            WHERE c.id = ANY(%s)
        """, (config_ids,))

        configs = cursor.fetchall()
        cursor.close()

        # Merge with Josh's opinion data
        config_map = {c['config_id']: c for c in config_candidates}
        for config in configs:
            josh_data = config_map.get(config['id'], {})
            config.update(josh_data)

        return configs

    def _calculate_josh_score(self, config: Dict, retrieval_results: List) -> float:
        """Calculate Josh score (0-1)"""
        josh_ranking = config.get('josh_ranking', 0)
        similarity = config.get('similarity', 0.0)

        # If Josh explicitly ranked this config
        if josh_ranking > 0:
            # #1 pick = 1.0, #2 = 0.9, #3 = 0.8, etc.
            rank_score = max(0.0, 1.0 - (josh_ranking - 1) * 0.1)

            # Boost by similarity (how relevant the content chunk was)
            final_score = rank_score * 0.8 + similarity * 0.2

            return min(1.0, final_score)

        # Josh mentioned but didn't rank - use similarity
        elif similarity > 0:
            return similarity * 0.7  # Reduce weight for non-ranked mentions

        # Not mentioned by Josh
        else:
            return 0.0

    def _calculate_spec_score(self, config: Dict, quiz_response: Dict) -> float:
        """Calculate spec match score (0-1)"""
        score = 0.0
        checks = 0

        # 1. Budget match
        budget_tiers = quiz_response.get('budget', [])
        if budget_tiers and config.get('current_price'):
            price = float(config['current_price'])

            budget_ranges = {
                'budget': (0, 700),
                'value': (700, 1500),
                'premium': (1500, 10000)
            }

            for tier in budget_tiers:
                min_price, max_price = budget_ranges.get(tier, (0, 10000))
                if min_price <= price <= max_price:
                    score += 1.0
                    break
            checks += 1

        # 2. Portability/weight match
        portability = quiz_response.get('portability', '')
        if portability and config.get('weight'):
            weight = float(config['weight'])

            portability_thresholds = {
                'light': 4.0,      # < 4 lbs
                'somewhat': 6.0,   # < 6 lbs
                'performance': 10.0  # Any weight
            }

            threshold = portability_thresholds.get(portability, 10.0)
            if weight <= threshold:
                score += 1.0
            checks += 1

        # 3. Screen size match
        screen_sizes = quiz_response.get('screen_size', [])
        if screen_sizes and config.get('screen_size'):
            config_screen = config['screen_size']

            for requested_size in screen_sizes:
                # Parse size range (e.g., "13-14 inches")
                if '-' in requested_size:
                    min_size, max_size = map(lambda x: float(x.split()[0]), requested_size.split('-'))
                    if min_size <= config_screen <= max_size:
                        score += 1.0
                        break
            checks += 1

        # 4. Use case match
        use_cases = quiz_response.get('use_case', [])
        if use_cases and config.get('use_cases'):
            config_use_cases = config['use_cases']
            overlap = set(use_cases) & set(config_use_cases)
            if overlap:
                score += len(overlap) / len(use_cases)
            checks += 1

        # Normalize by number of checks
        return score / checks if checks > 0 else 0.0

    def _generate_explanation(
        self,
        config: Dict,
        retrieval_results: List,
        quiz_response: Dict
    ) -> str:
        """Generate human-readable explanation"""
        explanations = []

        # 1. Josh's opinion (highest priority)
        if config.get('josh_ranking'):
            explanations.append(f"Josh's #{config['josh_ranking']} pick for this use case")
        elif config.get('josh_quote'):
            explanations.append(f"Josh says: \\"{config['josh_quote'][:100]}...\\"")

        # 2. Spec matches
        budget_tiers = quiz_response.get('budget', [])
        if budget_tiers:
            explanations.append(f"Matches your {', '.join(budget_tiers)} budget")

        portability = quiz_response.get('portability', '')
        if portability and config.get('weight'):
            portability_labels = {
                'light': 'lightweight and portable',
                'somewhat': 'balanced portability',
                'performance': 'performance-focused'
            }
            explanations.append(f"Good for {portability_labels.get(portability, portability)}")

        use_cases = quiz_response.get('use_case', [])
        if use_cases:
            explanations.append(f"Great for {', '.join(use_cases)}")

        return ". ".join(explanations) + "."

# Test ranker
if __name__ == '__main__':
    from src.rag.retriever import RAGRetriever

    retriever = RAGRetriever()
    ranker = SimpleRanker()

    # Test query
    quiz_response = {
        'profession': ['Student'],
        'use_case': ['video_editing'],
        'budget': ['value'],
        'portability': 'light'
    }

    # Retrieve
    retrieval_results = retriever.retrieve(quiz_response)

    # Rank
    ranked_configs = ranker.rank_configs(retrieval_results, quiz_response, top_k=5)

    print(f"\\n🏆 Top 5 Recommendations:")
    for i, config in enumerate(ranked_configs, 1):
        print(f"\\n{i}. {config.product_title}")
        print(f"   Confidence: {config.confidence_score:.3f} (Josh: {config.josh_score:.2f}, Spec: {config.spec_score:.2f})")
        print(f"   Explanation: {config.explanation}")
        if config.josh_quote:
            print(f"   Josh's quote: \\"{config.josh_quote}\\"")

```

**Test ranker:**

```bash
python src/rag/ranker.py

```

**Expected output:**

- ✅ Top 5 configs ranked
- ✅ Josh's #1 pick has highest score
- ✅ Explanations include Josh's quotes
- ✅ Spec matches correctly identified

---

### **Day 4-5: Build FastAPI Service**

### **Step 6.1: Create API Server**

```python
# src/api/server.py
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time
from loguru import logger
import os
from dotenv import load_dotenv

from src.rag.retriever import RAGRetriever
from src.rag.ranker import SimpleRanker

load_dotenv()

app = FastAPI(
    title="Just Josh RAG Recommendation Service",
    description="Josh-powered laptop recommendations using RAG",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
retriever = RAGRetriever()
ranker = SimpleRanker()

# API Key authentication
API_KEY = os.getenv("API_KEY", "your-api-key-here")

def verify_api_key(authorization: str = Header(None)):
    """Verify API key in Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    api_key = authorization.replace("Bearer ", "")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key

# Request/Response Models
class QuizResponse(BaseModel):
    profession: List[str] = Field(..., example=["Student", "Professional"])
    use_case: List[str] = Field(..., example=["video_editing"])
    budget: List[str] = Field(..., example=["value"])
    portability: str = Field(..., example="light")
    screen_size: Optional[List[str]] = Field(None, example=["13-14 inches"])

class RecommendationOptions(BaseModel):
    top_k: int = Field(5, ge=1, le=20)
    include_explanations: bool = True
    min_confidence: float = Field(0.7, ge=0.0, le=1.0)

class RecommendationRequest(BaseModel):
    quiz_response: QuizResponse
    options: Optional[RecommendationOptions] = RecommendationOptions()

class RecommendationItem(BaseModel):
    config_id: int
    public_config_id: str
    product_title: str
    confidence_score: float
    josh_score: float
    spec_score: float
    explanation: str
    josh_recommendation: Optional[Dict] = None
    tradeoffs: Optional[Dict] = None

class RecommendationResponse(BaseModel):
    success: bool
    recommendations: List[RecommendationItem]
    meta: Dict

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "rag-recommendation-service",
        "version": "1.0.0"
    }

@app.post("/api/v1/recommendations/predict")
async def predict_recommendations(
    request: RecommendationRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate recommendations based on quiz response

    Returns Josh-powered recommendations with explanations
    """
    start_time = time.time()

    try:
        # 1. Retrieve relevant content chunks
        retrieval_start = time.time()
        retrieval_results = retriever.retrieve(request.quiz_response.dict())
        retrieval_time = int((time.time() - retrieval_start) * 1000)

        # Check if confident enough
        if not retriever.is_confident(retrieval_results):
            logger.warning("⚠️  Low RAG confidence - returning fallback")
            # TODO: Implement spec-based fallback
            return {
                "success": True,
                "recommendations": [],
                "meta": {
                    "model_version": "1.0.0",
                    "recommendation_source": "low_confidence_fallback",
                    "retrieval_time_ms": retrieval_time,
                    "total_time_ms": int((time.time() - start_time) * 1000)
                }
            }

        # 2. Rank configs
        ranking_start = time.time()
        ranked_configs = ranker.rank_configs(
            retrieval_results,
            request.quiz_response.dict(),
            top_k=request.options.top_k
        )
        ranking_time = int((time.time() - ranking_start) * 1000)

        # 3. Format response
        recommendations = []
        for config in ranked_configs:
            recommendations.append(RecommendationItem(
                config_id=config.config_id,
                public_config_id=config.public_id,
                product_title=config.product_title,
                confidence_score=config.confidence_score,
                josh_score=config.josh_score,
                spec_score=config.spec_score,
                explanation=config.explanation,
                josh_recommendation={
                    "ranking": config.josh_ranking,
                    "quote": config.josh_quote,
                    "who_is_this_for": config.who_is_this_for
                } if config.josh_quote else None,
                tradeoffs={
                    "pros": config.pros,
                    "cons": config.cons
                } if config.pros or config.cons else None
            ))

        total_time = int((time.time() - start_time) * 1000)

        logger.success(f"✅ Returned {len(recommendations)} recommendations ({total_time}ms)")

        return RecommendationResponse(
            success=True,
            recommendations=recommendations,
            meta={
                "model_version": "1.0.0",
                "josh_content_version": "2025-01-21",
                "recommendation_source": "josh_rag_primary",
                "retrieval_time_ms": retrieval_time,
                "ranking_time_ms": ranking_time,
                "total_time_ms": total_time
            }
        )

    except Exception as e:
        logger.error(f"❌ Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
        log_level="info"
    )

```

**Start API server:**

```bash
python src/api/server.py

```

**Test API:**

```bash
# Health check
curl <http://localhost:8000/health>

# Get recommendations
curl -X POST <http://localhost:8000/api/v1/recommendations/predict> \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer your-api-key-here" \\
  -d '{
    "quiz_response": {
      "profession": ["Student"],
      "use_case": ["video_editing"],
      "budget": ["value"],
      "portability": "light"
    }
  }'

```

**Expected:**

- ✅ API returns 200 OK
- ✅ 5 recommendations returned
- ✅ Josh's quotes included
- ✅ <200ms latency

---

## ✅ **Phase 1 Completion Checklist**

Before deploying to production, verify:

### **Data Quality**

- [ ]  20-50 blog posts ingested
- [ ]  500-2000 content chunks created
- [ ]  All chunks have embeddings (768d)
- [ ]  Vector index created successfully

### **Retrieval Quality**

- [ ]  Test accuracy >85% (run `tests/test_retrieval.py`)
- [ ]  Average similarity >0.80
- [ ]  Manual review of 20-50 queries confirms relevance

### **API Functionality**

- [ ]  Health check endpoint works
- [ ]  Recommendations endpoint returns results
- [ ]  API key authentication works
- [ ]  Latency <200ms P95

### **Safety Rails**

- [ ]  Confidence threshold enforced (0.75+)
- [ ]  Spec-based fallback implemented (TODO)
- [ ]  Error handling for edge cases
- [ ]  Logging enabled for debugging

---

## 🚀 **Next Steps: Week 3 Integration & Deployment**

Continue in **Part 2** of this guide:

- Week 3: Node.js integration, Redis caching, A/B testing
- Production deployment steps
- Monitoring & metrics
- Phase 2: Adding ML (if needed)