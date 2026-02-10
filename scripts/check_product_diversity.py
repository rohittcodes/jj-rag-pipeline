"""Quick script to check product diversity in database."""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'josh_rag'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

print("\n=== PRODUCT DIVERSITY CHECK ===\n")

# Check unique products in laptop_configs
print("1. Unique products in configs:")
cursor.execute("""
    SELECT DISTINCT brand, model 
    FROM configs 
    ORDER BY brand, model
    LIMIT 50
""")
products = cursor.fetchall()
print(f"   Found {len(products)} unique products:")
for brand, model in products:
    print(f"   - {brand} {model}")

# Check how many content chunks have config_ids
print("\n2. Content chunks with product links:")
cursor.execute("""
    SELECT 
        COUNT(*) as total_chunks,
        COUNT(DISTINCT config_id) as unique_products_linked,
        COUNT(CASE WHEN config_id IS NOT NULL THEN 1 END) as chunks_with_products
    FROM content_chunks
""")
stats = cursor.fetchone()
print(f"   Total chunks: {stats[0]}")
print(f"   Chunks with product links: {stats[2]}")
print(f"   Unique products linked: {stats[1]}")

# Check which products are most frequently mentioned
print("\n3. Most frequently mentioned products:")
cursor.execute("""
    SELECT 
        lc.brand,
        lc.model,
        COUNT(*) as mention_count
    FROM content_chunks cc
    JOIN configs lc ON cc.config_id = lc.id
    GROUP BY lc.brand, lc.model
    ORDER BY mention_count DESC
    LIMIT 20
""")
top_products = cursor.fetchall()
for brand, model, count in top_products:
    print(f"   - {brand} {model}: {count} chunks")

# Check content sources
print("\n4. Content sources:")
cursor.execute("""
    SELECT 
        source_type,
        COUNT(DISTINCT content_id) as content_count,
        COUNT(*) as chunk_count
    FROM content_chunks
    GROUP BY source_type
""")
sources = cursor.fetchall()
for source_type, content_count, chunk_count in sources:
    print(f"   - {source_type}: {content_count} articles, {chunk_count} chunks")

cursor.close()
conn.close()

print("\n=== END ===\n")
