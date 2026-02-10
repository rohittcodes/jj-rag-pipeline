"""
Unified database setup script.
Creates all tables and indexes for the RAG pipeline.

Usage:
    python scripts/setup.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
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


def setup_main_database():
    """Setup main database schema for blog content and configs."""
    print("\n[*] Setting up main database schema...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Enable pgvector extension
        print("[*] Enabling pgvector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create josh_content table
        print("[*] Creating josh_content table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS josh_content (
                id SERIAL PRIMARY KEY,
                sanity_id VARCHAR(255) UNIQUE,
                title TEXT NOT NULL,
                content_type VARCHAR(50),
                raw_content TEXT NOT NULL,
                url TEXT,
                publish_date DATE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT[],
                metadata JSONB
            );
        """)
        
        # Create content_chunks table
        print("[*] Creating content_chunks table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_chunks (
                id SERIAL PRIMARY KEY,
                content_id INTEGER REFERENCES josh_content(id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                section_title TEXT,
                metadata JSONB,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create configs table (local product data)
        print("[*] Creating configs table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configs (
                id SERIAL PRIMARY KEY,
                config_id INTEGER UNIQUE NOT NULL,
                product_id INTEGER,
                product_name TEXT NOT NULL,
                brand TEXT,
                model TEXT,
                price DECIMAL(10, 2),
                specs JSONB,
                test_data JSONB,
                rating DECIMAL(3, 2),
                josh_context TEXT,
                last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create rag_query_logs table
        print("[*] Creating rag_query_logs table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_query_logs (
                id SERIAL PRIMARY KEY,
                query_text TEXT NOT NULL,
                quiz_response JSONB,
                top_results JSONB,
                confidence_score FLOAT,
                recommendation_source VARCHAR(50),
                latency_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create test_data_chunks table
        print("[*] Creating test_data_chunks table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_data_chunks (
                id SERIAL PRIMARY KEY,
                config_id INTEGER NOT NULL,
                test_type VARCHAR(100),
                test_description TEXT,
                benchmark_results JSONB,
                chunk_text TEXT NOT NULL,
                embedding vector(768),
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_config FOREIGN KEY (config_id) 
                    REFERENCES configs(config_id) ON DELETE CASCADE
            );
        """)
        
        # Create product_spec_chunks table
        print("[*] Creating product_spec_chunks table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS product_spec_chunks (
                id SERIAL PRIMARY KEY,
                config_id INTEGER NOT NULL REFERENCES configs(config_id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes
        print("[*] Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_josh_content_publish_date ON josh_content(publish_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_josh_content_updated_at ON josh_content(updated_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_chunks_content_id ON content_chunks(content_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_configs_config_id ON configs(config_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_configs_product_id ON configs(product_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_query_logs_created_at ON rag_query_logs(created_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_data_chunks_config_id ON test_data_chunks(config_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_data_chunks_test_type ON test_data_chunks(test_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_spec_chunks_config_id ON product_spec_chunks(config_id);")
        
        conn.commit()
        print("[+] Main database schema created successfully!")
        
    except Exception as e:
        print(f"[-] Error setting up main database: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()
    
    return True


def setup_youtube_schema():
    """Setup YouTube-specific schema."""
    print("\n[*] Setting up YouTube schema...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Create youtube_content table
        print("[*] Creating youtube_content table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS youtube_content (
                id SERIAL PRIMARY KEY,
                video_id VARCHAR(255) UNIQUE NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                description TEXT,
                publish_date DATE,
                thumbnail_url TEXT,
                transcript_type VARCHAR(50),
                language VARCHAR(100),
                full_transcript TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create youtube_chunks table
        print("[*] Creating youtube_chunks table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS youtube_chunks (
                id SERIAL PRIMARY KEY,
                youtube_content_id INTEGER REFERENCES youtube_content(id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_time FLOAT,
                end_time FLOAT,
                metadata JSONB,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes
        print("[*] Creating YouTube indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_youtube_content_video_id ON youtube_content(video_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_youtube_content_publish_date ON youtube_content(publish_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_youtube_chunks_content_id ON youtube_chunks(youtube_content_id);")
        
        conn.commit()
        print("[+] YouTube schema created successfully!")
        
    except Exception as e:
        print(f"[-] Error setting up YouTube schema: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()
    
    return True


def create_vector_indexes():
    """Create HNSW indexes for vector similarity search."""
    print("\n[*] Creating vector indexes...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Create HNSW index for blog chunks
        print("[*] Creating HNSW index for content_chunks...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_chunks_embedding 
            ON content_chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
        
        # Create HNSW index for YouTube chunks
        print("[*] Creating HNSW index for youtube_chunks...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_youtube_chunks_embedding 
            ON youtube_chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
        
        # Create HNSW index for test data chunks
        print("[*] Creating HNSW index for test_data_chunks...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_data_chunks_embedding 
            ON test_data_chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)

        # Create HNSW index for product spec chunks
        print("[*] Creating HNSW index for product_spec_chunks...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_product_spec_chunks_embedding 
            ON product_spec_chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
        
        conn.commit()
        print("[+] Vector indexes created successfully!")
        
    except Exception as e:
        print(f"[-] Error creating vector indexes: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()
    
    return True


def main():
    """Setup complete database schema and indexes."""
    print("="*60)
    print("Database Setup")
    print("="*60)
    
    success = True
    
    # Setup all schemas
    if not setup_main_database():
        success = False
    
    if not setup_youtube_schema():
        success = False
    
    if not create_vector_indexes():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("[+] Setup completed successfully!")
        print("\nNext steps:")
        print("  1. Run: uv run jj sync products    # Sync product data")
        print("  2. Run: uv run jj sync blogs       # Sync blog content")
        print("  3. Run: uv run jj sync youtube     # Sync YouTube videos")
    else:
        print("[-] Setup completed with errors")
    print("="*60)


if __name__ == "__main__":
    main()
