"""
Content ingestion service for processing Sanity CMS articles.
Handles fetching, chunking, and embedding of content.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from datetime import datetime
from typing import Optional, Dict, List
from dotenv import load_dotenv

from src.data_pipeline.sanity_client import SanityClient
from src.data_pipeline.content_extractor import ContentExtractor
from src.data_pipeline.embedding_generator import EmbeddingGenerator

load_dotenv()


class ContentIngestionService:
    """Service for ingesting content from Sanity CMS into the RAG system."""
    
    def __init__(self):
        """Initialize the ingestion service."""
        self.sanity_client = SanityClient()
        self.content_extractor = ContentExtractor()
        self.embedding_generator = EmbeddingGenerator()
        
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'josh_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '512'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '50'))
    
    def ingest_article_by_id(self, article_id: str) -> bool:
        """
        Fetch and ingest a single article by ID.
        
        Args:
            article_id: Sanity document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n[*] Ingesting article: {article_id}")
            
            # Fetch article from Sanity
            article_data = self.sanity_client.fetch_article_by_id(article_id)
            
            if not article_data:
                print(f"[-] Article {article_id} not found in Sanity")
                return False
            
            # Extract content
            extracted = self.content_extractor.extract_from_json(
                article_data,
                filename=article_id
            )
            
            if not extracted or not extracted.get('raw_content'):
                print(f"[-] Failed to extract content from article {article_id}")
                return False
            
            print(f"[+] Extracted: {extracted['title']}")
            
            # Connect to database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if article already exists
            cursor.execute("""
                SELECT id FROM josh_content WHERE url = %s
            """, (extracted['url'],))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing article
                content_id = existing[0]
                print(f"[*] Updating existing article (ID: {content_id})")
                
                cursor.execute("""
                    UPDATE josh_content
                    SET title = %s,
                        raw_content = %s,
                        structured_data = %s,
                        tags = %s,
                        publish_date = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (
                    extracted['title'],
                    extracted['raw_content'],
                    json.dumps(extracted['structured_data']),
                    extracted['tags'],
                    extracted['publish_date'],
                    content_id
                ))
                
                # Delete old chunks
                cursor.execute("""
                    DELETE FROM content_chunks WHERE content_id = %s
                """, (content_id,))
                
            else:
                # Insert new article
                print(f"[*] Inserting new article")
                
                cursor.execute("""
                    INSERT INTO josh_content 
                    (content_type, title, url, publish_date, raw_content, structured_data, tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    extracted['content_type'],
                    extracted['title'],
                    extracted['url'],
                    extracted['publish_date'],
                    extracted['raw_content'],
                    json.dumps(extracted['structured_data']),
                    extracted['tags']
                ))
                
                content_id = cursor.fetchone()[0]
            
            # Chunk the content
            chunks = self.content_extractor.chunk_content(
                extracted['raw_content'],
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            print(f"[*] Created {len(chunks)} chunks")
            
            # Map chunks to sections for config_ids
            sections = extracted['structured_data'].get('sections', [])
            section_map = {}
            for section in sections:
                heading = section.get('heading', '')
                config_ids = section.get('config_ids', [])
                if heading and config_ids:
                    section_map[heading] = config_ids
            
            # Insert chunks
            chunk_count = 0
            for chunk in chunks:
                section_title = None
                config_ids = []
                chunk_text = chunk['text']
                
                # Extract section title from chunk
                if chunk_text.startswith('## '):
                    first_line = chunk_text.split('\n')[0]
                    section_title = first_line.replace('## ', '').strip()
                    config_ids = section_map.get(section_title, [])
                
                cursor.execute("""
                    INSERT INTO content_chunks
                    (content_id, chunk_text, chunk_index, section_title, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    content_id,
                    chunk_text,
                    chunk['index'],
                    section_title,
                    json.dumps({
                        'char_count': chunk['char_count'],
                        'config_ids': config_ids
                    })
                ))
                
                chunk_id = cursor.fetchone()[0]
                chunk_count += 1
            
            conn.commit()
            print(f"[+] Inserted {chunk_count} chunks")
            
            # Generate embeddings
            print(f"[*] Generating embeddings...")
            cursor.execute("""
                SELECT id, chunk_text 
                FROM content_chunks 
                WHERE content_id = %s AND embedding IS NULL
            """, (content_id,))
            
            chunks_to_embed = cursor.fetchall()
            
            if chunks_to_embed:
                for chunk_id, chunk_text in chunks_to_embed:
                    embedding = self.embedding_generator.generate_embedding(chunk_text)
                    
                    cursor.execute("""
                        UPDATE content_chunks
                        SET embedding = %s
                        WHERE id = %s
                    """, (embedding, chunk_id))
                
                conn.commit()
                print(f"[+] Generated {len(chunks_to_embed)} embeddings")
            
            cursor.close()
            conn.close()
            
            print(f"[+] Successfully ingested article: {extracted['title']}")
            return True
            
        except Exception as e:
            print(f"[-] Error ingesting article {article_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def sync_all_articles(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Sync all articles from Sanity.
        
        Args:
            limit: Optional limit on number of articles
            
        Returns:
            Dictionary with sync statistics
        """
        try:
            print(f"\n[*] Starting full sync from Sanity (limit: {limit or 'all'})")
            
            # Fetch articles
            articles = self.sanity_client.fetch_articles(limit=limit)
            
            if not articles:
                print("[-] No articles fetched from Sanity")
                return {"success": 0, "failed": 0, "total": 0}
            
            print(f"[+] Fetched {len(articles)} articles from Sanity")
            
            success_count = 0
            failed_count = 0
            
            for article in articles:
                article_id = article.get('_id')
                title = article.get('title', 'Untitled')
                
                print(f"\n[*] Processing: {title}")
                
                # Extract and ingest
                extracted = self.content_extractor.extract_from_json(article, filename=article_id)
                
                if not extracted:
                    print(f"[-] Failed to extract: {title}")
                    failed_count += 1
                    continue
                
                # Use the same ingestion logic
                if self._ingest_extracted_content(extracted):
                    success_count += 1
                else:
                    failed_count += 1
            
            print(f"\n[+] Sync complete: {success_count} success, {failed_count} failed")
            
            return {
                "success": success_count,
                "failed": failed_count,
                "total": len(articles)
            }
            
        except Exception as e:
            print(f"[-] Error during full sync: {e}")
            import traceback
            traceback.print_exc()
            return {"success": 0, "failed": 0, "total": 0, "error": str(e)}
    
    def _ingest_extracted_content(self, extracted: Dict) -> bool:
        """Helper method to ingest already extracted content."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute("SELECT id FROM josh_content WHERE url = %s", (extracted['url'],))
            existing = cursor.fetchone()
            
            if existing:
                content_id = existing[0]
                cursor.execute("""
                    UPDATE josh_content
                    SET title = %s, raw_content = %s, structured_data = %s,
                        tags = %s, publish_date = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (
                    extracted['title'], extracted['raw_content'],
                    json.dumps(extracted['structured_data']),
                    extracted['tags'], extracted['publish_date'], content_id
                ))
                cursor.execute("DELETE FROM content_chunks WHERE content_id = %s", (content_id,))
            else:
                cursor.execute("""
                    INSERT INTO josh_content 
                    (content_type, title, url, publish_date, raw_content, structured_data, tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """, (
                    extracted['content_type'], extracted['title'], extracted['url'],
                    extracted['publish_date'], extracted['raw_content'],
                    json.dumps(extracted['structured_data']), extracted['tags']
                ))
                content_id = cursor.fetchone()[0]
            
            # Chunk and insert
            chunks = self.content_extractor.chunk_content(
                extracted['raw_content'],
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            sections = extracted['structured_data'].get('sections', [])
            section_map = {s.get('heading', ''): s.get('config_ids', []) for s in sections if s.get('heading')}
            
            for chunk in chunks:
                section_title = None
                config_ids = []
                if chunk['text'].startswith('## '):
                    section_title = chunk['text'].split('\n')[0].replace('## ', '').strip()
                    config_ids = section_map.get(section_title, [])
                
                cursor.execute("""
                    INSERT INTO content_chunks
                    (content_id, chunk_text, chunk_index, section_title, metadata)
                    VALUES (%s, %s, %s, %s, %s) RETURNING id;
                """, (
                    content_id, chunk['text'], chunk['index'], section_title,
                    json.dumps({'char_count': chunk['char_count'], 'config_ids': config_ids})
                ))
                
                chunk_id = cursor.fetchone()[0]
                
                # Generate embedding
                embedding = self.embedding_generator.generate_embedding(chunk['text'])
                cursor.execute("UPDATE content_chunks SET embedding = %s WHERE id = %s", (embedding, chunk_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"[+] Ingested: {extracted['title']} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            print(f"[-] Error ingesting {extracted.get('title')}: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_articles,
                    COUNT(*) FILTER (WHERE content_type = 'blog') as blog_count,
                    MAX(updated_at) as last_updated
                FROM josh_content
            """)
            content_stats = cursor.fetchone()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as embedded_chunks
                FROM content_chunks
            """)
            chunk_stats = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                "total_articles": content_stats['total_articles'],
                "blog_articles": content_stats['blog_count'],
                "last_updated": str(content_stats['last_updated']) if content_stats['last_updated'] else None,
                "total_chunks": chunk_stats['total_chunks'],
                "embedded_chunks": chunk_stats['embedded_chunks']
            }
            
        except Exception as e:
            print(f"[-] Error getting stats: {e}")
            return {}
