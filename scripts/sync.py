"""
Unified sync script.
Handles syncing from external sources (Sanity CMS, Production DB).

Usage:
    python scripts/sync.py --sanity --all           # Sync all articles from Sanity
    python scripts/sync.py --sanity --id=<ID>       # Sync specific article
    python scripts/sync.py --products               # Sync product configs from production
    python scripts/sync.py --all                    # Sync everything
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from datetime import datetime

from src.data_pipeline.sanity_client import SanityClient
from src.data_pipeline.content_extractor import ContentExtractor
from src.data_pipeline.embedding_generator import EmbeddingGenerator
from src.rag.product_client import ConfigDatabaseClient
from src.data_pipeline.s3_client import S3Client
from src.data_pipeline.test_data_parser import TestDataParser

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'josh_rag'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

PROD_DB_CONFIG = {
    'host': os.getenv('DATABASE_HOST'),
    'port': os.getenv('DATABASE_PORT', '5432'),
    'database': os.getenv('DATABASE_NAME'),
    'user': os.getenv('DATABASE_USER'),
    'password': os.getenv('DATABASE_PASSWORD')
}


def sync_from_sanity(article_id=None, sync_all=False):
    """Sync content from Sanity CMS."""
    if not sync_all and not article_id:
        print("[-] Must specify either --all or --id=<article_id>")
        return False
    
    print("\n[*] Syncing from Sanity CMS...")
    
    try:
        sanity = SanityClient()
        extractor = ContentExtractor()
        generator = EmbeddingGenerator()
        config_client = ConfigDatabaseClient()
    except Exception as e:
        print(f"[-] Failed to initialize clients: {e}")
        return False
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Fetch articles
        if sync_all:
            print("[*] Fetching all articles from Sanity...")
            articles = sanity.fetch_all_articles()
        else:
            print(f"[*] Fetching article {article_id} from Sanity...")
            article = sanity.fetch_article(article_id)
            articles = [article] if article else []
        
        if not articles:
            print("[-] No articles found")
            return False
        
        print(f"[+] Found {len(articles)} articles")
        
        synced = 0
        for article in articles:
            sanity_id = article.get('_id')
            title = article.get('title', '')
            raw_content = article.get('body', '')
            url = article.get('slug', {}).get('current', '')
            publish_date = article.get('publishedAt', '').split('T')[0] if article.get('publishedAt') else None
            tags = article.get('tags', [])
            
            if not raw_content:
                print(f"[!] Skipping {title}: No content")
                continue
            
            # Insert or update content
            cursor.execute("""
                INSERT INTO josh_content (sanity_id, title, content_type, raw_content, url, publish_date, tags, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (sanity_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    raw_content = EXCLUDED.raw_content,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id;
            """, (sanity_id, title, 'blog', raw_content, url, publish_date, tags, json.dumps(article)))
            
            content_id = cursor.fetchone()[0]
            
            # Delete old chunks
            cursor.execute("DELETE FROM content_chunks WHERE content_id = %s;", (content_id,))
            
            # Extract and chunk content
            chunks = extractor.extract_and_chunk(raw_content, title)
            
            # Insert chunks with embeddings
            for i, chunk in enumerate(chunks):
                # Extract product mentions
                product_mentions = extractor.extract_product_mentions(chunk['text'])
                config_ids = []
                
                for mention in product_mentions:
                    configs = config_client.search_configs_by_name(mention)
                    if configs:
                        config_ids.extend([c['config_id'] for c in configs[:3]])
                
                chunk_metadata = {
                    'config_ids': list(set(config_ids)),
                    'product_mentions': product_mentions
                }
                
                # Generate embedding
                embedding = generator.generate_embeddings([chunk['text']], show_progress=False)[0]
                
                cursor.execute("""
                    INSERT INTO content_chunks (content_id, chunk_text, chunk_index, section_title, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (content_id, chunk['text'], i, chunk.get('section_title'), json.dumps(chunk_metadata), embedding))
            
            conn.commit()
            synced += 1
            print(f"[+] Synced: {title} ({len(chunks)} chunks)")
        
        print(f"\n[+] Successfully synced {synced}/{len(articles)} articles")
        return True
        
    except Exception as e:
        print(f"[-] Error syncing from Sanity: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def sync_products():
    """Sync product configs from production database."""
    print("\n[*] Syncing product configs from production...")
    
    # Check production DB config
    if not all([PROD_DB_CONFIG['host'], PROD_DB_CONFIG['database'], PROD_DB_CONFIG['user']]):
        print("[-] Production database credentials not configured")
        print("    Set DATABASE_HOST, DATABASE_NAME, DATABASE_USER in .env")
        return False
    
    try:
        # Connect to production DB (read-only)
        prod_conn = psycopg2.connect(**PROD_DB_CONFIG)
        prod_cursor = prod_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Connect to local DB
        local_conn = psycopg2.connect(**DB_CONFIG)
        local_cursor = local_conn.cursor()
        
        # Fetch configs from production
        print("[*] Fetching configs from production...")
        prod_cursor.execute("""
            SELECT 
                c.id as config_id,
                c.product_id,
                p.name as product_name,
                p.brand,
                p.model,
                c.price,
                jsonb_object_agg(
                    COALESCE(cp.name, ''),
                    COALESCE(cp.value, '')
                ) FILTER (WHERE cp.name IS NOT NULL) as specs,
                c.rating,
                c.josh_context
            FROM configs c
            LEFT JOIN products p ON c.product_id = p.id
            LEFT JOIN config_properties cp ON c.id = cp.config_id
            GROUP BY c.id, c.product_id, p.name, p.brand, p.model, c.price, c.rating, c.josh_context;
        """)
        
        configs = prod_cursor.fetchall()
        print(f"[+] Found {len(configs)} configs in production")
        
        # Fetch test data
        print("[*] Fetching test data...")
        prod_cursor.execute("""
            SELECT 
                ctd.config_id,
                jsonb_object_agg(
                    ctdc.name,
                    jsonb_build_object(
                        'value', ctd.value,
                        'metric', ctdm.name
                    )
                ) as test_data
            FROM config_test_data ctd
            LEFT JOIN config_test_data_categories ctdc ON ctd.category_id = ctdc.id
            LEFT JOIN config_test_data_metrics ctdm ON ctd.metric_id = ctdm.id
            GROUP BY ctd.config_id;
        """)
        
        test_data_map = {row['config_id']: row['test_data'] for row in prod_cursor.fetchall()}
        
        # Sync to local DB
        print("[*] Syncing to local database...")
        synced = 0
        
        for config in configs:
            config_id = config['config_id']
            test_data = test_data_map.get(config_id, {})
            
            local_cursor.execute("""
                INSERT INTO configs (
                    config_id, product_id, product_name, brand, model,
                    price, specs, test_data, rating, josh_context, last_synced
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (config_id) DO UPDATE SET
                    product_id = EXCLUDED.product_id,
                    product_name = EXCLUDED.product_name,
                    brand = EXCLUDED.brand,
                    model = EXCLUDED.model,
                    price = EXCLUDED.price,
                    specs = EXCLUDED.specs,
                    test_data = EXCLUDED.test_data,
                    rating = EXCLUDED.rating,
                    josh_context = EXCLUDED.josh_context,
                    last_synced = CURRENT_TIMESTAMP;
            """, (
                config_id,
                config['product_id'],
                config['product_name'],
                config['brand'],
                config['model'],
                config['price'],
                json.dumps(config['specs']) if config['specs'] else None,
                json.dumps(test_data) if test_data else None,
                config['rating'],
                config['josh_context']
            ))
            
            synced += 1
            if synced % 100 == 0:
                print(f"[*] Synced {synced}/{len(configs)} configs...")
        
        local_conn.commit()
        print(f"\n[+] Successfully synced {synced} product configs")
        
        prod_cursor.close()
        prod_conn.close()
        local_cursor.close()
        local_conn.close()
        
        return True
        
    except Exception as e:
        print(f"[-] Error syncing products: {e}")
        import traceback
        traceback.print_exc()
        return False


def sync_test_data(config_id=None, limit=None):
    """
    Sync test data PDFs from S3, parse them, and create searchable chunks.
    
    Args:
        config_id: Optional specific config ID to sync (otherwise sync all)
        limit: Optional limit on number of configs to process
    
    Returns:
        True if successful, False otherwise
    """
    print("\n[*] Syncing test data from S3...")
    
    try:
        # Initialize clients
        s3_client = S3Client()
        parser = TestDataParser()
        embedding_gen = EmbeddingGenerator()
        
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get configs that have test data PDFs
        if config_id:
            print(f"[*] Syncing test data for config ID: {config_id}")
            cursor.execute("""
                SELECT config_id, product_name, test_data_pdf_url, test_data_pdf_key
                FROM configs
                WHERE config_id = %s AND test_data_pdf_key IS NOT NULL
            """, (config_id,))
        else:
            print(f"[*] Syncing test data for all configs with PDFs...")
            query = """
                SELECT config_id, product_name, test_data_pdf_url, test_data_pdf_key
                FROM configs
                WHERE test_data_pdf_key IS NOT NULL
                ORDER BY config_id
            """
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
        
        configs = cursor.fetchall()
        print(f"[+] Found {len(configs)} configs with test data PDFs")
        
        if not configs:
            print("[!] No configs with test data PDFs found")
            return True
        
        # Create tmp directory for downloads
        tmp_dir = Path('tmp/test_data')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        synced = 0
        skipped = 0
        errors = 0
        
        for config in configs:
            config_id = config['config_id']
            product_name = config['product_name']
            s3_key = config['test_data_pdf_key']
            
            print(f"\n[*] Processing config {config_id}: {product_name}")
            
            try:
                # Check if already synced
                cursor.execute("""
                    SELECT COUNT(*) as count FROM test_data_chunks
                    WHERE config_id = %s
                """, (config_id,))
                existing_count = cursor.fetchone()['count']
                
                if existing_count > 0:
                    print(f"[!] Skipping (already has {existing_count} chunks)")
                    skipped += 1
                    continue
                
                # Download PDF
                local_path = tmp_dir / f"{config_id}.pdf"
                print(f"    Downloading from S3: {s3_key}")
                
                if not s3_client.download_file(s3_key, local_path):
                    print(f"[-] Failed to download PDF")
                    errors += 1
                    continue
                
                # Parse PDF
                print(f"    Parsing PDF...")
                test_data = parser.parse_pdf(local_path, product_name)
                
                # Create chunks
                print(f"    Creating chunks...")
                chunks = parser.create_chunks(test_data, config_id)
                print(f"    Created {len(chunks)} chunks")
                
                # Generate embeddings for all chunks at once
                chunk_texts = [chunk['chunk_text'] for chunk in chunks]
                embeddings = embedding_gen.generate_embeddings(chunk_texts, show_progress=False)
                
                # Insert chunks with embeddings
                for chunk, embedding in zip(chunks, embeddings):
                    cursor.execute("""
                        INSERT INTO test_data_chunks
                        (config_id, test_type, test_description, chunk_text, 
                         benchmark_results, embedding, source_file)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        chunk['config_id'],
                        chunk['test_type'],
                        chunk['test_description'],
                        chunk['chunk_text'],
                        json.dumps(chunk['benchmark_results']),
                        embedding,
                        s3_key
                    ))
                
                conn.commit()
                
                # Clean up downloaded file
                local_path.unlink()
                
                synced += 1
                print(f"[+] Successfully synced test data for config {config_id}")
                
            except Exception as e:
                print(f"[-] Error processing config {config_id}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
                conn.rollback()
                continue
        
        print(f"\n[*] Test data sync summary:")
        print(f"    Synced: {synced}")
        print(f"    Skipped: {skipped}")
        print(f"    Errors: {errors}")
        
        cursor.close()
        conn.close()
        
        return errors == 0
        
    except Exception as e:
        print(f"[-] Error syncing test data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Sync data from external sources')
    parser.add_argument('--all', action='store_true', help='Sync everything')
    parser.add_argument('--sanity', action='store_true', help='Sync from Sanity CMS')
    parser.add_argument('--id', type=str, help='Specific article ID to sync (with --sanity)')
    parser.add_argument('--products', action='store_true', help='Sync product configs from production')
    parser.add_argument('--test-data', action='store_true', help='Sync test data PDFs from S3')
    parser.add_argument('--config-id', type=int, help='Specific config ID for test data sync')
    parser.add_argument('--limit', type=int, help='Limit number of test data files to process')
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("="*60)
    print("Data Sync")
    print("="*60)
    
    success = True
    
    if args.all or args.sanity:
        sync_all = args.all or (args.sanity and not args.id)
        if not sync_from_sanity(article_id=args.id, sync_all=sync_all):
            success = False
    
    if args.all or args.products:
        if not sync_products():
            success = False
    
    if args.all or args.test_data:
        if not sync_test_data(config_id=args.config_id, limit=args.limit):
            success = False
    
    print("\n" + "="*60)
    if success:
        print("[+] Sync completed successfully!")
    else:
        print("[-] Sync completed with errors")
    print("="*60)


if __name__ == "__main__":
    main()
