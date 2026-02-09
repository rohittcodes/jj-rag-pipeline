"""
Unified ingestion script.
Handles content ingestion and embedding generation.

Usage:
    python scripts/ingest.py --all                   # Ingest everything and generate embeddings
    python scripts/ingest.py --blogs                 # Ingest blog content from raw/
    python scripts/ingest.py --youtube               # Ingest YouTube transcripts from raw/
    python scripts/ingest.py --embeddings            # Generate embeddings for all content
    python scripts/ingest.py --embeddings --blogs-only    # Generate embeddings for blogs only
    python scripts/ingest.py --embeddings --youtube-only  # Generate embeddings for YouTube only
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
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI

from src.data_pipeline.content_extractor import ContentExtractor
from src.data_pipeline.embedding_generator import EmbeddingGenerator
from src.data_pipeline.sentiment_analyzer import SentimentAnalyzer
from src.rag.product_client import ConfigDatabaseClient
from src.data_pipeline.youtube_script_parser import YouTubeScriptParser

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'josh_rag'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

BATCH_SIZE = 32


def ingest_blogs(clear_existing=False):
    """Ingest blog content from raw/blogs/ directory."""
    print("\n[*] Ingesting blog content...")
    
    raw_dir = project_root / 'raw' / 'blogs'
    if not raw_dir.exists():
        print(f"[-] Directory not found: {raw_dir}")
        return False
    
    json_files = list(raw_dir.glob('*.json'))
    if not json_files:
        print(f"[-] No JSON files found in {raw_dir}")
        return False
    
    print(f"[+] Found {len(json_files)} blog files")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    extractor = ContentExtractor()
    cursor = conn.cursor()
    extractor = ContentExtractor()
    config_client = ConfigDatabaseClient()
    
    # Initialize OpenAI client for product extraction
    api_key = os.getenv('OPENAI_API_KEY')
    openai_client = OpenAI(api_key=api_key) if api_key else None
    if not openai_client:
        print("[!] Warning: OPENAI_API_KEY not found. Falling back to regex extraction.")
    
    try:
        if clear_existing:
            print("[*] Clearing existing blog content...")
            cursor.execute("DELETE FROM content_chunks;")
            cursor.execute("DELETE FROM josh_content;")
            conn.commit()
        
        ingested = 0
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract content
            sanity_id = data.get('_id')
            title = data.get('title', '')
            raw_content = data.get('body', '')
            url = data.get('slug', {}).get('current', '')
            publish_date = data.get('publishedAt', '').split('T')[0] if data.get('publishedAt') else None
            tags = data.get('tags', [])
            # NEW: Ensure tags is a list of strings
            if tags and isinstance(tags, list):
                tags = [str(t) for t in tags]
            
            if not raw_content:
                print(f"[!] Skipping {json_file.name}: No content")
                continue
            
            # Insert content
            cursor.execute("""
                INSERT INTO josh_content (sanity_id, title, content_type, raw_content, url, publish_date, tags, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (sanity_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    raw_content = EXCLUDED.raw_content,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id;
            """, (sanity_id, title, 'blog', raw_content, url, publish_date, tags, json.dumps(data)))
            
            content_id = cursor.fetchone()[0]
            
            # Extract and chunk content
            chunks = extractor.extract_and_chunk(raw_content, title)
            
            # Initialize sentiment analyzer if enabled
            sentiment_enabled = os.getenv('SENTIMENT_ENABLED', 'true').lower() == 'true'
            sentiment_analyzer = SentimentAnalyzer() if sentiment_enabled else None
            
            # Insert chunks
            for i, chunk in enumerate(chunks):
                # Extract product mentions (Try LLM first)
                if openai_client:
                    product_mentions = extractor.extract_product_mentions_llm(chunk['text'], openai_client)
                else:
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
                
                # Add sentiment analysis if enabled and products mentioned
                if sentiment_enabled and sentiment_analyzer and config_ids:
                    # Analyze sentiment for EACH product mentioned in the chunk
                    sentiments_by_config = {}
                    
                    for config_id in list(set(config_ids)):
                        # Use ConfigDatabaseClient to get name from PROD DB
                        config_data = config_client.get_config_by_id(config_id)
                        product_name = config_data.get('product_name') if config_data else None
                        
                        if product_name:
                            sentiment = sentiment_analyzer.analyze_chunk(chunk['text'], product_name)
                            sentiments_by_config[str(config_id)] = {
                                'sentiment_score': sentiment['sentiment_score'],
                                'sentiment_label': sentiment['sentiment_label'],
                                'context_type': sentiment['context_type'],
                                'reasoning': sentiment['reasoning']
                            }
                    
                    # Store per-product sentiments
                    if sentiments_by_config:
                        chunk_metadata['sentiments'] = sentiments_by_config
                
                cursor.execute("""
                    INSERT INTO content_chunks (content_id, chunk_text, chunk_index, section_title, metadata)
                    VALUES (%s, %s, %s, %s, %s);
                """, (content_id, chunk['text'], i, chunk.get('section_title'), json.dumps(chunk_metadata)))
            
            conn.commit()
            ingested += 1
            print(f"[+] Ingested: {title} ({len(chunks)} chunks)")
        
        print(f"\n[+] Successfully ingested {ingested}/{len(json_files)} blog articles")
        return True
        
    except Exception as e:
        print(f"[-] Error ingesting blogs: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def ingest_youtube(clear_existing=False):
    """Ingest YouTube transcripts from raw/youtube/ directory."""
    print("\n[*] Ingesting YouTube content...")
    
    raw_dir = project_root / 'raw' / 'youtube'
    if not raw_dir.exists():
        print(f"[-] Directory not found: {raw_dir}")
        return False
    
    # Load metadata
    metadata_file = raw_dir / 'videos_metadata.json'
    if not metadata_file.exists():
        print(f"[-] Metadata file not found: {metadata_file}")
        # Continue without metadata if youtube-scripts are main target, or handle gracefully
        videos_metadata = {}
    else:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            videos_metadata_list = json.load(f)
        
        # Convert list to dict keyed by video_id
        videos_metadata = {v['video_id']: v for v in videos_metadata_list if 'video_id' in v}
    
    # Load transcripts
    transcript_files = [f for f in raw_dir.glob('*.json') if f.name != 'videos_metadata.json']
    if not transcript_files:
        print(f"[-] No transcript files found in {raw_dir}")
        return False
    
    print(f"[+] Found {len(transcript_files)} transcript files")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    extractor = ContentExtractor()
    config_client = ConfigDatabaseClient()
    
    # Initialize OpenAI client for product extraction
    api_key = os.getenv('OPENAI_API_KEY')
    openai_client = OpenAI(api_key=api_key) if api_key else None
    if not openai_client:
        print("[!] Warning: OPENAI_API_KEY not found. Falling back to regex extraction.")
    
    try:
        if clear_existing:
            print("[*] Clearing existing YouTube content...")
            cursor.execute("DELETE FROM youtube_chunks;")
            cursor.execute("DELETE FROM youtube_content;")
            conn.commit()
        
        ingested = 0
        for transcript_file in transcript_files:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            
            video_id = transcript.get('video_id')
            if not video_id:
                print(f"[!] Skipping {transcript_file.name}: No video_id")
                continue
            
            # Get metadata
            metadata = videos_metadata.get(video_id, {})
            full_text = transcript.get('full_text', '')
            
            if not full_text:
                print(f"[!] Skipping {video_id}: No transcript text")
                continue
            
            # Parse publish date
            publish_date = None
            if metadata.get('published_at'):
                try:
                    publish_date = datetime.fromisoformat(metadata['published_at'].replace('Z', '+00:00')).date()
                except:
                    pass
            
            # Insert content
            cursor.execute("""
                INSERT INTO youtube_content
                (video_id, title, url, description, publish_date,
                 thumbnail_url, transcript_type, language, full_transcript)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (video_id) DO NOTHING
                RETURNING id;
            """, (
                video_id,
                metadata.get('title', f'Video {video_id}'),
                f"https://www.youtube.com/watch?v={video_id}",
                metadata.get('description', ''),
                publish_date,
                metadata.get('thumbnail', ''),
                transcript.get('transcript_type', 'auto-generated'),
                transcript.get('language', 'English'),
                full_text
            ))
            
            result = cursor.fetchone()
            if not result:
                print(f"[!] Skipping {video_id}: Already exists")
                continue
            
            content_id = result[0]
            
            # Chunk the transcript
            chunks = extractor.extract_and_chunk(full_text, metadata.get('title', ''))
            
            # Insert chunks
            for i, chunk in enumerate(chunks):
                chunk_text = chunk['text']
                
                # Extract product mentions (Try LLM first)
                if openai_client:
                    product_mentions = extractor.extract_product_mentions_llm(chunk_text, openai_client)
                else:
                    product_mentions = extractor.extract_product_mentions(chunk_text)
                config_ids = []
                
                for mention in product_mentions:
                    configs = config_client.search_configs_by_name(mention)
                    if configs:
                        config_ids.extend([c['config_id'] for c in configs[:3]])
                
                chunk_metadata = {
                    'config_ids': list(set(config_ids)),
                    'product_mentions': product_mentions,
                    'video_id': video_id
                }
                
                cursor.execute("""
                    INSERT INTO youtube_chunks
                    (youtube_content_id, chunk_text, chunk_index, metadata)
                    VALUES (%s, %s, %s, %s);
                """, (content_id, chunk_text, i, json.dumps(chunk_metadata)))
            
            conn.commit()
            ingested += 1
            print(f"[+] Ingested: {metadata.get('title', video_id)} ({len(chunks)} chunks)")
        
        print(f"\n[+] Successfully ingested {ingested}/{len(transcript_files)} YouTube videos")
        return True
        
    except Exception as e:
        print(f"[-] Error ingesting YouTube: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def generate_embeddings(blogs_only=False, youtube_only=False, regenerate=False):
    """Generate embeddings for content chunks."""
    if blogs_only and youtube_only:
        print("[-] Cannot specify both --blogs-only and --youtube-only")
        return False
    
    source = "YouTube" if youtube_only else "blog" if blogs_only else "all"
    print(f"\n[*] Generating embeddings for {source} content...")
    
    try:
        generator = EmbeddingGenerator()
    except Exception as e:
        print(f"[-] Failed to initialize embedding generator: {e}")
        return False
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        # Determine which tables to process
        tables_to_process = []
        
        if youtube_only:
            tables_to_process.append(('youtube_chunks', 'youtube_content_id'))
        elif blogs_only:
            tables_to_process.append(('content_chunks', 'content_id'))
        else:
            tables_to_process.append(('content_chunks', 'content_id'))
            tables_to_process.append(('youtube_chunks', 'youtube_content_id'))
        
        # Fetch chunks
        all_chunks = []
        for table_name, _ in tables_to_process:
            if regenerate:
                cursor.execute(f"SELECT id, chunk_text, '{table_name}' as source_table FROM {table_name} ORDER BY id;")
            else:
                cursor.execute(f"SELECT id, chunk_text, '{table_name}' as source_table FROM {table_name} WHERE embedding IS NULL ORDER BY id;")
            
            chunks = cursor.fetchall()
            all_chunks.extend(chunks)
        
        if not all_chunks:
            print("[*] No chunks need embedding generation")
            return True
        
        total_chunks = len(all_chunks)
        print(f"[+] Found {total_chunks} chunks to process")
        
        processed = 0
        
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            batch_ids = [chunk['id'] for chunk in batch]
            batch_texts = [chunk['chunk_text'] for chunk in batch]
            batch_tables = [chunk['source_table'] for chunk in batch]
            
            print(f"[*] Processing batch {i//BATCH_SIZE + 1}/{(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE}...")
            
            try:
                embeddings = generator.generate_embeddings(
                    batch_texts,
                    batch_size=BATCH_SIZE,
                    show_progress=False
                )
                
                for chunk_id, embedding, table_name in zip(batch_ids, embeddings, batch_tables):
                    cursor.execute(
                        f"UPDATE {table_name} SET embedding = %s WHERE id = %s;",
                        (embedding, chunk_id)
                    )
                
                conn.commit()
                processed += len(batch)
                
            except Exception as e:
                print(f"[-] Error processing batch: {e}")
                conn.rollback()
                continue
        
        print(f"[+] Processed {processed}/{total_chunks} chunks")
        
        # Show stats
        if not blogs_only:
            cursor.execute("SELECT COUNT(*) as total, COUNT(embedding) as with_embeddings FROM youtube_chunks;")
            stats = cursor.fetchone()
            if stats and stats['total'] > 0:
                print(f"[+] YouTube - Total: {stats['total']}, With embeddings: {stats['with_embeddings']}")
        
        if not youtube_only:
            cursor.execute("SELECT COUNT(*) as total, COUNT(embedding) as with_embeddings FROM content_chunks;")
            stats = cursor.fetchone()
            if stats and stats['total'] > 0:
                print(f"[+] Blogs - Total: {stats['total']}, With embeddings: {stats['with_embeddings']}")
        
        return True
        
    except Exception as e:
        print(f"[-] Error generating embeddings: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()



def scan_video_folders(youtube_dir: Path, verbose: bool = False):
    """
    Scan YouTube directory for video folders.
    
    Returns list of folders with pattern T followed by digits (e.g., T223, T207).
    """
    video_folders = []
    
    # Scan year directories (2025, 2026, etc.)
    for year_dir in youtube_dir.iterdir():
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        
        # Scan video folders within year
        for folder in year_dir.iterdir():
            if folder.is_dir() and folder.name.startswith('T'):
                video_folders.append(folder)
    
    # Also check "Completed Videos" directory
    completed_dir = youtube_dir / 'Completed Videos' / 'Tech'
    if completed_dir.exists():
        for year_dir in completed_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for folder in year_dir.iterdir():
                if folder.is_dir() and folder.name.startswith('T'):
                    video_folders.append(folder)
    
    if verbose:
        print(f"[+] Found {len(video_folders)} video folders")
    
    return video_folders


def ingest_youtube_scripts(clear_existing=False, sentiment_enabled=True, verbose=True):
    """Ingest YouTube video scripts from .docx files."""
    print("\\n[*] Ingesting YouTube scripts (.docx)...")
    
    # Initialize components
    youtube_dir = project_root / 'raw' / 'youtube'
    
    if not youtube_dir.exists():
        print(f"[-] YouTube directory not found: {youtube_dir}")
        return False
    
    parser = YouTubeScriptParser(verbose=verbose)
    extractor = ContentExtractor()
    config_client = ConfigDatabaseClient()
    
    # Check for sentiment analysis override
    if sentiment_enabled:
        sentiment_enabled_env = os.getenv('SENTIMENT_ENABLED', 'true').lower() == 'true'
        if not sentiment_enabled_env:
            print("[*] Sentiment analysis disabled by environment variable")
            sentiment_enabled = False
    
    sentiment_analyzer = None
    if sentiment_enabled:
        print("[*] Sentiment analysis enabled")
        sentiment_analyzer = SentimentAnalyzer(verbose=verbose, rate_limit_delay=0.3)
            
    # Initialize OpenAI client for product extraction
    api_key = os.getenv('OPENAI_API_KEY')
    openai_client = OpenAI(api_key=api_key) if api_key else None
    if not openai_client:
        print("[!] Warning: OPENAI_API_KEY not found. Falling back to regex extraction.")
    
    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    try:
        # Clear existing data if requested
        if clear_existing:
            print("[*] Clearing existing YouTube content...")
            cursor.execute("DELETE FROM youtube_chunks;")
            cursor.execute("DELETE FROM youtube_content;")
            conn.commit()
            print("[+] Cleared existing YouTube data")
        
        # Scan for video folders
        print("[*] Scanning for video folders...")
        video_folders = scan_video_folders(youtube_dir, verbose=verbose)
        
        if not video_folders:
            print("[-] No video folders found")
            return False
        
        # Process each video folder
        ingested_count = 0
        skipped_count = 0
        error_count = 0
        
        for i, folder in enumerate(video_folders, 1):
            if verbose:
                print(f"\\n[{i}/{len(video_folders)}] Processing: {folder.name}")
            
            try:
                # Select main script file
                main_script = parser.select_main_script(folder)
                if not main_script:
                    if verbose: print(f"    [!] No main script found, skipping")
                    skipped_count += 1
                    continue
                
                if verbose: print(f"    [+] Selected: {main_script.name}")
                
                # Parse script
                script_data = parser.parse_script(main_script)
                if not script_data:
                    if verbose: print(f"    [!] Failed to parse script, skipping")
                    error_count += 1
                    continue
                
                video_id = script_data['video_id']
                title = script_data['title']
                full_text = script_data['text']
                
                if verbose:
                    print(f"    [+] Video ID: {video_id}, Title: {title}")
                    print(f"    [+] Text length: {len(full_text)} chars")
                
                # Get file modification date as publish date
                publish_date = datetime.fromtimestamp(main_script.stat().st_mtime).date()
                
                # Insert into youtube_content
                cursor.execute("""
                    INSERT INTO youtube_content
                    (video_id, title, url, description, publish_date,
                     thumbnail_url, transcript_type, language, full_transcript)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (video_id) DO UPDATE
                    SET title = EXCLUDED.title,
                        full_transcript = EXCLUDED.full_transcript,
                        publish_date = EXCLUDED.publish_date
                    RETURNING id;
                """, (
                    video_id,
                    title,
                    f"https://www.youtube.com/watch?v={video_id}",  # Placeholder URL
                    '',  # No description
                    publish_date,
                    '',  # No thumbnail
                    'script',  # Type: script (not transcript)
                    'English',
                    full_text
                ))
                
                result = cursor.fetchone()
                content_id = result[0]
                
                # Delete existing chunks for this content (if updating)
                cursor.execute("DELETE FROM youtube_chunks WHERE youtube_content_id = %s", (content_id,))
                
                # Chunk the script
                if verbose: print(f"    [*] Chunking content...")
                chunks = extractor.extract_and_chunk(full_text, title)
                if verbose: print(f"    [+] Created {len(chunks)} chunks")
                
                # Process chunks
                chunk_data = []
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_text = chunk['text']
                    
                    # Extract product mentions (Try LLM first)
                    if openai_client:
                        product_mentions = extractor.extract_product_mentions_llm(chunk_text, openai_client)
                    else:
                        product_mentions = extractor.extract_product_mentions(chunk_text)
                    
                    # Convert product mentions to config_ids using Prod DB
                    config_ids = []
                    for mention in product_mentions:
                        configs = config_client.search_configs_by_name(mention)
                        if configs:
                            config_ids.extend([c['config_id'] for c in configs[:3]])
                    
                    # Build metadata
                    chunk_metadata = {
                        'char_count': len(chunk_text),
                        'config_ids': list(set(config_ids)),
                        'product_mentions': product_mentions
                    }
                    
                    # Add sentiment analysis if enabled and products mentioned
                    if sentiment_enabled and sentiment_analyzer and config_ids:
                        sentiments_by_config = {}
                        
                        for config_id in list(set(config_ids)):
                            # Use ConfigDatabaseClient to get name from PROD DB
                            config_data = config_client.get_config_by_id(config_id)
                            product_name = config_data.get('product_name') if config_data else f"Config {config_id}"
                            
                            try:
                                sentiment = sentiment_analyzer.analyze_chunk(chunk_text, product_name)
                                sentiments_by_config[str(config_id)] = {
                                    'sentiment_score': sentiment['sentiment_score'],
                                    'sentiment_label': sentiment['sentiment_label'],
                                    'context_type': sentiment['context_type'],
                                    'reasoning': sentiment['reasoning']
                                }
                            except Exception as e:
                                if verbose: print(f"    [!] Sentiment analysis error for {product_name}: {e}")
                        
                        if sentiments_by_config:
                            chunk_metadata['sentiments'] = sentiments_by_config
                    
                    # Note: Embedding generation is separate step now to be consistent with other ingestion
                    chunk_data.append((
                        content_id,
                        chunk_text,
                        chunk_idx,
                        json.dumps(chunk_metadata)
                    ))
                
                # Batch insert chunks (without embeddings for now)
                execute_batch(cursor, """
                    INSERT INTO youtube_chunks
                    (youtube_content_id, chunk_text, chunk_index, metadata)
                    VALUES (%s, %s, %s, %s)
                """, chunk_data, page_size=100)
                
                conn.commit()
                ingested_count += 1
                if verbose: print(f"    [+] Successfully ingested {len(chunks)} chunks")
                
            except Exception as e:
                print(f"    [!] Error processing {folder.name}: {e}")
                error_count += 1
                conn.rollback()
                continue
        
        # Summary
        print("\\n" + "="*60)
        print(f"[+] Successfully ingested: {ingested_count} videos")
        print(f"[!] Skipped (no script): {skipped_count} videos")
        print(f"[!] Errors: {error_count} videos")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\\n[!] Fatal error: {e}")
        conn.rollback()
        return False
        
    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Ingest content and generate embeddings')
    parser.add_argument('--all', action='store_true', help='Ingest all content and generate embeddings')
    parser.add_argument('--blogs', action='store_true', help='Ingest blog content')
    parser.add_argument('--youtube', action='store_true', help='Ingest YouTube transcripts (from JSON)')
    parser.add_argument('--youtube-scripts', action='store_true', help='Ingest YouTube scripts (from DOCX)')
    parser.add_argument('--embeddings', action='store_true', help='Generate embeddings')
    parser.add_argument('--blogs-only', action='store_true', help='Only process blog content (with --embeddings)')
    parser.add_argument('--youtube-only', action='store_true', help='Only process YouTube content (with --embeddings)')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate all embeddings (with --embeddings)')
    parser.add_argument('--clear', action='store_true', help='Clear existing data before ingesting')
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("="*60)
    print("Content Ingestion")
    print("="*60)
    
    success = True
    
    if args.all or args.blogs:
        if not ingest_blogs(clear_existing=args.clear):
            success = False
    
    if args.all or args.youtube:
        if not ingest_youtube(clear_existing=args.clear):
            success = False
            
    if args.all or args.youtube_scripts:
        if not ingest_youtube_scripts(clear_existing=args.clear, sentiment_enabled=True):
            success = False
    
    if args.all or args.embeddings:
        if not generate_embeddings(
            blogs_only=args.blogs_only,
            youtube_only=args.youtube_only,
            regenerate=args.regenerate
        ):
            success = False
    
    print("\n" + "="*60)
    if success:
        print("[+] Ingestion completed successfully!")
        if args.all or args.embeddings:
            print("\n[*] Next step: Create vector indexes")
            print("    python scripts/setup.py --indexes")
    else:
        print("[-] Ingestion completed with errors")
    print("="*60)


if __name__ == "__main__":
    main()
