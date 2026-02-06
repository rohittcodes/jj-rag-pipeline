"""
YouTube operations script.
Handles fetching transcripts and testing.

Usage:
    python scripts/youtube.py --fetch --count=20    # Fetch 20 videos
    python scripts/youtube.py --test                # Test fetch with 1 video
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import time
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
CHANNEL_ID = 'UCpSLcDFOXFAam4WkT3aF_Jw'  # @JustJoshTech
OUTPUT_DIR = project_root / 'raw' / 'youtube'


class YouTubeTranscriptFetcher:
    def __init__(self, api_key, verbose=True):
        self.api_key = api_key
        self.verbose = verbose
        self.quota_used = 0
        
        if not api_key:
            raise ValueError("YouTube API key not found. Set YOUTUBE_API_KEY in .env")
        
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
        if verbose:
            print("[*] YouTube Transcript Fetcher initialized")
            print("[!] Note: YouTube API has quota limits (10,000 units/day)")
            print("    - Each search costs ~100 units")
            print("    - Transcripts are fetched without using API quota")
    
    def get_channel_videos(self, channel_id, max_results=50):
        """Fetch video list from channel."""
        if self.verbose:
            print(f"\n[*] Fetching videos from channel...")
        
        videos = []
        request = self.youtube.search().list(
            part='id,snippet',
            channelId=channel_id,
            maxResults=min(max_results, 50),
            order='date',
            type='video'
        )
        
        response = request.execute()
        self.quota_used += 100
        
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            snippet = item['snippet']
            
            videos.append({
                'video_id': video_id,
                'title': snippet['title'],
                'description': snippet['description'],
                'published_at': snippet['publishedAt'],
                'thumbnail': snippet['thumbnails']['high']['url']
            })
        
        if self.verbose:
            print(f"[+] Found {len(videos)} videos")
        
        return videos
    
    def get_transcript(self, video_id):
        """Fetch transcript for a video (no API quota used)."""
        try:
            api = YouTubeTranscriptApi()
            transcript_obj = api.fetch(video_id)
            
            snippets = transcript_obj.snippets
            full_text = ' '.join([snippet.text for snippet in snippets])
            transcript_type = 'auto-generated' if transcript_obj.is_generated else 'manual'
            
            segments = [
                {'text': snippet.text, 'start': snippet.start, 'duration': snippet.duration}
                for snippet in snippets
            ]
            
            return {
                'video_id': video_id,
                'transcript_type': transcript_type,
                'language': transcript_obj.language,
                'language_code': transcript_obj.language_code,
                'segments': segments,
                'full_text': full_text,
                'fetched_at': datetime.now().isoformat()
            }
        except Exception as e:
            if self.verbose:
                print(f"[-] Error fetching transcript for {video_id}: {e}")
            return None
    
    def fetch_and_save(self, channel_id, max_videos=20):
        """Fetch videos and transcripts, save to files."""
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Fetch video list
        videos = self.get_channel_videos(channel_id, max_results=max_videos)
        
        if not videos:
            print("[-] No videos found")
            return False
        
        # Save metadata
        metadata_file = OUTPUT_DIR / 'videos_metadata.json'
        metadata = {v['video_id']: v for v in videos}
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"[+] Saved metadata to {metadata_file}")
        
        # Fetch transcripts
        print(f"\n[*] Fetching transcripts for {len(videos)} videos...")
        print("[!] Using rate limiting to avoid IP blocking (2-3s delay between requests)")
        
        fetched = 0
        for i, video in enumerate(videos, 1):
            video_id = video['video_id']
            title = video['title']
            
            print(f"\n[{i}/{len(videos)}] {title}")
            print(f"    Video ID: {video_id}")
            
            transcript = self.get_transcript(video_id)
            
            if transcript:
                # Save transcript
                transcript_file = OUTPUT_DIR / f"{video_id}.json"
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    json.dump(transcript, f, indent=2, ensure_ascii=False)
                
                fetched += 1
                print(f"    [+] Saved transcript ({len(transcript['segments'])} segments)")
            else:
                print(f"    [-] Failed to fetch transcript")
            
            # Rate limiting
            if i < len(videos):
                delay = 2.0 if i <= 5 else 3.0
                print(f"    [*] Waiting {delay}s...")
                time.sleep(delay)
        
        print(f"\n[+] Successfully fetched {fetched}/{len(videos)} transcripts")
        print(f"[*] API quota used: ~{self.quota_used} units")
        print(f"[*] Transcripts saved to: {OUTPUT_DIR}")
        
        return True


def test_fetch():
    """Test fetching a single video."""
    print("="*60)
    print("Testing YouTube Transcript Fetch")
    print("="*60)
    
    if not YOUTUBE_API_KEY:
        print("[-] YouTube API key not found. Set YOUTUBE_API_KEY in .env")
        return False
    
    # Test video ID (one of Josh's videos)
    test_video_id = "A_jHa0d1aFA"
    
    print(f"\n[*] Testing with video: {test_video_id}")
    
    try:
        fetcher = YouTubeTranscriptFetcher(YOUTUBE_API_KEY, verbose=True)
        
        # Test channel access
        print("\n[*] Testing channel access...")
        videos = fetcher.get_channel_videos(CHANNEL_ID, max_results=1)
        
        if videos:
            print(f"[+] Channel access OK - Found video: {videos[0]['title']}")
        else:
            print("[-] No videos found")
            return False
        
        # Test transcript fetch
        print(f"\n[*] Testing transcript fetch...")
        transcript = fetcher.get_transcript(test_video_id)
        
        if transcript:
            print(f"[+] Transcript fetch OK")
            print(f"    - Type: {transcript['transcript_type']}")
            print(f"    - Language: {transcript['language']}")
            print(f"    - Segments: {len(transcript['segments'])}")
            print(f"    - Text length: {len(transcript['full_text'])} characters")
            print(f"\n    Preview: {transcript['full_text'][:200]}...")
        else:
            print("[-] Failed to fetch transcript")
            return False
        
        print("\n[+] All tests passed!")
        return True
        
    except Exception as e:
        print(f"[-] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='YouTube operations')
    parser.add_argument('--fetch', action='store_true', help='Fetch transcripts')
    parser.add_argument('--count', type=int, default=20, help='Number of videos to fetch (default: 20)')
    parser.add_argument('--test', action='store_true', help='Test fetch with 1 video')
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    if args.test:
        test_fetch()
        return
    
    if args.fetch:
        print("="*60)
        print(f"Fetching YouTube Transcripts ({args.count} videos)")
        print("="*60)
        
        if not YOUTUBE_API_KEY:
            print("[-] YouTube API key not found. Set YOUTUBE_API_KEY in .env")
            return
        
        try:
            fetcher = YouTubeTranscriptFetcher(YOUTUBE_API_KEY, verbose=True)
            success = fetcher.fetch_and_save(CHANNEL_ID, max_videos=args.count)
            
            if success:
                print("\n" + "="*60)
                print("[+] Fetch completed successfully!")
                print("\n[*] Next steps:")
                print("    1. Ingest transcripts: python scripts/ingest.py --youtube")
                print("    2. Generate embeddings: python scripts/ingest.py --embeddings --youtube-only")
                print("="*60)
            
        except Exception as e:
            print(f"[-] Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
