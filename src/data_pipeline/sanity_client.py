"""
Sanity CMS API client for fetching blog content.
"""
import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class SanityClient:
    """Client for interacting with Sanity CMS API."""
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset: Optional[str] = None,
        token: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        """
        Initialize Sanity client.
        
        Args:
            project_id: Sanity project ID (defaults to env var)
            dataset: Dataset name (defaults to env var)
            token: API token (defaults to env var)
            api_version: API version (defaults to env var)
        """
        self.project_id = project_id or os.getenv('SANITY_PROJECT_ID')
        self.dataset = dataset or os.getenv('SANITY_DATASET', 'production')
        self.token = token or os.getenv('SANITY_API_TOKEN')
        self.api_version = api_version or os.getenv('SANITY_API_VERSION', '2023-05-03')
        
        if not self.project_id:
            raise ValueError("SANITY_PROJECT_ID is required")
        
        self.base_url = f"https://{self.project_id}.api.sanity.io/{self.api_version}/data/query/{self.dataset}"
    
    def fetch_articles(
        self,
        updated_since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch articles from Sanity CMS.
        
        Args:
            updated_since: Only fetch articles updated after this datetime
            limit: Maximum number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        # Build GROQ query
        query = '*[_type == "articles"'
        
        if updated_since:
            iso_date = updated_since.isoformat()
            query += f' && _updatedAt > "{iso_date}"'
        
        query += '] | order(_updatedAt desc)'
        
        if limit:
            query += f'[0...{limit}]'
        
        # Execute query
        params = {'query': query}
        headers = {}
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('result', [])
            
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"[-] Error fetching from Sanity: {e}")
            return []
    
    def fetch_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single article by ID.
        
        Args:
            article_id: Sanity document ID
            
        Returns:
            Article dictionary or None if not found
        """
        query = f'*[_type == "articles" && _id == "{article_id}"][0]'
        params = {'query': query}
        headers = {}
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('result')
            
        except requests.exceptions.RequestException as e:
            print(f"[-] Error fetching article {article_id}: {e}")
            return None
    
    def get_latest_update_time(self) -> Optional[datetime]:
        """
        Get the timestamp of the most recently updated article.
        
        Returns:
            Datetime of latest update or None
        """
        query = '*[_type == "articles"] | order(_updatedAt desc)[0]._updatedAt'
        params = {'query': query}
        headers = {}
        
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            timestamp = data.get('result')
            
            if timestamp:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"[-] Error fetching latest update time: {e}")
            return None


if __name__ == "__main__":
    client = SanityClient()
    
    print("[*] Testing Sanity API connection...")
    
    articles = client.fetch_articles(limit=5)
    
    if articles:
        print(f"[+] Fetched {len(articles)} articles")
        for article in articles:
            print(f"  - {article.get('title', 'Untitled')}")
    else:
        print("[-] No articles fetched")
    
    latest = client.get_latest_update_time()
    if latest:
        print(f"[*] Latest update: {latest}")
