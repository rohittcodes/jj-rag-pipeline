"""
RAG Retriever - Semantic search and content retrieval.
"""
import os
from datetime import date, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

from src.data_pipeline.embedding_generator import EmbeddingGenerator

load_dotenv()


@dataclass
class RetrievalResult:
    """Single retrieval result from semantic search."""
    chunk_id: int
    content_id: int
    chunk_text: str
    section_title: Optional[str]
    similarity: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class RAGRetriever:
    """
    RAG Retriever for semantic search over Josh's content.
    
    Uses vector similarity search with pgvector to find relevant content chunks
    based on user queries constructed from quiz responses.
    """
    
    def __init__(self, top_k: int = None, confidence_threshold: float = None, verbose: bool = False):
        """
        Initialize the RAG retriever.
        
        Args:
            top_k: Number of results to retrieve (default from env)
            confidence_threshold: Minimum similarity score (default 0.75)
            verbose: Whether to print initialization messages
        """
        self.embedding_generator = EmbeddingGenerator(verbose=verbose)
        self.top_k = top_k or int(os.getenv('TOP_K_RESULTS', 10))
        self.confidence_threshold = confidence_threshold or 0.75
        self.verbose = verbose
        self.min_content_date = self._parse_content_date_threshold()

        # Database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'josh_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        
        if verbose:
            print(f"[+] RAG Retriever initialized")
            print(f"    - Top K: {self.top_k}")
            print(f"    - Confidence threshold: {self.confidence_threshold}")
            if self.min_content_date:
                print(f"    - Min content date: {self.min_content_date}")
            print(f"    - Database: {self.db_config['database']}")

    def _parse_content_date_threshold(self) -> Optional[date]:
        """
        Parse CONTENT_MIN_PUBLISH_DATE (YYYY-MM-DD) or CONTENT_MAX_AGE_YEARS from env.
        Only content with publish_date (or updated_at) on or after this date is retrieved.
        """
        explicit = os.getenv("CONTENT_MIN_PUBLISH_DATE", "").strip()
        if explicit:
            try:
                return date.fromisoformat(explicit)
            except ValueError:
                pass
        years_str = os.getenv("CONTENT_MAX_AGE_YEARS", "").strip()
        if years_str:
            try:
                years = float(years_str)
                if years > 0:
                    return date.today() - timedelta(days=int(365 * years))
            except ValueError:
                pass
        return None

    def construct_query(self, quiz_response: Dict) -> str:
        """
        Construct a natural language query from quiz response.
        
        Args:
            quiz_response: Dictionary containing quiz answers
                - profession: List[str]
                - use_case: List[str]
                - budget: List[str]
                - portability: str
                - screen_size: List[str]
        
        Returns:
            Natural language query string
        """
        query_parts = []
        
        # Extract fields
        professions = quiz_response.get('profession', [])
        use_cases = quiz_response.get('use_case', [])
        budgets = quiz_response.get('budget', [])
        portability = quiz_response.get('portability', '')
        screen_sizes = quiz_response.get('screen_size', [])
        
        # Build query
        if professions:
            query_parts.append(f"laptop for {', '.join(professions).lower()}")
        
        if use_cases:
            query_parts.append(f"good for {', '.join(use_cases).lower()}")
        
        if budgets:
            budget_desc = {
                'budget': 'affordable budget',
                'value': 'good value',
                'premium': 'premium high-end'
            }
            budget_str = ', '.join([budget_desc.get(b, b) for b in budgets])
            query_parts.append(f"with {budget_str}")
        
        if portability:
            portability_map = {
                'light': 'lightweight and portable',
                'somewhat': 'balanced portability',
                'performance': 'prioritizing performance'
            }
            query_parts.append(portability_map.get(portability, portability))
        
        if screen_sizes:
            query_parts.append(f"{' or '.join(screen_sizes)} screen")
        
        # Combine into natural query
        if query_parts:
            query = "What are Josh's recommendations for a " + " ".join(query_parts) + "?"
        else:
            query = "What are Josh's laptop recommendations?"
        
        return query
    
    def retrieve(
        self,
        query: str = None,
        quiz_response: Dict = None,
        top_k: int = None,
        filters: Dict = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant content chunks using semantic search.
        
        Args:
            query: Direct query string (optional)
            quiz_response: Quiz response dict to construct query from (optional)
            top_k: Override default top_k
            filters: Additional filters (e.g., use_case_tags, content_type)
        
        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        # Construct query if not provided
        if query is None and quiz_response is not None:
            query = self.construct_query(quiz_response)
        elif query is None:
            raise ValueError("Either 'query' or 'quiz_response' must be provided")
        
        if self.verbose:
            print(f"\n[*] Query: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        if self.verbose:
            print(f"[+] Query embedding generated ({len(query_embedding)} dimensions)")
        
        # Connect to database
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Build SQL query
        k = top_k or self.top_k
        
        # Base query - UNION of blog chunks and YouTube chunks
        sql = """
            SELECT 
                cc.id as chunk_id,
                cc.content_id,
                cc.chunk_text,
                cc.section_title,
                cc.metadata,
                jc.title as content_title,
                jc.content_type,
                jc.url,
                jc.publish_date,
                jc.tags as use_case_tags,
                1 - (cc.embedding <=> %s::vector) as similarity,
                'blog' as source_type
            FROM content_chunks cc
            JOIN josh_content jc ON cc.content_id = jc.id
            WHERE cc.embedding IS NOT NULL
        """
        params = [query_embedding]

        if self.min_content_date:
            sql += " AND (COALESCE(jc.publish_date, jc.updated_at::date) >= %s)"
            params.append(self.min_content_date)
        
        # Add YouTube chunks
        sql += """
            UNION ALL
            SELECT 
                yc.id as chunk_id,
                yc.youtube_content_id as content_id,
                yc.chunk_text,
                NULL as section_title,
                yc.metadata,
                yt.title as content_title,
                'youtube' as content_type,
                yt.url,
                yt.publish_date,
                ARRAY['youtube', 'video'] as use_case_tags,
                1 - (yc.embedding <=> %s::vector) as similarity,
                'youtube' as source_type
            FROM youtube_chunks yc
            JOIN youtube_content yt ON yc.youtube_content_id = yt.id
            WHERE yc.embedding IS NOT NULL
        """
        params.append(query_embedding)
        
        if self.min_content_date:
            sql += " AND yt.publish_date >= %s"
            params.append(self.min_content_date)
        
        # Add test data chunks
        sql += """
            UNION ALL
            SELECT 
                td.id as chunk_id,
                td.config_id as content_id,
                td.chunk_text,
                td.test_type as section_title,
                td.benchmark_results as metadata,
                CONCAT(c.product_name, ' - ', td.test_type, ' Test Data') as content_title,
                'test_data' as content_type,
                COALESCE(c.test_data_pdf_url, '') as url,
                c.updated_at::date as publish_date,
                ARRAY['performance', 'benchmark', 'test'] as use_case_tags,
                1 - (td.embedding <=> %s::vector) as similarity,
                'test_data' as source_type
            FROM test_data_chunks td
            JOIN configs c ON td.config_id = c.config_id
            WHERE td.embedding IS NOT NULL
        """
        params.append(query_embedding)

        # Add product spec chunks
        sql += """
            UNION ALL
            SELECT 
                ps.id as chunk_id,
                ps.config_id as content_id,
                ps.chunk_text,
                'Technical Specifications' as section_title,
                NULL as metadata,
                c.product_name as content_title,
                'product_spec' as content_type,
                NULL as url,
                c.last_synced::date as publish_date,
                ARRAY['specs', 'technical', 'hardware'] as use_case_tags,
                1 - (ps.embedding <=> %s::vector) as similarity,
                'product_spec' as source_type
            FROM product_spec_chunks ps
            JOIN configs c ON ps.config_id = c.config_id
            WHERE ps.embedding IS NOT NULL
        """
        params.append(query_embedding)

        # Order by similarity and limit (applies to UNION result)
        sql += """
            ORDER BY similarity DESC
            LIMIT %s;
        """
        params.append(k)
        
        # Execute query
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # Convert to RetrievalResult objects
        results = []
        for row in rows:
            # Parse metadata to extract config_ids
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata)
            
            # Build metadata dict
            result_metadata = {
                'content_title': row['content_title'],
                'content_type': row['content_type'],
                'url': row['url'],
                'publish_date': str(row['publish_date']) if row['publish_date'] else None,
                'use_case_tags': row['use_case_tags'] or [],
                'source_type': row.get('source_type', 'blog')
            }
            
            # For test_data, the content_id is actually the config_id
            if row.get('source_type') == 'test_data':
                result_metadata['config_ids'] = [row['content_id']]
                result_metadata['benchmark_results'] = metadata  # benchmark_results from test_data_chunks
            else:
                result_metadata['config_ids'] = metadata.get('config_ids', [])
            
            result = RetrievalResult(
                chunk_id=row['chunk_id'],
                content_id=row['content_id'],
                chunk_text=row['chunk_text'],
                section_title=row['section_title'],
                similarity=float(row['similarity']),
                metadata=result_metadata
            )
            results.append(result)
        
        # Log results
        if self.verbose:
            print(f"[+] Retrieved {len(results)} chunks")
            if results:
                print(f"    - Top similarity: {results[0].similarity:.4f}")
                print(f"    - Lowest similarity: {results[-1].similarity:.4f}")
                
                # Check confidence
                if results[0].similarity < self.confidence_threshold:
                    print(f"[!] Warning: Low confidence (top similarity {results[0].similarity:.4f} < {self.confidence_threshold})")
            else:
                print("[!] Warning: No results found")
        
        return results
    
    def is_confident(self, results: List[RetrievalResult]) -> bool:
        """
        Check if retrieval results are confident enough.
        
        Args:
            results: List of retrieval results
        
        Returns:
            True if top result meets confidence threshold
        """
        if not results:
            return False
        return results[0].similarity >= self.confidence_threshold
    
    def get_stats(self, results: List[RetrievalResult]) -> Dict:
        """
        Get statistics about retrieval results.
        
        Args:
            results: List of retrieval results
        
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {
                'count': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'is_confident': False
            }
        
        similarities = [r.similarity for r in results]
        
        return {
            'count': len(results),
            'avg_similarity': sum(similarities) / len(similarities),
            'max_similarity': max(similarities),
            'min_similarity': min(similarities),
            'is_confident': self.is_confident(results),
            'unique_articles': len(set(r.content_id for r in results))
        }


if __name__ == "__main__":
    """Test the retriever with sample queries."""
    
    retriever = RAGRetriever(top_k=5)
    
    print("Test 1: Direct Query")
    results = retriever.retrieve(query="best laptop for programming and coding")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.metadata['content_title']} (similarity: {result.similarity:.4f})")
    
    print("\nTest 2: Quiz Response")
    quiz_response = {
        'profession': ['Student', 'Developer'],
        'use_case': ['programming'],
        'budget': ['value'],
        'portability': 'light',
        'screen_size': ['13-14 inches']
    }
    
    results = retriever.retrieve(quiz_response=quiz_response)
    
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result.metadata['content_title']} (similarity: {result.similarity:.4f})")
