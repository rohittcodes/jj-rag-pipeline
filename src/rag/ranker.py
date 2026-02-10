"""
RAG Ranker - Score and rank product recommendations.

This module takes retrieved content chunks and converts them into
ranked product recommendations by:
1. Extracting product mentions and config IDs
2. Scoring based on Josh's recommendations + spec matching
3. Ranking by combined confidence score
4. Generating explanations
"""
import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from dotenv import load_dotenv

from src.rag.retriever import RetrievalResult
from src.rag.product_client import ConfigDatabaseClient

load_dotenv()


@dataclass
class ProductRecommendation:
    """A ranked product recommendation."""
    # Product identification
    product_name: str
    config_id: Optional[int]  # Will be populated when connected to EC2 DB
    
    # Scoring
    confidence_score: float  # Combined score (0-1)
    josh_score: float  # Score from Josh's content (0-1)
    spec_score: float  # Score from spec matching (0-1)
    test_data_score: float  # Score from test data benchmarks (0-1)
    
    # Josh's opinion
    ranking: Optional[int]  # Josh's ranking (#1, #2, etc.)
    recommendation_type: str  # 'top_pick', 'recommended', 'mentioned'
    josh_quote: Optional[str]  # Quote from Josh
    
    # Context
    source_article: str
    source_url: str
    section_title: Optional[str]
    
    # Explanation
    explanation: str
    pros: List[str]
    cons: List[str]
    who_is_this_for: Optional[str]
    
    # Metadata
    similarity: float  # Retrieval similarity score
    chunk_ids: List[int]  # Source chunks
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class RAGRanker:
    """
    Rank products from retrieved content chunks.
    
    Combines Josh's recommendations with spec matching to generate
    ranked product recommendations with explanations.
    """
    
    def __init__(
        self,
        josh_weight: float = 0.60,
        spec_weight: float = 0.25,
        test_data_weight: float = 0.15,
        use_ec2_db: bool = False,
        ec2_db_config: Optional[Dict] = None,
        verbose: bool = False
    ):
        """
        Initialize the ranker.
        
        Args:
            josh_weight: Weight for Josh's recommendations (default 0.60)
            spec_weight: Weight for spec matching (default 0.25)
            test_data_weight: Weight for test data benchmarks (default 0.15)
            use_ec2_db: Whether to connect to EC2 product database
            ec2_db_config: EC2 database connection config
            verbose: Whether to print initialization messages
        """
        self.josh_weight = josh_weight
        self.spec_weight = spec_weight
        self.test_data_weight = test_data_weight
        self.manual_tag_weight = 0.5  # Boost for manual tag match
        self.similarity_weight = 0.4  # Weight for retrieval similarity (query relevance)
        
        self.use_ec2_db = use_ec2_db
        self.ec2_db_config = ec2_db_config
        self.verbose = verbose
        
        # Mapping from quiz use_case strings to DB use_case_ids
        self.manual_use_case_mapping = {
            'student': [1],          # Student
            'gaming': [2],           # Gamer
            'video_editing': [3, 8], # Video Editor, Graphic Designer
            'programming': [4, 5, 12], # Data Scientist, Programmer, Engineer
            'content_creation': [3, 8, 10, 11], # Editor, Designer, Photo, Audio
            'business': [9, 14],     # Trader, Corporate
            'general': [6]           # Everyday
        }
        
        # Initialize config database client (uses local synced configs)
        try:
            self.config_client = ConfigDatabaseClient()
        except ValueError as e:
            if verbose:
                print(f"[!] Warning: ConfigDatabaseClient not available: {e}")
                print(f"[!] Ranker will work with limited functionality")
            self.config_client = None
        
        # Local database for content
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'josh_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        
        if verbose:
            print(f"[+] RAG Ranker initialized")
            print(f"    - Josh weight: {self.josh_weight}")
            print(f"    - Spec weight: {self.spec_weight}")
            print(f"    - Test data weight: {self.test_data_weight}")
            print(f"    - Similarity weight: {self.similarity_weight}")
            print(f"    - Config database: Local (synced)")
            config_count = self.config_client.get_config_count()
            print(f"    - Available configs: {config_count}")
    
    def rank(
        self,
        retrieval_results: List[RetrievalResult],
        quiz_response: Dict,
        top_k: int = 5
    ) -> List[ProductRecommendation]:
        """
        Rank products from retrieval results.
        
        Args:
            retrieval_results: Results from RAG retriever
            quiz_response: User's quiz response for spec matching
            top_k: Number of recommendations to return
        
        Returns:
            List of ranked ProductRecommendation objects
        """
        if not retrieval_results:
            if self.verbose:
                print("[!] No retrieval results to rank")
            return []

        self.current_quiz_response = quiz_response

        if self.verbose:
            print(f"\n[*] Ranking products from {len(retrieval_results)} chunks...")
        
        # Step 1: Group chunks by config_id (no DB calls yet)
        product_candidates = self._group_chunks_by_config(retrieval_results)
        if self.verbose:
            print(f"[+] Found {len(product_candidates)} unique product candidates")
        
        # Step 2: Calculate preliminary scores (chunk-based only, no DB)
        scored_candidates = []
        for candidate in product_candidates:
            # Calculate Josh score (based on chunk content)
            josh_score = self._calculate_josh_score_from_chunks(candidate)
            
            # Calculate test data score (based on chunk content)
            test_data_score = self._calculate_test_data_score_from_chunks(candidate, quiz_response)
            
            # Preliminary score (no spec score yet - requires DB)
            preliminary_score = (josh_score * self.josh_weight) + (test_data_score * self.test_data_weight)
            
            scored_candidates.append({
                'candidate': candidate,
                'josh_score': josh_score,
                'test_data_score': test_data_score,
                'preliminary_score': preliminary_score
            })
        
        # Step 3: Sort by preliminary score and take top candidates
        scored_candidates.sort(key=lambda x: x['preliminary_score'], reverse=True)
        top_candidates = scored_candidates[:top_k * 2]  # Get 2x to have buffer for spec filtering
        
        if self.verbose:
            print(f"[+] Selected top {len(top_candidates)} candidates for detailed scoring")
        
        # Step 4: Batch fetch ALL config details (names, use cases, properties) for top candidates
        config_ids = [c['candidate']['config_id'] for c in top_candidates if c['candidate'].get('config_id')]
        
        # Batch fetch all data in parallel (conceptually)
        product_names = self.config_client.get_product_names_batch(config_ids) if config_ids else {}
        use_case_map = self.config_client.get_config_use_cases(config_ids) if config_ids else {}
        
        # KEY OPTIMIZATION: Fetch full config objects with properties in one batch query
        # This prevents N+1 queries inside the loop below
        config_map = self.config_client.get_configs_by_ids(config_ids, include_properties=True) if config_ids else {}
        
        if self.verbose and config_ids:
            print(f"[*] Fetched names for {len(product_names)} configs")
            print(f"[*] Fetched manual tags for {len(use_case_map)} configs")
            print(f"[*] Fetched full details for {len(config_map)} configs")
        
        # Step 5: Final scoring with spec matching AND manual tags (only for top candidates)
        recommendations = []
        for scored in top_candidates:
            candidate = scored['candidate']
            config_id = candidate.get('config_id')
            product_name = product_names.get(config_id, f"Config {config_id}") if config_id else candidate.get('product_name', 'Unknown')
            
            # Get full config object from batch map
            config = config_map.get(config_id) if config_id else None
            
            # Calculate spec score (uses pre-fetched config to avoid DB hits)
            spec_score = self._calculate_spec_score(product_name, quiz_response, config_id, config_obj=config)
            
            # Calculate manual tag score
            manual_score = 0.0
            if config_id and config_id in use_case_map:
                manual_score = self._calculate_manual_use_case_score(use_case_map[config_id], quiz_response)
            
            # Final combined score - NOW INCLUDING SIMILARITY (query relevance)
            # This ensures query-specific results instead of always showing the same products
            similarity_score = candidate['max_similarity']  # Highest similarity from retrieved chunks
            
            confidence_score = (scored['josh_score'] * self.josh_weight) + \
                             (spec_score * self.spec_weight) + \
                             (scored['test_data_score'] * self.test_data_weight) + \
                             (manual_score * self.manual_tag_weight) + \
                             (similarity_score * self.similarity_weight)
            
            # Normalize to max 1.0
            confidence_score = min(1.0, confidence_score)
            
            # Generate explanation
            explanation = self._generate_explanation_from_candidate(
                product_name, candidate, quiz_response
            )
            
            # Create recommendation
            recommendation = ProductRecommendation(
                product_name=product_name,
                config_id=config_id,
                confidence_score=confidence_score,
                josh_score=scored['josh_score'],
                spec_score=spec_score,
                test_data_score=scored['test_data_score'],
                ranking=candidate.get('ranking'),
                recommendation_type=candidate.get('recommendation_type', 'mentioned'),
                josh_quote=candidate.get('josh_quote'),
                source_article=candidate['source_article'],
                source_url=candidate['source_url'],
                section_title=candidate.get('section_title'),
                explanation=explanation,
                pros=candidate.get('pros', []),
                cons=candidate.get('cons', []),
                who_is_this_for=candidate.get('who_is_this_for'),
                similarity=candidate['max_similarity'],
                chunk_ids=candidate['chunk_ids']
            )
            
            recommendations.append(recommendation)
        
        # Step 6: Final sort by complete confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Step 7: Deduplicate by product name (keep highest scoring config)
        seen_products = set()
        unique_recommendations = []
        
        for rec in recommendations:
            # Normalize product name for comparison
            name_key = rec.product_name.lower().strip()
            
            # Simple heuristic to handle slight variations (e.g. "Blade 16" vs "Razer Blade 16")
            # If we've already seen a product that contains or is contained by this name, skip
            is_duplicate = False
            for seen in seen_products:
                if name_key in seen or seen in name_key:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_recommendations.append(rec)
                seen_products.add(name_key)
        
        if self.verbose:
            print(f"[+] Ranked {len(unique_recommendations)} unique products (deduplicated from {len(recommendations)})")
            if unique_recommendations:
                print(f"    - Top recommendation: {unique_recommendations[0].product_name}")
                print(f"    - Confidence: {unique_recommendations[0].confidence_score:.3f}")
        
        # Step 8: Filter by brand preferences/exclusions
        filtered_recommendations = self._filter_by_brand_preferences(unique_recommendations, quiz_response)
        
        if self.verbose and len(filtered_recommendations) < len(unique_recommendations):
            print(f"[*] Filtered out {len(unique_recommendations) - len(filtered_recommendations)} products based on brand preferences")
        
        return filtered_recommendations[:top_k]
    
    def _group_chunks_by_config(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> List[Dict]:
        """
        Group chunks by config_id without fetching product names yet.
        Returns list of candidate dictionaries with aggregated chunk data.
        """
        # Group by config_id first (most reliable)
        configs = defaultdict(lambda: {
            'chunk_ids': [],
            'chunks': [],
            'similarities': [],
            'rankings': [],
            'quotes': [],
            'pros': [],
            'cons': [],
            'sections': [],
            'chunk_texts': [],
            'test_data': [],
            'sentiments': []  # Store sentiment data for each chunk
        })
        
        for result in retrieval_results:
            chunk_text = result.chunk_text
            metadata = result.metadata
            section_title = result.section_title
            
            # Extract ranking from section title (e.g., "#1 - MacBook Pro 14")
            ranking = None
            if section_title:
                ranking_match = re.search(r'#(\d+)', section_title)
                if ranking_match:
                    ranking = int(ranking_match.group(1))
            
            # Extract pros/cons if present
            pros = self._extract_list_items(chunk_text, r'\*\*Pros?:?\*\*\s*\n((?:[-•]\s*.+\n?)+)')
            cons = self._extract_list_items(chunk_text, r'\*\*Cons?:?\*\*\s*\n((?:[-•]\s*.+\n?)+)')
            
            # Extract quotes
            quotes = re.findall(r'"([^"]{20,200})"', chunk_text)
            
            # Get config_ids from metadata
            config_ids = metadata.get('config_ids', [])
            if isinstance(config_ids, int):
                config_ids = [config_ids]
            
            if config_ids:
                for config_id in config_ids:
                    key = f"config_{config_id}"
                    configs[key]['config_id'] = config_id
                    configs[key]['chunk_ids'].append(result.chunk_id)
                    configs[key]['chunks'].append(result)
                    configs[key]['similarities'].append(result.similarity)
                    configs[key]['source_article'] = metadata['content_title']
                    configs[key]['source_url'] = metadata['url']
                    
                    if section_title:
                        configs[key]['sections'].append(section_title)
                        configs[key]['section_title'] = section_title
                    
                    if ranking:
                        configs[key]['rankings'].append(ranking)
                    
                    if pros:
                        configs[key]['pros'].extend(pros)
                    
                    if cons:
                        configs[key]['cons'].extend(cons)
                    
                    if quotes:
                        configs[key]['quotes'].extend(quotes)
                    
                    configs[key]['chunk_texts'].append(chunk_text[:500])
                    
                    # Store sentiment data if present (per-product)
                    sentiments_dict = metadata.get('sentiments', {})
                    if sentiments_dict and str(config_id) in sentiments_dict:
                        # Get sentiment specific to this config_id
                        sentiment = sentiments_dict[str(config_id)]
                        configs[key]['sentiments'].append({
                            'score': sentiment.get('sentiment_score', 0.0),
                            'label': sentiment.get('sentiment_label', 'neutral'),
                            'context': sentiment.get('context_type', 'mention'),
                            'reasoning': sentiment.get('reasoning', '')
                        })
                    
                    # Store test data if present (with parsed benchmark scores)
                    if metadata.get('source_type') == 'test_data':
                        configs[key]['test_data'].append({
                            'benchmark_name': metadata.get('benchmark_name'),
                            'benchmark_category': metadata.get('benchmark_category'),
                            'benchmark_results': metadata.get('benchmark_results', {}),  # Include parsed scores
                            'content': chunk_text
                        })
        
        # Convert to list of candidates
        candidates = []
        for key, data in configs.items():
            data['max_similarity'] = max(data['similarities']) if data['similarities'] else 0.0
            data['avg_similarity'] = sum(data['similarities']) / len(data['similarities']) if data['similarities'] else 0.0
            candidates.append(data)
        
        return candidates
    
    def _extract_products(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> Dict[str, Dict]:
        """
        Extract product mentions from retrieval results.
        
        Strategy:
        1. Prioritize config_ids from metadata (direct mapping)
        2. Fall back to product name extraction for content without config_ids
        
        Returns:
            Dict mapping unique_key -> product_data
        """
        # Group by config_id first (most reliable)
        configs = defaultdict(lambda: {
            'chunk_ids': [],
            'chunks': [],  # Store actual RetrievalResult objects
            'similarities': [],
            'rankings': [],
            'quotes': [],
            'pros': [],
            'cons': [],
            'sections': [],
            'chunk_texts': []
        })
        
        # Track chunks without config_ids for name-based extraction
        chunks_without_configs = []
        
        for result in retrieval_results:
            chunk_text = result.chunk_text
            section_title = result.section_title
            
            # Extract ranking from section title (e.g., "#1 - MacBook Pro 14")
            ranking = None
            if section_title:
                ranking_match = re.search(r'#(\d+)', section_title)
                if ranking_match:
                    ranking = int(ranking_match.group(1))
            
            # Extract config IDs from metadata
            config_ids = result.metadata.get('config_ids', [])
            
            # Extract pros/cons if present
            pros = self._extract_list_items(chunk_text, r'\*\*Pros?:?\*\*\s*\n((?:[-•]\s*.+\n?)+)')
            cons = self._extract_list_items(chunk_text, r'\*\*Cons?:?\*\*\s*\n((?:[-•]\s*.+\n?)+)')
            
            # Extract quotes
            quotes = re.findall(r'"([^"]{20,200})"', chunk_text)
            
            if config_ids:
                # Group by config_id (can have multiple configs per chunk)
                for config_id in config_ids:
                    key = f"config_{config_id}"
                    configs[key]['config_id'] = config_id
                    configs[key]['chunk_ids'].append(result.chunk_id)
                    configs[key]['chunks'].append(result)  # Store the actual result object
                    configs[key]['similarities'].append(result.similarity)
                    configs[key]['source_article'] = result.metadata['content_title']
                    configs[key]['source_url'] = result.metadata['url']
                    
                    if section_title:
                        configs[key]['sections'].append(section_title)
                        configs[key]['section_title'] = section_title
                    
                    if ranking:
                        configs[key]['rankings'].append(ranking)
                    
                    if pros:
                        configs[key]['pros'].extend(pros)
                    
                    if cons:
                        configs[key]['cons'].extend(cons)
                    
                    if quotes:
                        configs[key]['quotes'].extend(quotes)
                    
                    configs[key]['chunk_texts'].append(chunk_text[:500])
            else:
                # No config_id, save for name-based extraction
                chunks_without_configs.append({
                    'result': result,
                    'ranking': ranking,
                    'pros': pros,
                    'cons': cons,
                    'quotes': quotes
                })
        
        # Process config-based products
        # Batch fetch all product names at once (fast - from local DB)
        config_ids = [data['config_id'] for data in configs.values()]
        product_names = self.config_client.get_product_names_batch(config_ids)
        
        processed_products = {}
        for key, data in configs.items():
            config_id = data['config_id']
            
            # Get product name from batch fetch
            product_name = product_names.get(config_id, f"Config {config_id}")
            
            processed_products[key] = {
                'product_name': product_name,
                'config_id': config_id,
                'chunk_ids': data['chunk_ids'],
                'chunks': data['chunks'],  # Include the actual RetrievalResult objects
                'max_similarity': max(data['similarities']) if data['similarities'] else 0.0,
                'avg_similarity': sum(data['similarities']) / len(data['similarities']) if data['similarities'] else 0.0,
                'ranking': min(data['rankings']) if data['rankings'] else None,
                'recommendation_type': 'top_pick' if (data['rankings'] and min(data['rankings']) == 1) else 'recommended',
                'josh_quote': data['quotes'][0] if data['quotes'] else None,
                'pros': list(set(data['pros']))[:5],
                'cons': list(set(data['cons']))[:5],
                'source_article': data['source_article'],
                'source_url': data['source_url'],
                'section_title': data.get('section_title'),
                'chunk_texts': data['chunk_texts']
            }
        
        # Fall back to name-based extraction for chunks without config_ids
        if chunks_without_configs:
            name_based_products = self._extract_products_by_name(chunks_without_configs)
            processed_products.update(name_based_products)
        
        return processed_products
    
    def _extract_products_by_name(self, chunks_data: List[Dict]) -> Dict[str, Dict]:
        """
        Extract products by name for chunks without config_ids.
        This is a fallback for content that doesn't have config mappings.
        """
        products = defaultdict(lambda: {
            'chunk_ids': [],
            'chunks': [],  # Store actual RetrievalResult objects
            'similarities': [],
            'rankings': [],
            'quotes': [],
            'pros': [],
            'cons': [],
            'sections': [],
            'chunk_texts': []
        })
        
        # Common laptop patterns
        laptop_patterns = [
            r'(MacBook (?:Pro|Air) \d+)',
            r'(ThinkPad [A-Z]\d+[a-z]?)',
            r'(Dell (?:XPS|Precision|Latitude) \d+)',
            r'(HP (?:Spectre|Envy|EliteBook|Omen|Victus|ProBook) [\w\s]+\d+)',
            r'(Lenovo (?:Legion|Yoga|IdeaPad|ThinkBook) [\w\s]+\d+)',
            r'(ASUS (?:ROG|TUF|Zenbook|Vivobook|ProArt) [\w\s]+\d+)',
            r'(Acer (?:Predator|Nitro|Swift|Aspire) [\w\s]+\d+)',
            r'(MSI (?:Stealth|Raider|Katana|Thin|Prestige) [\w\s]+\d+)',
            r'(Razer Blade \d+)',
            r'(LG Gram \d+)',
            r'(Surface (?:Laptop|Book) \d+)',
            r'(Framework Laptop \d+)',
            r'(Gigabyte (?:Aero|Gaming) [\w\s]+\d+)',
        ]
        
        for chunk_data in chunks_data:
            result = chunk_data['result']
            chunk_text = result.chunk_text
            section_title = result.section_title
            
            # Find laptop mentions in text
            found_products = set()
            for pattern in laptop_patterns:
                matches = re.findall(pattern, chunk_text, re.IGNORECASE)
                found_products.update(matches)
            
            # Also check section title
            if section_title:
                for pattern in laptop_patterns:
                    matches = re.findall(pattern, section_title, re.IGNORECASE)
                    found_products.update(matches)
            
            # Store product data
            for product in found_products:
                product = product.strip()
                key = f"name_{product}"
                
                products[key]['product_name'] = product
                products[key]['chunk_ids'].append(result.chunk_id)
                products[key]['chunks'].append(result)  # Store the actual result object
                products[key]['similarities'].append(result.similarity)
                products[key]['source_article'] = result.metadata['content_title']
                products[key]['source_url'] = result.metadata['url']
                
                if section_title:
                    products[key]['sections'].append(section_title)
                    products[key]['section_title'] = section_title
                
                if chunk_data['ranking']:
                    products[key]['rankings'].append(chunk_data['ranking'])
                
                if chunk_data['pros']:
                    products[key]['pros'].extend(chunk_data['pros'])
                
                if chunk_data['cons']:
                    products[key]['cons'].extend(chunk_data['cons'])
                
                if chunk_data['quotes']:
                    products[key]['quotes'].extend(chunk_data['quotes'])
                
                products[key]['chunk_texts'].append(chunk_text[:500])
        
        # Process name-based products and map to config_ids when possible
        processed = {}
        quiz = getattr(self, 'current_quiz_response', {})
        for key, data in products.items():
            product_name = data['product_name']
            chunk_context = data['chunk_texts'][0] if data.get('chunk_texts') else None
            config_id = self.config_client.find_best_config_for_product(
                product_name=product_name,
                quiz_response=quiz,
                chunk_context=chunk_context
            )
            if config_id:
                config = self.config_client.get_config_by_id(config_id)
                if config:
                    product_name = config.get('product_name', product_name)
            processed[key] = {
                'product_name': product_name,
                'config_id': config_id,
                'chunk_ids': data['chunk_ids'],
                'chunks': data['chunks'],  # Include the actual RetrievalResult objects
                'max_similarity': max(data['similarities']) if data['similarities'] else 0.0,
                'avg_similarity': sum(data['similarities']) / len(data['similarities']) if data['similarities'] else 0.0,
                'ranking': min(data['rankings']) if data['rankings'] else None,
                'recommendation_type': 'top_pick' if (data['rankings'] and min(data['rankings']) == 1) else 'recommended',
                'josh_quote': data['quotes'][0] if data['quotes'] else None,
                'pros': list(set(data['pros']))[:5],
                'cons': list(set(data['cons']))[:5],
                'source_article': data['source_article'],
                'source_url': data['source_url'],
                'section_title': data.get('section_title'),
                'chunk_texts': data['chunk_texts']
            }
        
        return processed
    
    def _extract_list_items(self, text: str, pattern: str) -> List[str]:
        """Extract list items from text using regex pattern."""
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if not match:
            return []
        
        list_text = match.group(1)
        items = re.findall(r'[-•]\s*(.+)', list_text)
        return [item.strip() for item in items if item.strip()]
    
    def _calculate_josh_score_from_chunks(self, candidate: Dict) -> float:
        """
        Calculate score based on Josh's recommendation from chunk data.
        No DB calls - uses only chunk metadata.
        
        Factors:
        - Ranking (#1 = highest score)
        - Similarity score (how relevant the content is)
        - Number of mentions (only positive/neutral)
        - Sentiment multiplier (boosts positive, penalizes negative)
        """
        # Calculate base score
        base_score = 0.0
        
        # Ranking bonus (0.5 weight)
        if candidate.get('rankings'):
            ranking = min(candidate['rankings'])  # Best ranking
            # #1 = 1.0, #2 = 0.9, #3 = 0.8, etc.
            rank_score = max(0.0, 1.0 - (ranking - 1) * 0.1)
            base_score += rank_score * 0.5
        else:
            # No explicit ranking, but mentioned
            base_score += 0.3
        
        # Similarity bonus (0.3 weight)
        similarity = candidate.get('max_similarity', 0.0)
        base_score += similarity * 0.3
        
        # Multiple mentions bonus (0.2 weight) - only count positive/neutral mentions
        sentiments = candidate.get('sentiments', [])
        if sentiments:
            # Only count positive and neutral mentions
            positive_mentions = [
                s for s in sentiments 
                if s['label'] in ['highly_positive', 'positive', 'neutral']
            ]
            num_mentions = len(positive_mentions)
        else:
            # No sentiment data, count all mentions
            num_mentions = len(candidate.get('chunk_ids', []))
        
        mention_score = min(1.0, num_mentions / 3.0)  # Cap at 3 mentions
        base_score += mention_score * 0.2
        
        # Apply sentiment multiplier
        sentiment_multiplier = self._calculate_sentiment_multiplier(candidate)
        final_score = base_score * sentiment_multiplier
        
        return min(1.0, final_score)
    
    def _calculate_sentiment_multiplier(self, candidate: Dict) -> float:
        """
        Calculate sentiment multiplier based on chunk sentiments.
        
        Returns multiplier between 0.1 and 1.2:
        - highly_positive: 1.2x (boost 20%)
        - positive: 1.0x (no change)
        - neutral: 0.7x (reduce 30%)
        - negative: 0.3x (reduce 70%)
        - highly_negative: 0.1x (reduce 90%)
        """
        sentiments = candidate.get('sentiments', [])
        
        if not sentiments:
            # No sentiment data, treat as neutral (no penalty)
            return 1.0
        
        # Calculate average sentiment score
        avg_sentiment = sum(s['score'] for s in sentiments) / len(sentiments)
        
        # Count sentiment labels
        label_counts = {}
        for s in sentiments:
            label = s['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Determine dominant sentiment
        dominant_label = max(label_counts, key=label_counts.get)
        
        # Apply multiplier based on dominant sentiment
        multipliers = {
            'highly_positive': 1.2,
            'positive': 1.0,
            'neutral': 0.7,
            'negative': 0.3,
            'highly_negative': 0.1
        }
        
        multiplier = multipliers.get(dominant_label, 1.0)
        
        # Special case: If ALL mentions are negative, heavily penalize
        all_negative = all(s['label'] in ['negative', 'highly_negative'] for s in sentiments)
        if all_negative:
            multiplier = 0.1
        
        # Special case: If mix of positive and negative, use average sentiment score
        has_positive = any(s['label'] in ['highly_positive', 'positive'] for s in sentiments)
        has_negative = any(s['label'] in ['negative', 'highly_negative'] for s in sentiments)
        
        if has_positive and has_negative:
            # Mixed sentiment - use average score to determine multiplier
            if avg_sentiment > 0.5:
                multiplier = 1.1  # Mostly positive
            elif avg_sentiment > 0.0:
                multiplier = 0.9  # Slightly positive
            elif avg_sentiment > -0.5:
                multiplier = 0.5  # Slightly negative
            else:
                multiplier = 0.2  # Mostly negative
        
        return multiplier
    
    
    def _calculate_test_data_score_from_chunks(self, candidate: Dict, quiz_response: Dict) -> float:
        """
        TRUE RAG-BASED test data scoring using vector similarity search.
        
        Instead of keyword matching, this:
        1. Constructs a semantic query based on user's use case
        2. Performs vector search on test_data_chunks embeddings
        3. Retrieves relevant benchmarks based on similarity
        4. Scores based on retrieved benchmark data
        
        Args:
            candidate: Product candidate with config_id
            quiz_response: User's quiz response with use cases
            
        Returns:
            Score between 0.0 and 1.0
        """
        config_id = candidate.get('config_id')
        if not config_id:
            return 0.0
        
        # Extract user requirements
        use_cases = quiz_response.get('use_case', [])
        extracted_reqs = quiz_response.get('extracted_requirements', {})
        needs_long_battery = extracted_reqs.get('needs_long_battery', False)
        
        if not use_cases and not needs_long_battery:
            return 0.0
        
        # 1. Build semantic query based on use case
        query_text = self._build_test_data_query(use_cases, quiz_response)
        
        if not query_text:
            return 0.0
        
        # 2. Retrieve relevant test data using vector search
        try:
            relevant_benchmarks = self._retrieve_relevant_test_data(
                config_id=config_id,
                query_text=query_text,
                top_k=5
            )
            
            if not relevant_benchmarks:
                return 0.0
            
            # 3. Score based on retrieved benchmarks
            score = self._score_from_retrieved_benchmarks(
                relevant_benchmarks,
                use_cases,
                quiz_response
            )
            
            return score
            
        except Exception as e:
            if self.verbose:
                print(f"[!] Error in RAG test data scoring for config {config_id}: {e}")
            return 0.0
    
    def _build_test_data_query(self, use_cases: List[str], quiz_response: Dict) -> str:
        """
        Build a semantic query for test data retrieval based on user's use case.
        
        This query will be embedded and used for vector similarity search.
        """
        query_parts = []
        
        # Gaming use case
        if 'gaming' in use_cases:
            query_parts.append("gaming performance FPS frame rate graphics benchmark 3DMark")
        
        # Video editing use case
        if 'video_editing' in use_cases:
            query_parts.append("video editing rendering export encoding performance Premiere Blender")
        
        # Programming/data science use case
        if 'programming' in use_cases or 'data_science' in use_cases:
            query_parts.append("CPU performance multi-core Geekbench Cinebench compile time")
        
        # Battery life
        extracted_reqs = quiz_response.get('extracted_requirements', {})
        if extracted_reqs.get('needs_long_battery') or quiz_response.get('portability') in ['light', 'somewhat']:
            query_parts.append("battery life runtime hours endurance")
        
        # Portability (weight)
        if quiz_response.get('portability') == 'light':
            query_parts.append("weight portability lightweight")
        
        return " ".join(query_parts)
    
    def _retrieve_relevant_test_data(
        self,
        config_id: int,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant test data chunks using vector similarity search.
        
        This is TRUE RAG - using embeddings and cosine similarity!
        """
        # Import embedding generator (lazy import to avoid circular dependency)
        from src.data_pipeline.embedding_generator import EmbeddingGenerator
        
        # Generate query embedding
        embedding_gen = EmbeddingGenerator(verbose=False)
        query_embedding = embedding_gen.generate_embedding(query_text)
        
        # Vector search on test_data_chunks
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Use pgvector's cosine similarity operator (<=>)
        cursor.execute("""
            SELECT 
                test_type,
                test_description,
                benchmark_results,
                chunk_text,
                1 - (embedding <=> %s::vector) as similarity
            FROM test_data_chunks
            WHERE config_id = %s
                AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, config_id, query_embedding, top_k))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if self.verbose and results:
            print(f"[*] Retrieved {len(results)} relevant test data chunks via vector search")
            print(f"    Top similarity: {results[0]['similarity']:.3f}")
        
        return results
    
    def _score_from_retrieved_benchmarks(
        self,
        benchmarks: List[Dict],
        use_cases: List[str],
        quiz_response: Dict
    ) -> float:
        """
        Score product based on retrieved benchmark data.
        
        Uses the actual benchmark_results JSON data from vector-retrieved chunks.
        """
        if not benchmarks:
            return 0.0
        
        scores = []
        
        for benchmark in benchmarks:
            similarity = benchmark.get('similarity', 0.0)
            benchmark_data = benchmark.get('benchmark_results') or {}
            chunk_text = benchmark.get('chunk_text', '').lower()
            
            # Weight by retrieval similarity (RAG confidence)
            similarity_weight = similarity
            
            # Gaming performance
            if 'gaming' in use_cases:
                gaming_score = self._extract_gaming_score(benchmark_data, chunk_text)
                if gaming_score > 0:
                    scores.append(gaming_score * similarity_weight)
            
            # Video editing performance
            if 'video_editing' in use_cases:
                video_score = self._extract_video_score(benchmark_data, chunk_text)
                if video_score > 0:
                    scores.append(video_score * similarity_weight)
            
            # CPU performance
            if 'programming' in use_cases or 'data_science' in use_cases:
                cpu_score = self._extract_cpu_score(benchmark_data, chunk_text)
                if cpu_score > 0:
                    scores.append(cpu_score * similarity_weight)
            
            # Battery life
            extracted_reqs = quiz_response.get('extracted_requirements', {})
            if extracted_reqs.get('needs_long_battery') or quiz_response.get('portability') in ['light', 'somewhat']:
                battery_score = self._extract_battery_score(benchmark_data, chunk_text)
                if battery_score > 0:
                    scores.append(battery_score * similarity_weight)
        
        # Return average weighted score
        return sum(scores) / len(scores) if scores else 0.0
    
    def _extract_gaming_score(self, benchmark_data: Dict, chunk_text: str) -> float:
        """Extract gaming performance score from benchmark data."""
        # Check structured benchmark_results JSON first
        if benchmark_data:
            # Look for FPS values in benchmark data
            fps_values = []
            
            # Check for 3DMark scores
            if '3dmark' in benchmark_data:
                scores = benchmark_data['3dmark'].get('scores', {})
                # 3DMark scores: normalize to 0-1 scale
                # Timespy: 2000-8000 range
                if 'timespy' in scores:
                    timespy = scores['timespy']
                    if timespy >= 6000:
                        return 1.0
                    elif timespy >= 4000:
                        return 0.8
                    elif timespy >= 2500:
                        return 0.6
                    else:
                        return 0.4
            
            # Check for direct FPS values
            for key, value in benchmark_data.items():
                if 'fps' in key.lower() and isinstance(value, (int, float)):
                    fps_values.append(float(value))
            
            if fps_values:
                avg_fps = sum(fps_values) / len(fps_values)
                if avg_fps >= 60:
                    return 1.0
                elif avg_fps >= 45:
                    return 0.8
                elif avg_fps >= 30:
                    return 0.6
                else:
                    return 0.4
        
        # Fallback: semantic understanding from chunk text
        # (This is where RAG shines - understanding context!)
        if any(term in chunk_text for term in ['excellent gaming', 'high fps', 'smooth gameplay', '60+ fps']):
            return 0.9
        elif any(term in chunk_text for term in ['good gaming', 'playable', 'decent fps']):
            return 0.7
        elif any(term in chunk_text for term in ['poor gaming', 'low fps', 'stuttering']):
            return 0.3
        
        return 0.5
    
    def _extract_video_score(self, benchmark_data: Dict, chunk_text: str) -> float:
        """Extract video editing performance score from benchmark data."""
        if benchmark_data:
            # Check for rendering/export times
            for key, value in benchmark_data.items():
                key_lower = key.lower()
                if any(term in key_lower for term in ['render', 'export', 'encode']):
                    if isinstance(value, (int, float)):
                        # Lower time = better (assume minutes)
                        if value <= 5:
                            return 1.0
                        elif value <= 10:
                            return 0.8
                        elif value <= 20:
                            return 0.6
                        else:
                            return 0.4
        
        # Semantic understanding
        if any(term in chunk_text for term in ['fast render', 'quick export', 'excellent for editing']):
            return 0.9
        elif any(term in chunk_text for term in ['slow render', 'long export', 'struggles with editing']):
            return 0.3
        
        return 0.5
    
    def _extract_cpu_score(self, benchmark_data: Dict, chunk_text: str) -> float:
        """Extract CPU performance score from benchmark data."""
        if benchmark_data:
            # Check for Geekbench scores
            if 'geekbench' in benchmark_data:
                scores = benchmark_data['geekbench'].get('scores', {})
                if 'multi_core' in scores:
                    multi_score = scores['multi_core']
                    # Geekbench multi-core: 5000-15000 range
                    if multi_score >= 10000:
                        return 1.0
                    elif multi_score >= 7000:
                        return 0.8
                    elif multi_score >= 5000:
                        return 0.6
                    else:
                        return 0.4
            
            # Check for Cinebench scores
            if 'cinebench' in benchmark_data:
                scores = benchmark_data['cinebench'].get('scores', {})
                if 'multi_core' in scores:
                    cinebench = scores['multi_core']
                    # Cinebench: 1000-3000 range
                    if cinebench >= 2000:
                        return 1.0
                    elif cinebench >= 1500:
                        return 0.8
                    elif cinebench >= 1000:
                        return 0.6
                    else:
                        return 0.4
        
        # Semantic understanding
        if any(term in chunk_text for term in ['excellent cpu', 'fast processor', 'high performance']):
            return 0.9
        elif any(term in chunk_text for term in ['slow cpu', 'weak processor', 'poor performance']):
            return 0.3
        
        return 0.5
    
    def _extract_battery_score(self, benchmark_data: Dict, chunk_text: str) -> float:
        """Extract battery life score from benchmark data."""
        import re
        
        if benchmark_data:
            # Check for battery hours in structured data
            if 'battery' in benchmark_data:
                battery_info = benchmark_data['battery'].get('scores', {})
                hours = battery_info.get('hours') or battery_info.get('minutes', 0) / 60
                
                if hours >= 12:
                    return 1.0
                elif hours >= 10:
                    return 0.9
                elif hours >= 8:
                    return 0.7
                elif hours >= 6:
                    return 0.5
                else:
                    return 0.3
        
        # Parse from text (fallback)
        battery_matches = re.findall(r'(\d+(?:\.\d+)?)\s*hours?', chunk_text, re.IGNORECASE)
        if battery_matches:
            hours = max([float(h) for h in battery_matches])
            if hours >= 12:
                return 1.0
            elif hours >= 10:
                return 0.9
            elif hours >= 8:
                return 0.7
            elif hours >= 6:
                return 0.5
            else:
                return 0.3
        
        # Semantic understanding
        if any(term in chunk_text for term in ['excellent battery', 'long battery life', 'all-day battery']):
            return 0.9
        elif any(term in chunk_text for term in ['poor battery', 'short battery life', 'needs charging']):
            return 0.3
        
        return 0.5
    
    def _calculate_josh_score(self, product_data: Dict) -> float:
        """
        Calculate score based on Josh's recommendation.
        
        Factors:
        - Ranking (#1 = highest score)
        - Similarity score (how relevant the content is)
        - Number of mentions
        """
        score = 0.0
        
        # Ranking bonus (0.5 weight)
        if product_data.get('ranking'):
            ranking = product_data['ranking']
            # #1 = 1.0, #2 = 0.9, #3 = 0.8, etc.
            rank_score = max(0.0, 1.0 - (ranking - 1) * 0.1)
            score += rank_score * 0.5
        else:
            # No explicit ranking, but mentioned
            score += 0.3
        
        # Similarity bonus (0.3 weight)
        similarity = product_data.get('max_similarity', 0.0)
        score += similarity * 0.3
        
        # Multiple mentions bonus (0.2 weight)
        num_mentions = len(product_data.get('chunk_ids', []))
        mention_score = min(1.0, num_mentions / 3.0)  # Cap at 3 mentions
        score += mention_score * 0.2
        
        return min(1.0, score)
    
    def _calculate_manual_use_case_score(self, config_use_case_ids: List[int], quiz_response: Dict) -> float:
        """
        Calculate score boost if manual DB tags match user's quiz intent.
        
        Args:
            config_use_case_ids: List of manual use case IDs for this config
            quiz_response: User's quiz response
            
        Returns:
            1.0 if manual tag matches user intent, 0.0 otherwise.
        """
        if not config_use_case_ids:
            return 0.0
            
        user_intents = quiz_response.get('use_case', [])
        if not user_intents:
            return 0.0
            
        # Check for intersection between user's requested use cases and the config's manual tags
        for intent in user_intents:
            target_ids = self.manual_use_case_mapping.get(intent, [])
            for tid in target_ids:
                if tid in config_use_case_ids:
                    if self.verbose:
                        print(f"    [+] Manual Tag Match! Intent '{intent}' (ID {tid}) found in config tags.")
                    return 1.0
                    
        return 0.0

    def _calculate_spec_score(self, product_name: str, quiz_response: Dict, config_id: Optional[int] = None, config_obj: Optional[Dict] = None) -> float:
        """
        Calculate score based on real spec matching.
        
        Queries local config database for actual specs and matches against user requirements.
        Falls back to name-based heuristics if config not found.
        """
        # Try to get real config data
        config = config_obj
        if not config and config_id:
            # Fallback to individual fetch if not provided in batch
            config = self.config_client.get_config_by_id(config_id, include_properties=True)
        
        # Check if config exists and has either specs column OR property_groups
        if not config or (not config.get('specs') and not config.get('property_groups')):
            # Fallback to name-based heuristics
            return self._calculate_spec_score_heuristic(product_name, quiz_response)
        
        # Extract specs (handles both legacy and new schema)
        specs = self.config_client._extract_specs_from_config(config)
        
        score = 0.0
        checks = 0
        
        # Use case matching (based on real specs)
        use_cases = quiz_response.get('use_case', [])
        if use_cases:
            checks += 1
            use_case_score = 0.0
            
            for use_case in use_cases:
                if use_case == 'gaming':
                    # Check for dedicated GPU
                    if specs['has_gpu']:
                        use_case_score = 1.0
                        break
                
                elif use_case == 'programming':
                    # Check for good CPU and RAM
                    ram_gb = specs['ram_gb']
                    cpu = specs['processor'].lower()
                    
                    if ram_gb >= 16 and ('i7' in cpu or 'ryzen 7' in cpu or 'm3' in cpu or 'm2' in cpu or 'm1' in cpu):
                        use_case_score = max(use_case_score, 1.0)
                    elif ram_gb >= 8:
                        use_case_score = max(use_case_score, 0.7)
                
                elif use_case == 'video_editing':
                    # Check for GPU, RAM
                    if specs['has_gpu'] and specs['ram_gb'] >= 16:
                        use_case_score = max(use_case_score, 1.0)
                    elif specs['ram_gb'] >= 16 or specs['has_gpu']:
                        use_case_score = max(use_case_score, 0.7)
            
            score += use_case_score
        
        # Portability matching (based on weight and screen size)
        portability = quiz_response.get('portability', '')
        if portability:
            checks += 1
            weight_lbs = specs['weight_lbs']
            screen_inches = specs['screen_size']
            
            if portability == 'light':
                if weight_lbs > 0 and weight_lbs < 3.5 and screen_inches <= 14:
                    score += 1.0
                elif weight_lbs > 0 and weight_lbs < 4.0:
                    score += 0.7
            elif portability == 'performance':
                if screen_inches >= 15:
                    score += 1.0
                else:
                    score += 0.5
            elif portability == 'somewhat':
                if 14 <= screen_inches <= 16:
                    score += 1.0
                else:
                    score += 0.7
        
        # Budget matching (based on actual price)
        budgets = quiz_response.get('budget', [])
        if budgets and config.get('price'):
            checks += 1
            price = float(config['price'])
            
            budget_score = 0.0
            for budget in budgets:
                if budget == 'budget' and price < 800:
                    budget_score = 1.0
                    break
                elif budget == 'value' and 800 <= price < 1500:
                    budget_score = 1.0
                    break
                elif budget == 'premium' and price >= 1500:
                    budget_score = 1.0
                    break
            
            score += budget_score
        
        # Screen size matching
        screen_sizes = quiz_response.get('screen_size', [])
        if screen_sizes:
            checks += 1
            screen_size = specs.get('Display Size', '')
            screen_inches = self._extract_number(screen_size)
            
            screen_score = 0.0
            for size_pref in screen_sizes:
                if '13-14' in size_pref and 13 <= screen_inches <= 14:
                    screen_score = 1.0
                    break
                elif '15-16' in size_pref and 15 <= screen_inches <= 16:
                    screen_score = 1.0
                    break
                elif '17+' in size_pref and screen_inches >= 17:
                    screen_score = 1.0
                    break
            
            score += screen_score
        
        return score / checks if checks > 0 else 0.5
    
    def _calculate_spec_score_heuristic(self, product_name: str, quiz_response: Dict) -> float:
        """Fallback heuristic-based spec matching when config data not available."""
        score = 0.0
        checks = 0
        
        product_lower = product_name.lower()
        
        # Use case matching
        use_cases = quiz_response.get('use_case', [])
        if use_cases:
            checks += 1
            for use_case in use_cases:
                if use_case == 'programming' and any(x in product_lower for x in ['thinkpad', 'macbook', 'xps', 'framework']):
                    score += 1.0
                    break
                elif use_case == 'gaming' and any(x in product_lower for x in ['legion', 'rog', 'tuf', 'predator', 'nitro', 'omen', 'blade']):
                    score += 1.0
                    break
                elif use_case == 'video_editing' and any(x in product_lower for x in ['macbook pro', 'legion', 'rog', 'blade', 'precision']):
                    score += 1.0
                    break
        
        # Portability matching
        portability = quiz_response.get('portability', '')
        if portability:
            checks += 1
            if portability == 'light' and any(x in product_lower for x in ['air', '13', '14', 'gram', 'swift', 'zenbook']):
                score += 1.0
            elif portability == 'performance' and any(x in product_lower for x in ['16', '17', '18', 'legion', 'rog']):
                score += 1.0
            elif portability == 'somewhat' and '15' in product_lower:
                score += 1.0
        
        # Budget matching
        budgets = quiz_response.get('budget', [])
        if budgets:
            checks += 1
            for budget in budgets:
                if budget == 'budget' and any(x in product_lower for x in ['victus', 'nitro', 'ideapad', 'vivobook', 'thin']):
                    score += 1.0
                    break
                elif budget == 'premium' and any(x in product_lower for x in ['macbook pro', 'blade', 'spectre', 'xps', 'rog']):
                    score += 1.0
                    break
                elif budget == 'value' and any(x in product_lower for x in ['legion', 'omen', 'tuf', 'yoga', 'thinkpad']):
                    score += 1.0
                    break
        
        return score / checks if checks > 0 else 0.5
    
    def _extract_number(self, text: str) -> float:
        """Extract first number from text string."""
        if not text:
            return 0.0
        
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', str(text))
        return float(match.group(1)) if match else 0.0
    
    def _calculate_test_data_score_from_chunks(self, candidate: Dict, quiz_response: Dict) -> float:
        """
        Calculate score based on actual benchmark performance scores.
        Uses numeric benchmark values to rank products by real performance.
        
        Scoring approach:
        - Extract actual numeric scores from test data
        - Normalize scores relative to expected ranges
        - Weight by use case relevance
        
        Args:
            candidate: Candidate dictionary with test_data list containing benchmark_results
            quiz_response: User's quiz response
        
        Returns:
            Score from 0.0 to 1.0 based on actual performance
        """
        test_data = candidate.get('test_data', [])
        
        if not test_data:
            return 0.5  # Neutral score if no test data
        
        use_cases = quiz_response.get('use_case', [])
        portability = quiz_response.get('portability', '')
        
        # Collect benchmark scores from test data
        # test_data items have 'benchmark_name', 'benchmark_category', 'content'
        # We need to extract the actual benchmark_results JSON from the chunks
        
        # Since test_data is simplified in candidate, we'll work with what we have
        # The actual benchmark scores are in the chunk metadata
        # For now, use a simplified approach based on benchmark presence and context
        
        performance_score = 0.5  # Start neutral
        score_components = []
        
        # Extract actual benchmark scores from test data (already parsed and stored)
        cpu_score_value = 0
        gpu_score_value = 0
        battery_minutes = 0
        weight_grams = 0
        
        for test_item in test_data:
            benchmark_results = test_item.get('benchmark_results', {})
            
            # Extract CPU scores (Geekbench or Cinebench)
            if 'geekbench' in benchmark_results:
                geekbench_scores = benchmark_results['geekbench'].get('scores', {})
                if 'multi_core' in geekbench_scores and cpu_score_value == 0:
                    cpu_score_value = geekbench_scores['multi_core']
            
            if 'cinebench' in benchmark_results and cpu_score_value == 0:
                cinebench_scores = benchmark_results['cinebench'].get('scores', {})
                if 'multi_core' in cinebench_scores:
                    cpu_score_value = cinebench_scores['multi_core']
            
            # Extract GPU scores (3DMark)
            if '3dmark' in benchmark_results:
                dmark_scores = benchmark_results['3dmark'].get('scores', {})
                if 'timespy' in dmark_scores and gpu_score_value == 0:
                    gpu_score_value = dmark_scores['timespy']
                elif 'wildlife' in dmark_scores and gpu_score_value == 0:
                    gpu_score_value = dmark_scores['wildlife']
            
            # Extract battery life
            if 'battery' in benchmark_results and battery_minutes == 0:
                battery_scores = benchmark_results['battery'].get('scores', {})
                if 'minutes' in battery_scores:
                    battery_minutes = battery_scores['minutes']
                elif 'hours' in battery_scores:
                    battery_minutes = int(battery_scores['hours'] * 60)
            
            # Extract weight
            if 'weight' in benchmark_results and weight_grams == 0:
                weight_scores = benchmark_results['weight'].get('scores', {})
                if 'grams' in weight_scores:
                    weight_grams = weight_scores['grams']
        
        # Calculate performance score based on use case and actual benchmarks
        
        # CPU Performance (for programming, school, engineering, video_editing)
        if any(uc in use_cases for uc in ['programming', 'school', 'engineering', 'video_editing']):
            if cpu_score_value > 0:
                # Normalize Geekbench multi-core scores
                # Typical range: 500-1500 (good range for modern laptops)
                # Excellent: 1200+, Good: 900-1200, Average: 600-900, Below: <600
                if cpu_score_value >= 1200:
                    score_components.append(('cpu_excellent', 0.25))
                elif cpu_score_value >= 900:
                    score_components.append(('cpu_good', 0.20))
                elif cpu_score_value >= 600:
                    score_components.append(('cpu_average', 0.15))
                else:
                    score_components.append(('cpu_below', 0.10))
        
        # GPU Performance (for gaming, video_editing)
        if 'gaming' in use_cases or 'video_editing' in use_cases:
            if gpu_score_value > 0:
                # Normalize 3DMark Timespy scores
                # Typical range: 2000-8000 for gaming laptops
                # Excellent: 6000+, Good: 4000-6000, Average: 2500-4000, Below: <2500
                if gpu_score_value >= 6000:
                    score_components.append(('gpu_excellent', 0.30))
                elif gpu_score_value >= 4000:
                    score_components.append(('gpu_good', 0.25))
                elif gpu_score_value >= 2500:
                    score_components.append(('gpu_average', 0.18))
                else:
                    score_components.append(('gpu_below', 0.10))
        
        # Battery Life (for portability)
        if portability in ['light', 'somewhat']:
            if battery_minutes > 0:
                # Normalize battery minutes
                # Excellent: 600+ min (10h), Good: 420-600 (7-10h), Average: 300-420 (5-7h)
                if battery_minutes >= 600:
                    score_components.append(('battery_excellent', 0.20))
                elif battery_minutes >= 420:
                    score_components.append(('battery_good', 0.15))
                elif battery_minutes >= 300:
                    score_components.append(('battery_average', 0.10))
        
        # Weight (for portability)
        if portability == 'light':
            if weight_grams > 0:
                # Normalize weight in grams
                # Excellent: <1200g, Good: 1200-1500g, Average: 1500-1800g, Heavy: >1800g
                if weight_grams < 1200:
                    score_components.append(('weight_excellent', 0.15))
                elif weight_grams < 1500:
                    score_components.append(('weight_good', 0.10))
                elif weight_grams < 1800:
                    score_components.append(('weight_average', 0.05))
        
        # Sum up score components
        if score_components:
            total_bonus = sum(score for _, score in score_components)
            performance_score = min(1.0, 0.5 + total_bonus)
        
        return performance_score
    
    def _calculate_test_data_score(self, product_data: Dict, quiz_response: Dict) -> float:
        """
        Calculate score based on test data benchmarks.
        
        Args:
            product_data: Product data including chunks with test data
            quiz_response: User's quiz response
        
        Returns:
            Score from 0.0 to 1.0
        """
        # Check if this product has test data chunks
        chunks = product_data.get('chunks', [])
        test_data_chunks = [
            chunk for chunk in chunks 
            if chunk.metadata.get('source_type') == 'test_data'
        ]
        
        if not test_data_chunks:
            return 0.5  # Neutral score if no test data
        
        # Base score for having test data
        score = 0.6
        
        # Bonus for relevant benchmarks based on use case
        use_cases = quiz_response.get('use_case', [])
        benchmark_bonus = 0.0
        
        for chunk in test_data_chunks:
            benchmark_results = chunk.metadata.get('benchmark_results', {})
            
            # Check for relevant benchmarks
            if 'gaming' in use_cases or 'video_editing' in use_cases:
                # GPU/graphics performance matters
                if any(key in benchmark_results for key in ['3dmark', 'gpu', 'graphics']):
                    benchmark_bonus += 0.1
            
            if 'programming' in use_cases or 'video_editing' in use_cases:
                # CPU performance matters
                if any(key in benchmark_results for key in ['geekbench', 'cinebench', 'cpu']):
                    benchmark_bonus += 0.1
            
            if 'portability' in quiz_response:
                portability = quiz_response.get('portability', '')
                if portability in ['light', 'somewhat']:
                    # Battery and weight matter
                    if any(key in benchmark_results for key in ['battery', 'weight']):
                        benchmark_bonus += 0.1
        
        # Cap the bonus
        benchmark_bonus = min(benchmark_bonus, 0.4)
        
        return min(score + benchmark_bonus, 1.0)
    
    def _generate_explanation_from_candidate(
        self,
        product_name: str,
        candidate: Dict,
        quiz_response: Dict
    ) -> str:
        """
        Generate explanation from candidate data (optimized version).
        Uses candidate dictionary instead of product_data.
        """
        explanation_parts = []
        
        # Josh's ranking
        if candidate.get('rankings'):
            ranking = min(candidate['rankings'])
            if ranking == 1:
                explanation_parts.append("Josh's top pick")
            elif ranking <= 3:
                explanation_parts.append(f"Ranked #{ranking} by Josh")
        
        # Josh's quotes
        if candidate.get('quotes'):
            quote = candidate['quotes'][0]
            if len(quote) > 100:
                quote = quote[:97] + "..."
            explanation_parts.append(f'"{quote}"')
        
        # Pros
        if candidate.get('pros'):
            pros = candidate['pros']
            if pros:
                explanation_parts.append(pros[0])
        
        # Test data highlights
        test_data = candidate.get('test_data', [])
        if test_data:
            test_highlights = []
            use_cases = quiz_response.get('use_case', [])
            
            for test_item in test_data:
                benchmark_name = test_item.get('benchmark_name', '').lower()
                
                # Mention relevant benchmarks
                if ('gaming' in use_cases or 'video_editing' in use_cases) and '3dmark' in benchmark_name:
                    test_highlights.append("tested gaming performance")
                    break
                elif ('programming' in use_cases or 'video_editing' in use_cases) and 'geekbench' in benchmark_name:
                    test_highlights.append("verified CPU benchmarks")
                    break
                elif 'battery' in benchmark_name:
                    test_highlights.append("battery tested")
                    break
            
            if test_highlights:
                explanation_parts.append(test_highlights[0])
        
        return ". ".join(explanation_parts) if explanation_parts else f"Recommended based on your needs"
    
    def _generate_explanation(
        self,
        product_name: str,
        product_data: Dict,
        quiz_response: Dict
    ) -> str:
        """Generate human-readable explanation for recommendation."""
        explanation_parts = []
        
        # Get config data if available for specific details
        config_id = product_data.get('config_id')
        config = None
        if config_id:
            config = self.config_client.get_config_by_id(config_id)
        
        # Start with Josh's opinion or key pro
        if product_data.get('ranking'):
            ranking = product_data['ranking']
            if ranking == 1:
                explanation_parts.append("Josh's top pick")
            else:
                explanation_parts.append(f"Ranked #{ranking} by Josh")
        elif product_data.get('pros'):
            # Lead with the strongest pro if we have one
            pros = product_data.get('pros', [])
            if pros:
                explanation_parts.append(pros[0])
        
        # Add spec-based reasoning if we have config data
        if config and config.get('specs'):
            specs = config['specs']
            spec_highlights = []
            
            # GPU for gaming/video editing
            use_cases = quiz_response.get('use_case', [])
            if 'gaming' in use_cases or 'video_editing' in use_cases:
                gpu = specs.get('Dedicated Graphics (Yes/No)')
                if gpu == 'Yes':
                    spec_highlights.append("dedicated graphics")
            
            # RAM for power users
            if 'programming' in use_cases or 'video_editing' in use_cases:
                ram = specs.get('Memory Amount', '')
                ram_gb = self._extract_number(ram)
                if ram_gb >= 16:
                    spec_highlights.append(f"{int(ram_gb)}GB RAM")
                elif ram_gb >= 8:
                    spec_highlights.append(f"{int(ram_gb)}GB RAM")
            
            # Weight for portability
            portability = quiz_response.get('portability', '')
            if portability == 'light':
                weight = specs.get('Weight (lbs)', '')
                weight_lbs = self._extract_number(weight)
                if weight_lbs > 0 and weight_lbs < 4.0:
                    spec_highlights.append(f"{weight_lbs:.1f} lbs")
            
            # Screen size
            screen_size = specs.get('Display Size', '')
            if screen_size:
                screen_inches = self._extract_number(screen_size)
                if screen_inches > 0:
                    spec_highlights.append(f"{screen_inches:.1f}\" display")
            
            # Processor if notable
            cpu = specs.get('Processor', '')
            if cpu and any(notable in cpu.lower() for notable in ['ultra', 'm4', 'm3', 'ryzen 9', 'i9']):
                # Extract just the key part
                if 'm4' in cpu.lower():
                    spec_highlights.append("M4 chip")
                elif 'm3' in cpu.lower():
                    spec_highlights.append("M3 chip")
                elif 'ultra' in cpu.lower():
                    spec_highlights.append("Intel Ultra")
            
            if spec_highlights:
                explanation_parts.append(", ".join(spec_highlights))
        
        # Add test data highlights if available
        chunks = product_data.get('chunks', [])
        test_data_chunks = [
            chunk for chunk in chunks 
            if chunk.metadata.get('source_type') == 'test_data'
        ]
        
        if test_data_chunks:
            test_highlights = []
            use_cases = quiz_response.get('use_case', [])
            
            for chunk in test_data_chunks:
                benchmark_results = chunk.metadata.get('benchmark_results', {})
                
                # Mention relevant benchmarks
                if ('gaming' in use_cases or 'video_editing' in use_cases) and '3dmark' in benchmark_results:
                    test_highlights.append("tested gaming performance")
                    break
                elif ('programming' in use_cases or 'video_editing' in use_cases) and 'geekbench' in benchmark_results:
                    test_highlights.append("verified CPU benchmarks")
                    break
                elif 'battery' in benchmark_results:
                    test_highlights.append("battery tested")
                    break
            
            if test_highlights:
                explanation_parts.append(", ".join(test_highlights))
        
        # Add price context if available
        if config and config.get('price'):
            price = float(config['price'])
            budgets = quiz_response.get('budget', [])
            if budgets:
                budget_desc = self._get_price_description(price, budgets)
                if budget_desc:
                    explanation_parts.append(f"${int(price)}")
        
        # Fallback to generic explanation if we don't have enough specific details
        if len(explanation_parts) < 2:
            # Use case match
            use_cases = quiz_response.get('use_case', [])
            if use_cases:
                use_case_desc = {
                    'programming': 'coding',
                    'gaming': 'gaming',
                    'video_editing': 'content creation',
                    'general': 'everyday use'
                }
                formatted_uses = [use_case_desc.get(uc, uc) for uc in use_cases]
                explanation_parts.append(f"great for {', '.join(formatted_uses)}")
            
            # Portability
            portability = quiz_response.get('portability', '')
            if portability and len(explanation_parts) < 2:
                portability_map = {
                    'light': 'ultra-portable',
                    'somewhat': 'balanced',
                    'performance': 'powerful'
                }
                explanation_parts.append(portability_map.get(portability, portability))
        
        # Join with proper formatting
        if len(explanation_parts) == 0:
            return "Recommended by Josh"
        elif len(explanation_parts) == 1:
            return explanation_parts[0].capitalize() + "."
        else:
            # First part capitalized, rest lowercase, joined with periods or commas
            result = explanation_parts[0].capitalize()
            for i, part in enumerate(explanation_parts[1:], 1):
                if i == len(explanation_parts) - 1 and part.startswith('$'):
                    # Price at the end
                    result += f". {part}."
                else:
                    result += f". {part}"
            if not result.endswith('.'):
                result += "."
            return result
    def _get_price_description(self, price: float, budget_prefs: List[str]) -> Optional[str]:
        """Generate price description based on actual price and user budget preference."""
        if price < 800:
            price_tier = "budget"
        elif price < 1500:
            price_tier = "value"
        else:
            price_tier = "premium"
        
        # Only return price if it matches their budget preference
        if price_tier in budget_prefs:
            return f"${int(price)}"
        
        return None

    def _filter_by_brand_preferences(
        self,
        recommendations: List[ProductRecommendation],
        quiz_response: Dict
    ) -> List[ProductRecommendation]:
        """
        Filter recommendations based on brand preferences and exclusions.
        
        Args:
            recommendations: List of product recommendations
            quiz_response: User's quiz response with brand preferences
            
        Returns:
            Filtered list of recommendations
        """
        excluded_brands = quiz_response.get('excluded_brands', [])
        preferred_brands = quiz_response.get('preferred_brands', [])
        
        # If no brand preferences, return all
        if not excluded_brands and not preferred_brands:
            return recommendations
        
        filtered = []
        
        for rec in recommendations:
            product_name = rec.product_name.lower()
            
            # Check if product is from an excluded brand
            is_excluded = False
            for brand in excluded_brands:
                brand_lower = brand.lower()
                # Check if brand name appears in product name
                if brand_lower in product_name:
                    is_excluded = True
                    if self.verbose:
                        print(f"[*] Filtering out {rec.product_name} (excluded brand: {brand})")
                    break
            
            if not is_excluded:
                filtered.append(rec)
        
        # If there are preferred brands, boost their scores
        if preferred_brands:
            for rec in filtered:
                product_name = rec.product_name.lower()
                for brand in preferred_brands:
                    if brand.lower() in product_name:
                        # Boost confidence score by 10%
                        rec.confidence_score = min(1.0, rec.confidence_score * 1.1)
                        break
            
            # Re-sort after boosting
            filtered.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return filtered


if __name__ == "__main__":
    """Test the ranker."""
    from src.rag.retriever import RAGRetriever
    
    retriever = RAGRetriever(top_k=10)
    ranker = RAGRanker(josh_weight=0.7, spec_weight=0.3)
    
    quiz_response = {
        'profession': ['Student', 'Developer'],
        'use_case': ['programming'],
        'budget': ['value'],
        'portability': 'light',
        'screen_size': ['13-14 inches']
    }
    
    retrieval_results = retriever.retrieve(quiz_response=quiz_response, top_k=10)
    recommendations = ranker.rank(retrieval_results, quiz_response, top_k=5)
    
    print(f"TOP {len(recommendations)} RECOMMENDATIONS:")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.product_name}")
        print(f"   Confidence: {rec.confidence_score:.3f}")
        if rec.ranking:
            print(f"   Josh's Ranking: #{rec.ranking}")
        print(f"   Article: {rec.source_article}")
