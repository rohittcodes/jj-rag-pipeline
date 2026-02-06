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
        self.use_ec2_db = use_ec2_db
        self.ec2_db_config = ec2_db_config
        self.verbose = verbose
        
        # Initialize config database client (uses local synced configs)
        self.config_client = ConfigDatabaseClient()
        
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
        
        # Step 1: Extract products from chunks
        products = self._extract_products(retrieval_results)
        if self.verbose:
            print(f"[+] Extracted {len(products)} unique products")
        
        # Step 2: Score each product
        recommendations = []
        for key, product_data in products.items():
            product_name = product_data['product_name']
            
            # Calculate Josh score
            josh_score = self._calculate_josh_score(product_data)
            
            # Calculate spec score (with real config data)
            config_id = product_data.get('config_id')
            spec_score = self._calculate_spec_score(product_name, quiz_response, config_id)
            
            # Calculate test data score
            test_data_score = self._calculate_test_data_score(product_data, quiz_response)
            
            # Combined score
            confidence_score = (josh_score * self.josh_weight) + (spec_score * self.spec_weight) + (test_data_score * self.test_data_weight)
            
            # Generate explanation
            explanation = self._generate_explanation(
                product_name, product_data, quiz_response
            )
            
            # Create recommendation
            recommendation = ProductRecommendation(
                product_name=product_name,
                config_id=product_data.get('config_id'),
                confidence_score=confidence_score,
                josh_score=josh_score,
                spec_score=spec_score,
                test_data_score=test_data_score,
                ranking=product_data.get('ranking'),
                recommendation_type=product_data.get('recommendation_type', 'mentioned'),
                josh_quote=product_data.get('josh_quote'),
                source_article=product_data['source_article'],
                source_url=product_data['source_url'],
                section_title=product_data.get('section_title'),
                explanation=explanation,
                pros=product_data.get('pros', []),
                cons=product_data.get('cons', []),
                who_is_this_for=product_data.get('who_is_this_for'),
                similarity=product_data['max_similarity'],
                chunk_ids=product_data['chunk_ids']
            )
            
            recommendations.append(recommendation)
        
        # Step 3: Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        if self.verbose:
            print(f"[+] Ranked {len(recommendations)} products")
            if recommendations:
                print(f"    - Top recommendation: {recommendations[0].product_name}")
                print(f"    - Confidence: {recommendations[0].confidence_score:.3f}")
        
        return recommendations[:top_k]
    
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
        processed_products = {}
        for key, data in configs.items():
            config_id = data['config_id']
            
            # Get product name from database
            config = self.config_client.get_config_by_id(config_id)
            product_name = config['product_name'] if config else f"Config {config_id}"
            
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
    
    def _calculate_spec_score(self, product_name: str, quiz_response: Dict, config_id: Optional[int] = None) -> float:
        """
        Calculate score based on real spec matching.
        
        Queries local config database for actual specs and matches against user requirements.
        Falls back to name-based heuristics if config not found.
        """
        # Try to get real config data
        config = None
        if config_id:
            config = self.config_client.get_config_by_id(config_id)
        
        if not config or not config.get('specs'):
            # Fallback to name-based heuristics
            return self._calculate_spec_score_heuristic(product_name, quiz_response)
        
        specs = config['specs']
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
                    has_gpu = specs.get('Dedicated Graphics (Yes/No)') == 'Yes'
                    if has_gpu:
                        use_case_score = 1.0
                        break
                
                elif use_case == 'programming':
                    # Check for good CPU and RAM
                    ram = specs.get('Memory Amount', '')
                    cpu = specs.get('Processor', '')
                    
                    ram_gb = self._extract_number(ram)
                    if ram_gb >= 16 and ('i7' in cpu.lower() or 'ryzen 7' in cpu.lower() or 'm3' in cpu.lower()):
                        use_case_score = 1.0
                        break
                    elif ram_gb >= 8:
                        use_case_score = 0.7
                
                elif use_case == 'video_editing':
                    # Check for GPU, RAM, and good CPU
                    has_gpu = specs.get('Dedicated Graphics (Yes/No)') == 'Yes'
                    ram = specs.get('Memory Amount', '')
                    ram_gb = self._extract_number(ram)
                    
                    if has_gpu and ram_gb >= 16:
                        use_case_score = 1.0
                        break
                    elif ram_gb >= 16:
                        use_case_score = 0.7
            
            score += use_case_score
        
        # Portability matching (based on weight and screen size)
        portability = quiz_response.get('portability', '')
        if portability:
            checks += 1
            weight_str = specs.get('Weight (lbs)', '')
            screen_size = specs.get('Display Size', '')
            
            weight_lbs = self._extract_number(weight_str)
            screen_inches = self._extract_number(screen_size)
            
            if portability == 'light':
                if weight_lbs > 0 and weight_lbs < 3.5 and screen_inches <= 14:
                    score += 1.0
                elif weight_lbs < 4.0:
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
