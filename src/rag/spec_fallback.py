"""
Spec-based fallback recommender for low-confidence RAG scenarios.

When RAG confidence is below threshold, this provides recommendations
based purely on spec matching against user requirements.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SpecRecommendation:
    """Spec-based recommendation (no Josh content)."""
    config_id: int
    product_name: str
    confidence_score: float
    spec_score: float
    explanation: str
    price: Optional[float] = None
    brand: Optional[str] = None


class SpecFallbackRecommender:
    """
    Spec-only recommender for when RAG confidence is low.
    
    Matches configs purely on budget, use case, portability, screen size.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'josh_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
    
    def recommend(
        self,
        quiz_response: Dict,
        top_k: int = 5
    ) -> List[SpecRecommendation]:
        """
        Generate spec-only recommendations.
        
        Args:
            quiz_response: User's quiz response
            top_k: Number of recommendations to return
        
        Returns:
            List of SpecRecommendation objects
        """
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Fetch all active configs with specs
        cursor.execute("""
            SELECT
                config_id,
                product_name,
                brand,
                specs,
                price,
                final_rating
            FROM configs
            WHERE specs IS NOT NULL
            ORDER BY config_id;
        """)
        
        configs = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not configs:
            return []
        
        # Score each config
        scored_configs = []
        for config in configs:
            score = self._calculate_spec_score(config, quiz_response)
            if score > 0:
                explanation = self._generate_explanation(config, quiz_response, score)
                scored_configs.append(SpecRecommendation(
                    config_id=config['config_id'],
                    product_name=config['product_name'],
                    confidence_score=score,
                    spec_score=score,
                    explanation=explanation,
                    price=float(config['price']) if config.get('price') else None,
                    brand=config.get('brand')
                ))
        
        # Sort by score
        scored_configs.sort(key=lambda x: x.confidence_score, reverse=True)
        
        if self.verbose:
            print(f"[Spec Fallback] Scored {len(scored_configs)} configs, returning top {top_k}")
        
        return scored_configs[:top_k]
    
    def _calculate_spec_score(self, config: Dict, quiz_response: Dict) -> float:
        """Calculate spec match score (0-1)."""
        score = 0.0
        checks = 0
        
        specs = config.get('specs') or {}
        price = config.get('price')
        
        # Budget match
        budgets = quiz_response.get('budget', [])
        if budgets and price:
            checks += 1
            try:
                price_val = float(price)
                if price_val < 800 and 'budget' in budgets:
                    score += 1.0
                elif 800 <= price_val < 1500 and 'value' in budgets:
                    score += 1.0
                elif price_val >= 1500 and 'premium' in budgets:
                    score += 1.0
                else:
                    score += 0.3
            except (TypeError, ValueError):
                pass
        
        # Use case match
        use_cases = quiz_response.get('use_case', [])
        if use_cases:
            checks += 1
            has_gpu = (specs.get('Dedicated Graphics (Yes/No)') or '').lower() == 'yes'
            ram_str = specs.get('Memory Amount', '') or ''
            ram_gb = self._extract_number(ram_str)
            cpu = (specs.get('Processor') or '').lower()
            
            uc_score = 0.0
            for uc in use_cases:
                if uc == 'gaming' and has_gpu:
                    uc_score = max(uc_score, 1.0)
                elif uc == 'programming' and (ram_gb >= 8 or 'm3' in cpu or 'i7' in cpu):
                    uc_score = max(uc_score, 1.0 if ram_gb >= 16 else 0.7)
                elif uc == 'video_editing' and (has_gpu or ram_gb >= 16):
                    uc_score = max(uc_score, 1.0 if (has_gpu and ram_gb >= 16) else 0.7)
                elif uc == 'general':
                    uc_score = max(uc_score, 0.5)
            
            score += uc_score if uc_score > 0 else 0.3
        
        # Portability match
        portability = quiz_response.get('portability', '')
        if portability:
            checks += 1
            weight_str = specs.get('Weight (lbs)', '') or ''
            screen_str = specs.get('Display Size', '') or ''
            weight = self._extract_number(weight_str)
            screen = self._extract_number(screen_str)
            
            if portability == 'light' and weight > 0 and weight < 4.0:
                score += 1.0
            elif portability == 'performance' and screen >= 15:
                score += 1.0
            elif portability == 'somewhat' and 14 <= screen <= 16:
                score += 1.0
            else:
                score += 0.4
        
        # Screen size match
        screen_sizes = quiz_response.get('screen_size', [])
        if screen_sizes:
            checks += 1
            screen_str = specs.get('Display Size', '') or ''
            screen_inches = self._extract_number(screen_str)
            
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
            
            score += screen_score if screen_score > 0 else 0.3
        
        # Normalize
        return score / checks if checks > 0 else 0.0
    
    @staticmethod
    def _extract_number(text: str) -> float:
        """Extract first number from text."""
        if not text:
            return 0.0
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', str(text))
        return float(match.group(1)) if match else 0.0
    
    def _generate_explanation(self, config: Dict, quiz_response: Dict, score: float) -> str:
        """Generate explanation for spec-based recommendation."""
        parts = []
        
        specs = config.get('specs') or {}
        price = config.get('price')
        
        # Budget
        budgets = quiz_response.get('budget', [])
        if budgets and price:
            try:
                price_val = float(price)
                if price_val < 800:
                    parts.append(f"Budget-friendly at ${int(price_val)}")
                elif price_val < 1500:
                    parts.append(f"Good value at ${int(price_val)}")
                else:
                    parts.append(f"Premium option at ${int(price_val)}")
            except (TypeError, ValueError):
                pass
        
        # Use case
        use_cases = quiz_response.get('use_case', [])
        if use_cases:
            has_gpu = (specs.get('Dedicated Graphics (Yes/No)') or '').lower() == 'yes'
            ram_str = specs.get('Memory Amount', '') or ''
            ram_gb = self._extract_number(ram_str)
            
            if 'gaming' in use_cases and has_gpu:
                parts.append("dedicated graphics for gaming")
            if ('programming' in use_cases or 'video_editing' in use_cases) and ram_gb >= 16:
                parts.append(f"{int(ram_gb)}GB RAM")
        
        # Portability
        portability = quiz_response.get('portability', '')
        if portability:
            weight_str = specs.get('Weight (lbs)', '') or ''
            weight = self._extract_number(weight_str)
            if portability == 'light' and weight > 0 and weight < 4.0:
                parts.append(f"lightweight at {weight:.1f} lbs")
        
        if not parts:
            parts.append(f"Matches your requirements (score: {score:.2f})")
        
        return ". ".join(parts).capitalize() + "."


if __name__ == "__main__":
    # Test spec fallback
    fallback = SpecFallbackRecommender(verbose=True)
    
    quiz = {
        'profession': ['Student'],
        'use_case': ['programming'],
        'budget': ['value'],
        'portability': 'light'
    }
    
    recommendations = fallback.recommend(quiz, top_k=5)
    
    print(f"\nSpec-based recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.product_name}")
        print(f"   Score: {rec.confidence_score:.3f}")
        print(f"   Price: ${rec.price:.0f}" if rec.price else "   Price: N/A")
        print(f"   {rec.explanation}")
