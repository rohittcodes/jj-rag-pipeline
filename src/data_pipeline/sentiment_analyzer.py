"""
Sentiment Analyzer - Analyze sentiment of product mentions in content chunks.

Uses Gemini Flash to determine if a product mention is positive, negative, or neutral.
"""
import os
import json
import time
from typing import Dict, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class SentimentAnalyzer:
    """
    Analyze sentiment of product mentions in text chunks.
    
    Uses Gemini Flash to classify sentiment and context of product mentions.
    """
    
    def __init__(self, verbose: bool = False, rate_limit_delay: float = 0.2):
        """
        Initialize sentiment analyzer with Gemini client.
        
        Args:
            verbose: Print detailed output
            rate_limit_delay: Delay in seconds between API calls to avoid rate limits
        """
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.0-flash'
        self.verbose = verbose
        self.rate_limit_delay = rate_limit_delay
        self.last_call_time = 0
        
        if verbose:
            print(f"[+] SentimentAnalyzer initialized with {self.model_name}")
    
    def analyze_chunk(self, chunk_text: str, product_name: str) -> Dict:
        """
        Analyze sentiment toward a specific product in a text chunk.
        
        Args:
            chunk_text: The text content to analyze
            product_name: The product being mentioned
        
        Returns:
            Dictionary with:
            - sentiment_score: float (-1.0 to +1.0)
            - sentiment_label: str (highly_positive, positive, neutral, negative, highly_negative)
            - context_type: str (recommendation, comparison, criticism, example, alternative)
            - reasoning: str (brief explanation)
        """
        prompt = self._create_sentiment_prompt(chunk_text, product_name)
        
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_call)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            self.last_call_time = time.time()
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])
            
            sentiment_data = json.loads(response_text)
            
            # Validate and normalize
            sentiment_score = float(sentiment_data.get('sentiment_score', 0.0))
            sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1, 1]
            
            sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
            if sentiment_label not in ['highly_positive', 'positive', 'neutral', 'negative', 'highly_negative']:
                sentiment_label = 'neutral'
            
            context_type = sentiment_data.get('context_type', 'mention')
            if context_type not in ['recommendation', 'comparison', 'criticism', 'example', 'alternative', 'mention']:
                context_type = 'mention'
            
            result = {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'context_type': context_type,
                'reasoning': sentiment_data.get('reasoning', 'No reasoning provided')
            }
            
            if self.verbose:
                print(f"[+] Sentiment for '{product_name}': {sentiment_label} ({sentiment_score:.2f}) - {context_type}")
            
            return result
            
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[!] Failed to parse sentiment response: {e}")
            return self._get_neutral_sentiment()
        except Exception as e:
            if self.verbose:
                print(f"[!] Sentiment analysis error: {e}")
            return self._get_neutral_sentiment()
    
    def analyze_batch(self, chunks: list[tuple[str, str]]) -> list[Dict]:
        """
        Analyze sentiment for multiple chunks in a single API call.
        
        Args:
            chunks: List of (chunk_text, product_name) tuples
        
        Returns:
            List of sentiment dictionaries (same order as input)
        """
        if not chunks:
            return []
        
        # For now, process individually (can optimize later with batch prompts)
        results = []
        for chunk_text, product_name in chunks:
            result = self.analyze_chunk(chunk_text, product_name)
            results.append(result)
        
        return results
    
    def _create_sentiment_prompt(self, chunk_text: str, product_name: str) -> str:
        """Create prompt for sentiment analysis."""
        return f"""Analyze the sentiment toward "{product_name}" in this text excerpt from a laptop review/article.

Text:
{chunk_text}

Return ONLY a JSON object (no markdown, no explanation) with this exact structure:
{{
  "sentiment_score": <number from -1.0 to +1.0>,
  "sentiment_label": "<highly_positive|positive|neutral|negative|highly_negative>",
  "context_type": "<recommendation|comparison|criticism|example|alternative|mention>",
  "reasoning": "<brief 1-sentence explanation>"
}}

Guidelines:
- sentiment_score: -1.0 (very negative) to +1.0 (very positive)
- sentiment_label:
  * highly_positive: Strong recommendation, top pick, "best for X", "#1 choice"
  * positive: Recommended, good option, praised features
  * neutral: Factual mention, no clear opinion, just stating specs
  * negative: Not recommended, has issues, "avoid for X", criticized
  * highly_negative: Strongly discouraged, major flaws, "worst choice"
- context_type:
  * recommendation: Direct product recommendation or endorsement
  * comparison: Comparing products (may be positive or negative)
  * criticism: Pointing out flaws, issues, or limitations
  * example: Used as example (usually neutral)
  * alternative: Mentioned as alternative option
  * mention: Generic mention without strong context

Examples:
- "The MacBook Pro 14 is the best laptop for developers" → highly_positive, recommendation
- "I don't recommend the XPS 15 for gaming" → negative, criticism
- "The ThinkPad is mentioned here as an example" → neutral, example
- "For comparison, the Legion 5 has better GPU" → neutral/positive, comparison"""
    
    def _get_neutral_sentiment(self) -> Dict:
        """Return neutral sentiment as fallback."""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'context_type': 'mention',
            'reasoning': 'Unable to determine sentiment'
        }
