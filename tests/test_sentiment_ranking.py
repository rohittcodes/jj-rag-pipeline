"""
Test sentiment-aware ranking functionality.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.sentiment_analyzer import SentimentAnalyzer


def test_sentiment_analyzer():
    """Test basic sentiment analysis."""
    analyzer = SentimentAnalyzer(verbose=True)
    
    # Test positive mention
    positive_text = """
    The MacBook Pro 14 is the best laptop for developers. It has incredible 
    performance, amazing battery life, and a beautiful display. Highly recommended!
    """
    
    result = analyzer.analyze_chunk(positive_text, "MacBook Pro 14")
    print(f"\nPositive Test:")
    print(f"  Score: {result['sentiment_score']}")
    print(f"  Label: {result['sentiment_label']}")
    print(f"  Context: {result['context_type']}")
    print(f"  Reasoning: {result['reasoning']}")
    
    assert result['sentiment_label'] in ['positive', 'highly_positive'], "Should be positive"
    assert result['sentiment_score'] > 0.3, "Score should be positive"
    
    # Test negative mention
    negative_text = """
    I don't recommend the XPS 15 for gaming. The GPU is underpowered and 
    it thermal throttles constantly. Not worth the money for gamers.
    """
    
    result = analyzer.analyze_chunk(negative_text, "XPS 15")
    print(f"\nNegative Test:")
    print(f"  Score: {result['sentiment_score']}")
    print(f"  Label: {result['sentiment_label']}")
    print(f"  Context: {result['context_type']}")
    print(f"  Reasoning: {result['reasoning']}")
    
    assert result['sentiment_label'] in ['negative', 'highly_negative'], "Should be negative"
    assert result['sentiment_score'] < 0.0, "Score should be negative"
    
    # Test neutral mention
    neutral_text = """
    The ThinkPad X1 Carbon is mentioned here as an example of a business laptop.
    It has standard specs including 16GB RAM and 512GB SSD.
    """
    
    result = analyzer.analyze_chunk(neutral_text, "ThinkPad X1 Carbon")
    print(f"\nNeutral Test:")
    print(f"  Score: {result['sentiment_score']}")
    print(f"  Label: {result['sentiment_label']}")
    print(f"  Context: {result['context_type']}")
    print(f"  Reasoning: {result['reasoning']}")
    
    assert result['sentiment_label'] == 'neutral', "Should be neutral"
    assert -0.3 < result['sentiment_score'] < 0.3, "Score should be near zero"
    
    print("\n[+] All sentiment tests passed!")


def test_sentiment_multiplier():
    """Test sentiment multiplier calculation."""
    from src.rag.ranker import RAGRanker
    
    ranker = RAGRanker(verbose=False)
    
    # Test highly positive
    candidate = {
        'sentiments': [
            {'score': 0.9, 'label': 'highly_positive', 'context': 'recommendation'}
        ]
    }
    multiplier = ranker._calculate_sentiment_multiplier(candidate)
    print(f"\nHighly Positive Multiplier: {multiplier}")
    assert multiplier == 1.2, "Should boost by 20%"
    
    # Test negative
    candidate = {
        'sentiments': [
            {'score': -0.7, 'label': 'negative', 'context': 'criticism'}
        ]
    }
    multiplier = ranker._calculate_sentiment_multiplier(candidate)
    print(f"Negative Multiplier: {multiplier}")
    assert multiplier == 0.3, "Should reduce by 70%"
    
    # Test all negative (should be heavily penalized)
    candidate = {
        'sentiments': [
            {'score': -0.8, 'label': 'highly_negative', 'context': 'criticism'},
            {'score': -0.6, 'label': 'negative', 'context': 'criticism'}
        ]
    }
    multiplier = ranker._calculate_sentiment_multiplier(candidate)
    print(f"All Negative Multiplier: {multiplier}")
    assert multiplier == 0.1, "Should reduce by 90% when all negative"
    
    # Test mixed sentiment
    candidate = {
        'sentiments': [
            {'score': 0.8, 'label': 'positive', 'context': 'recommendation'},
            {'score': -0.4, 'label': 'negative', 'context': 'criticism'}
        ]
    }
    multiplier = ranker._calculate_sentiment_multiplier(candidate)
    print(f"Mixed Sentiment Multiplier: {multiplier}")
    assert 0.5 <= multiplier <= 1.1, "Should be moderate for mixed sentiment"
    
    print("\n[+] All multiplier tests passed!")


if __name__ == "__main__":
    print("="*80)
    print("SENTIMENT RANKING TESTS")
    print("="*80)
    
    try:
        test_sentiment_analyzer()
        test_sentiment_multiplier()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n[!] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
