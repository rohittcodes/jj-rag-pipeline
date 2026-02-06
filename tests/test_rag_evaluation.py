"""
Comprehensive RAG Pipeline Evaluation

This script validates the RAG pipeline end-to-end with multiple test cases
and evaluation metrics to ensure quality recommendations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.retriever import RAGRetriever
from src.rag.ranker import RAGRanker
from typing import List, Dict, Tuple
import json


class RAGEvaluator:
    """Evaluate RAG pipeline quality with multiple test scenarios."""
    
    def __init__(self):
        self.retriever = RAGRetriever(top_k=15, verbose=False)
        self.ranker = RAGRanker(josh_weight=0.7, spec_weight=0.3, verbose=False)
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> List[Dict]:
        """Load diverse test cases covering different user personas."""
        return [
            {
                "name": "Budget Student (Programming)",
                "quiz": {
                    "profession": ["Student"],
                    "use_case": ["programming"],
                    "budget": ["budget"],
                    "portability": "light",
                    "screen_size": ["13-14 inches"]
                },
                "expectations": {
                    "requires_results": True,
                    "requires_config_ids": True
                }
            },
            {
                "name": "Professional Video Editor (Premium)",
                "quiz": {
                    "profession": ["Content Creator"],
                    "use_case": ["video_editing"],
                    "budget": ["premium"],
                    "portability": "somewhat",
                    "screen_size": ["15-16 inches"]
                },
                "expectations": {
                    "requires_results": True,
                    "requires_config_ids": True,
                    "requires_gpu": True
                }
            },
            {
                "name": "Hardcore Gamer (Performance)",
                "quiz": {
                    "profession": ["Gamer"],
                    "use_case": ["gaming"],
                    "budget": ["value", "premium"],
                    "portability": "performance",
                    "screen_size": ["15-16 inches", "17+ inches"]
                },
                "expectations": {
                    "requires_results": True,
                    "requires_config_ids": True,
                    "requires_gpu": True
                }
            },
            {
                "name": "Business Professional (Portability)",
                "quiz": {
                    "profession": ["Business Professional"],
                    "use_case": ["general"],
                    "budget": ["value", "premium"],
                    "portability": "light",
                    "screen_size": ["13-14 inches"]
                },
                "expectations": {
                    "requires_results": True,
                    "requires_config_ids": True
                }
            },
            {
                "name": "Data Science Student (Balanced)",
                "quiz": {
                    "profession": ["Student"],
                    "use_case": ["programming", "video_editing"],
                    "budget": ["value"],
                    "portability": "somewhat",
                    "screen_size": ["15-16 inches"]
                },
                "expectations": {
                    "requires_results": True,
                    "requires_config_ids": True
                }
            },
            {
                "name": "Intel Panther Lake User (YouTube Content Test)",
                "quiz": {
                    "profession": ["General"],
                    "use_case": ["general"],
                    "budget": ["value"],
                    "portability": "somewhat",
                    "screen_size": ["14 inches"]
                },
                "expectations": {
                    "requires_results": True,
                    "requires_config_ids": True,
                    "expects_youtube": True  # This test should retrieve YouTube content
                }
            }
        ]
    
    def evaluate_all(self) -> Dict:
        """Run all test cases and return comprehensive results."""
        print("=" * 80)
        print("RAG PIPELINE EVALUATION")
        print("=" * 80)
        print()
        
        results = {
            "total_tests": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": [],
            "total_blog_chunks": 0,
            "total_youtube_chunks": 0
        }
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"Test {i}/{len(self.test_cases)}: {test_case['name']}")
            print("-" * 80)
            
            test_result = self._evaluate_test_case(test_case)
            results["test_results"].append(test_result)
            
            if "source_stats" in test_result:
                results["total_blog_chunks"] += test_result["source_stats"]["blog"]
                results["total_youtube_chunks"] += test_result["source_stats"]["youtube"]
            
            if test_result["passed"]:
                results["passed"] += 1
                print(f"[+] PASSED")
            else:
                results["failed"] += 1
                print(f"[-] FAILED")
            
            print()
        
        # Summary
        print("=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']} ({results['passed']/results['total_tests']*100:.1f}%)")
        print(f"Failed: {results['failed']} ({results['failed']/results['total_tests']*100:.1f}%)")
        print()
        
        # Source distribution summary
        total_chunks = results["total_blog_chunks"] + results["total_youtube_chunks"]
        if total_chunks > 0:
            print("SOURCE DISTRIBUTION:")
            print(f"  Blog chunks: {results['total_blog_chunks']} ({results['total_blog_chunks']/total_chunks*100:.1f}%)")
            print(f"  YouTube chunks: {results['total_youtube_chunks']} ({results['total_youtube_chunks']/total_chunks*100:.1f}%)")
            if results["total_youtube_chunks"] == 0:
                print("  [!] WARNING: No YouTube content retrieved across all tests")
            print()
        
        # Detailed failures
        if results["failed"] > 0:
            print("FAILED TESTS:")
            for result in results["test_results"]:
                if not result["passed"]:
                    print(f"\n{result['test_name']}:")
                    for issue in result["issues"]:
                        print(f"  - {issue}")
        
        return results
    
    def _evaluate_test_case(self, test_case: Dict) -> Dict:
        """Evaluate a single test case."""
        name = test_case["name"]
        quiz = test_case["quiz"]
        expectations = test_case["expectations"]
        
        # Run RAG pipeline
        try:
            # Use custom query for YouTube content test
            if "Intel Panther Lake" in name:
                retrieval_results = self.retriever.retrieve(query="Intel Panther Lake ThinkPad Gen 1 integrated GPU", top_k=15)
            else:
                retrieval_results = self.retriever.retrieve(quiz_response=quiz, top_k=15)
            recommendations = self.ranker.rank(retrieval_results, quiz, top_k=5)
        except Exception as e:
            return {
                "test_name": name,
                "passed": False,
                "issues": [f"Pipeline error: {str(e)}"],
                "recommendations": []
            }
        
        blog_count = sum(1 for r in retrieval_results if r.metadata.get('source_type') == 'blog')
        youtube_count = sum(1 for r in retrieval_results if r.metadata.get('source_type') == 'youtube')
        print(f"  Retrieved {len(retrieval_results)} chunks: {blog_count} blog, {youtube_count} YouTube")
        
        # Evaluate recommendations
        issues = []
        
        # Check 1: Got recommendations
        if not recommendations:
            issues.append("No recommendations returned")
            return {
                "test_name": name,
                "passed": False,
                "issues": issues,
                "recommendations": [],
                "source_stats": {"blog": blog_count, "youtube": youtube_count}
            }
        
        print(f"  Got {len(recommendations)} recommendations")
        
        # Check 2: Config IDs populated
        if expectations.get("requires_config_ids"):
            missing_configs = sum(1 for rec in recommendations if not rec.config_id)
            if missing_configs > 0:
                issues.append(f"{missing_configs}/{len(recommendations)} recommendations missing config_id")
        
        # Check 3: YouTube content expected
        if expectations.get("expects_youtube"):
            if youtube_count == 0:
                issues.append("Expected YouTube content but none retrieved")
        
        # Check 4: GPU requirement
        if expectations.get("requires_gpu"):
            gpu_found = False
            for rec in recommendations[:3]:
                if rec.config_id:
                    config = self.ranker.config_client.get_config_by_id(rec.config_id)
                    if config and config.get('specs'):
                        gpu = config['specs'].get('Dedicated Graphics (Yes/No)')
                        if gpu == 'Yes':
                            gpu_found = True
                            break
            
            if not gpu_found:
                issues.append("GPU required but not found in top 3 recommendations")
        
        # Check 6: All recommendations have explanations
        for rec in recommendations:
            if not rec.explanation or len(rec.explanation) < 20:
                issues.append(f"{rec.product_name}: Missing or too short explanation")
                break
        
        # Check 7: Confidence scores are reasonable
        for rec in recommendations:
            if rec.confidence_score < 0 or rec.confidence_score > 1:
                issues.append(f"{rec.product_name}: Invalid confidence score {rec.confidence_score}")
                break
        
        # Check 8: Top recommendation should have highest confidence
        if len(recommendations) > 1:
            if recommendations[0].confidence_score < recommendations[1].confidence_score:
                issues.append("Top recommendation doesn't have highest confidence score")
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec.product_name} (confidence: {rec.confidence_score:.3f})")
            if rec.config_id:
                config = self.ranker.config_client.get_config_by_id(rec.config_id)
                if config:
                    specs = []
                    if config.get('price'):
                        specs.append(f"${int(float(config['price']))}")
                    if config.get('specs'):
                        ram = config['specs'].get('Memory Amount', '')
                        if ram:
                            specs.append(ram)
                        gpu = config['specs'].get('Dedicated Graphics (Yes/No)')
                        if gpu == 'Yes':
                            specs.append("GPU")
                    if specs:
                        print(f"     {', '.join(specs)}")
        
        return {
            "test_name": name,
            "passed": len(issues) == 0,
            "issues": issues,
            "recommendations": [
                {
                    "name": rec.product_name,
                    "confidence": rec.confidence_score,
                    "config_id": rec.config_id
                }
                for rec in recommendations
            ],
            "source_stats": {"blog": blog_count, "youtube": youtube_count}
        }
    
    def test_retrieval_quality(self):
        """Test retrieval component specifically."""
        print("=" * 80)
        print("RETRIEVAL QUALITY TEST")
        print("=" * 80)
        print()
        
        test_queries = [
            {
                "name": "Gaming laptop query",
                "quiz": {"use_case": ["gaming"], "budget": ["value"]},
                "expected_keywords": ["gaming", "gpu", "graphics", "performance"]
            },
            {
                "name": "Student laptop query",
                "quiz": {"profession": ["Student"], "use_case": ["programming"], "budget": ["budget"]},
                "expected_keywords": ["student", "budget", "programming", "portable"]
            },
            {
                "name": "Video editing query",
                "quiz": {"use_case": ["video_editing"], "budget": ["premium"]},
                "expected_keywords": ["video", "editing", "creator", "performance"]
            }
        ]
        
        for test in test_queries:
            print(f"Test: {test['name']}")
            results = self.retriever.retrieve(quiz_response=test["quiz"], top_k=10)
            
            print(f"  Retrieved {len(results)} chunks")
            
            # Check if expected keywords appear in retrieved content
            all_text = " ".join([r.chunk_text.lower() for r in results])
            found_keywords = [kw for kw in test["expected_keywords"] if kw in all_text]
            
            print(f"  Found keywords: {found_keywords}")
            print(f"  Coverage: {len(found_keywords)}/{len(test['expected_keywords'])}")
            
            # Check similarity scores
            if results:
                avg_sim = sum(r.similarity for r in results) / len(results)
                print(f"  Avg similarity: {avg_sim:.3f}")
                print(f"  Top similarity: {results[0].similarity:.3f}")
            
            print()
    
    def test_ranking_consistency(self):
        """Test that ranking is consistent and logical."""
        print("=" * 80)
        print("RANKING CONSISTENCY TEST")
        print("=" * 80)
        print()
        
        # Run same query multiple times
        quiz = {
            "use_case": ["gaming"],
            "budget": ["value"],
            "portability": "performance"
        }
        
        print("Running same query 3 times to check consistency...")
        results_list = []
        
        for i in range(3):
            retrieval_results = self.retriever.retrieve(quiz_response=quiz, top_k=15)
            recommendations = self.ranker.rank(retrieval_results, quiz, top_k=5)
            results_list.append([rec.product_name for rec in recommendations])
            print(f"  Run {i+1}: {results_list[-1][0]}")  # Top recommendation
        
        # Check if top recommendation is consistent
        top_recs = [r[0] for r in results_list]
        if len(set(top_recs)) == 1:
            print("[+] Top recommendation is consistent")
        else:
            print(f"[-] Top recommendation varies: {top_recs}")
        
        print()


def main():
    """Run comprehensive RAG evaluation."""
    evaluator = RAGEvaluator()
    
    # Main evaluation
    results = evaluator.evaluate_all()
    
    # Additional tests
    evaluator.test_retrieval_quality()
    evaluator.test_ranking_consistency()
    
    # Save results
    output_file = Path(__file__).parent / "rag_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Exit code based on results
    if results["failed"] == 0:
        print("\n[+] All tests passed!")
        return 0
    else:
        print(f"\n[-] {results['failed']} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
