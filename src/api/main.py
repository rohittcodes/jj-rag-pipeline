"""
FastAPI application for laptop recommendation RAG service.
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.retriever import RAGRetriever
from src.rag.ranker import RAGRanker
from src.rag.spec_fallback import SpecFallbackRecommender
from src.api.webhooks import router as webhook_router

# Initialize FastAPI app
app = FastAPI(
    title="JustJosh RAG API",
    description="Laptop recommendation service using RAG (Retrieval-Augmented Generation)",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include webhook routes
app.include_router(webhook_router)

# Initialize RAG components (lazy loading)
retriever = None
ranker = None
spec_fallback = None

# API Key for authentication
API_KEY = os.getenv('API_KEY', '')
RAG_CONFIDENCE_THRESHOLD = float(os.getenv('RAG_CONFIDENCE_THRESHOLD', '0.75'))


def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key in Authorization header."""
    if not API_KEY:
        return True
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format. Use: Bearer <API_KEY>")
    provided_key = authorization.replace("Bearer ", "").strip()
    if provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_retriever():
    """Get or initialize retriever."""
    global retriever
    if retriever is None:
        retriever = RAGRetriever(verbose=False)
    return retriever


def get_ranker():
    """Get or initialize ranker."""
    global ranker
    if ranker is None:
        ranker = RAGRanker(verbose=False)
    return ranker


def get_spec_fallback():
    """Get or initialize spec fallback recommender."""
    global spec_fallback
    if spec_fallback is None:
        spec_fallback = SpecFallbackRecommender(verbose=False)
    return spec_fallback


def log_rag_query(
    query_text: str,
    quiz_response: Dict,
    top_results: List,
    confidence_score: float,
    recommendation_source: str,
    latency_ms: int
):
    """Log RAG query to database for monitoring."""
    try:
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'josh_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Format top results for storage
        results_json = [
            {
                'config_id': r.get('config_id'),
                'product_name': r.get('product_name'),
                'confidence': r.get('confidence_score')
            }
            for r in top_results[:5]
        ]
        
        cursor.execute("""
            INSERT INTO rag_query_logs
            (query_text, top_k, results, response_time_ms)
            VALUES (%s, %s, %s, %s)
        """, (
            query_text,
            len(top_results),
            psycopg2.extras.Json({'quiz': quiz_response, 'recommendations': results_json, 'source': recommendation_source, 'confidence': confidence_score}),
            latency_ms
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[!] Failed to log query: {e}")


# Request/Response Models
class QuizResponse(BaseModel):
    """User's quiz response for laptop recommendations."""
    profession: Optional[List[str]] = Field(default=[], description="User professions (e.g., ['student', 'developer'])")
    use_case: Optional[List[str]] = Field(default=[], description="Primary use cases (e.g., ['programming', 'gaming'])")
    budget: Optional[List[str]] = Field(default=[], description="Budget range (e.g., ['budget', 'value', 'premium'])")
    portability: Optional[str] = Field(default=None, description="Portability preference (light/somewhat/performance)")
    screen_size: Optional[List[str]] = Field(default=[], description="Preferred screen sizes (e.g., ['13-14 inch', '15-16 inch'])")
    
    class Config:
        json_schema_extra = {
            "example": {
                "profession": ["student", "developer"],
                "use_case": ["programming", "video_editing"],
                "budget": ["value", "premium"],
                "portability": "light",
                "screen_size": ["14 inch", "15-16 inch"]
            }
        }


class ProductRecommendation(BaseModel):
    """Single product recommendation."""
    product_name: str
    confidence_score: float
    josh_score: float
    spec_score: float
    ranking: Optional[int] = None
    source_article: str
    source_url: str
    explanation: str
    config_id: Optional[int] = None


class RecommendationResponse(BaseModel):
    """Response containing product recommendations."""
    recommendations: List[ProductRecommendation]
    query: str
    total_results: int


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "JustJosh RAG API",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    try:
        # Test retriever
        retriever = get_retriever()
        
        # Test ranker
        ranker = get_ranker()
        
        return {
            "status": "healthy",
            "retriever": "ready",
            "ranker": "ready"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    quiz_response: QuizResponse,
    top_k: int = 5,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Get laptop recommendations based on quiz response.
    
    Requires API key authentication if API_KEY is set in environment.
    
    Args:
        quiz_response: User's quiz answers
        top_k: Number of recommendations to return (default: 5)
    
    Returns:
        RecommendationResponse with ranked product recommendations
    """
    start_time = time.time()
    recommendation_source = "josh_rag_primary"
    
    try:
        # Get components
        retriever = get_retriever()
        ranker = get_ranker()
        
        # Convert to dict
        quiz_dict = quiz_response.model_dump()
        
        # Generate query for logging
        query = retriever.construct_query(quiz_dict)
        
        # Retrieve relevant content
        retrieval_results = retriever.retrieve(
            quiz_response=quiz_dict,
            top_k=20
        )
        
        # Check confidence
        is_confident = retrieval_results and retrieval_results[0].similarity >= RAG_CONFIDENCE_THRESHOLD
        
        if not retrieval_results or not is_confident:
            # Use spec-based fallback
            recommendation_source = "spec_fallback_low_confidence"
            fallback = get_spec_fallback()
            spec_recommendations = fallback.recommend(quiz_dict, top_k=top_k)
            
            if not spec_recommendations:
                raise HTTPException(
                    status_code=404,
                    detail="No recommendations found matching your criteria"
                )
            
            # Convert spec recommendations to standard format
            recommendation_list = [
                ProductRecommendation(
                    product_name=rec.product_name,
                    confidence_score=rec.confidence_score,
                    josh_score=0.0,
                    spec_score=rec.spec_score,
                    ranking=None,
                    source_article="Spec-based recommendation",
                    source_url="",
                    explanation=rec.explanation,
                    config_id=rec.config_id
                )
                for rec in spec_recommendations
            ]
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Log query
            log_rag_query(
                query_text=query,
                quiz_response=quiz_dict,
                top_results=[r.model_dump() for r in recommendation_list],
                confidence_score=spec_recommendations[0].confidence_score if spec_recommendations else 0.0,
                recommendation_source=recommendation_source,
                latency_ms=latency_ms
            )
            
            return RecommendationResponse(
                recommendations=recommendation_list,
                query=query,
                total_results=len(recommendation_list)
            )
        
        # RAG path: rank products
        recommendations = ranker.rank(
            retrieval_results=retrieval_results,
            quiz_response=quiz_dict,
            top_k=top_k
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No product recommendations found"
            )
        
        # Convert to response format
        recommendation_list = [
            ProductRecommendation(
                product_name=rec.product_name,
                confidence_score=rec.confidence_score,
                josh_score=rec.josh_score,
                spec_score=rec.spec_score,
                ranking=rec.ranking,
                source_article=rec.source_article,
                source_url=rec.source_url,
                explanation=rec.explanation,
                config_id=rec.config_id
            )
            for rec in recommendations
        ]
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log query
        log_rag_query(
            query_text=query,
            quiz_response=quiz_dict,
            top_results=[r.model_dump() for r in recommendation_list],
            confidence_score=recommendations[0].confidence_score if recommendations else 0.0,
            recommendation_source=recommendation_source,
            latency_ms=latency_ms
        )
        
        return RecommendationResponse(
            recommendations=recommendation_list,
            query=query,
            total_results=len(recommendation_list)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@app.post("/search")
async def search_content(
    query: str,
    top_k: int = 10,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Direct semantic search in Josh's content.
    
    Requires API key authentication if API_KEY is set in environment.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
    
    Returns:
        List of relevant content chunks
    """
    try:
        retriever = get_retriever()
        
        results = retriever.retrieve(
            query=query,
            top_k=top_k
        )
        
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "content_title": r.metadata.get("content_title"),
                    "chunk_text": r.chunk_text[:500] + "..." if len(r.chunk_text) > 500 else r.chunk_text,
                    "similarity": r.similarity,
                    "section_title": r.section_title,
                    "url": r.metadata.get("url")
                }
                for r in results
            ],
            "total_results": len(results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
