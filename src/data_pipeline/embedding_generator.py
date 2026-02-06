"""
Embedding generator using sentence-transformers.
Generates vector embeddings for text chunks.
"""
import os
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers models."""
    
    def __init__(self, model_name: str = None, verbose: bool = True):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
                       Defaults to EMBEDDING_MODEL from .env
            verbose: Whether to print loading messages
        """
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'intfloat/e5-base-v2')
        
        if verbose:
            print(f"[*] Loading embedding model: {self.model_name}")
            print(f"[*] This may take a while on first run (downloading ~400MB)...")
        
        # Load the model (downloads automatically if not cached)
        self.model = SentenceTransformer(self.model_name)
        
        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        if verbose:
            print(f"[+] Model loaded successfully")
            print(f"[+] Embedding dimension: {self.dimension}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding vector
        """
        # For e5 models, prefix with "query: " or "passage: "
        # Use "passage: " for content we're indexing
        if 'e5' in self.model_name.lower():
            text = f"passage: {text}"
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        # For e5 models, prefix with "passage: "
        if 'e5' in self.model_name.lower():
            texts = [f"passage: {text}" for text in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Uses "query: " prefix for e5 models.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector for the query
        """
        # For e5 models, prefix with "query: " for search queries
        if 'e5' in self.model_name.lower():
            query = f"query: {query}"
        
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()


if __name__ == "__main__":
    # Test the embedding generator
    print("[*] Testing embedding generator...")
    
    generator = EmbeddingGenerator()
    
    # Test single embedding
    test_text = "This is a test sentence about laptops."
    embedding = generator.generate_embedding(test_text)
    
    print(f"\n[*] Test text: '{test_text}'")
    print(f"[+] Embedding dimension: {len(embedding)}")
    print(f"[+] First 5 values: {embedding[:5]}")
    
    # Test batch embeddings
    test_texts = [
        "Best laptops for programming",
        "Gaming laptops with long battery life",
        "Thin and light performance laptops"
    ]
    
    print(f"\n[*] Generating embeddings for {len(test_texts)} texts...")
    embeddings = generator.generate_embeddings(test_texts, show_progress=True)
    
    print(f"[+] Generated {len(embeddings)} embeddings")
    print(f"[+] Each embedding has {len(embeddings[0])} dimensions")
    
    # Test query embedding
    test_query = "laptop for video editing"
    query_embedding = generator.generate_query_embedding(test_query)
    
    print(f"\n[*] Query: '{test_query}'")
    print(f"[+] Query embedding dimension: {len(query_embedding)}")
