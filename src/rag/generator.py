"""
RAG Generator - Generate natural language answers from retrieved context.
"""
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai

load_dotenv()


class RAGGenerator:
    """
    Generate natural language answers using retrieved context chunks.
    Uses Gemini Flash for fast, cost-effective generation.
    """
    
    def __init__(self, model_name: str = 'gemini-3-flash-preview', verbose: bool = False):
        """
        Initialize the RAG generator.
        
        Args:
            model_name: Gemini model to use for generation
            verbose: Whether to print detailed output
        """
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.verbose = verbose
        
        if verbose:
            print(f"[+] RAG Generator initialized with {model_name}")
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved context chunks.
        
        Args:
            query: User's question
            retrieved_chunks: List of ranked chunks from retriever
            max_context_chunks: Maximum number of chunks to include in context
        
        Returns:
            Dictionary containing:
                - answer: Generated answer text
                - sources: List of sources used
                - confidence: Overall confidence score
        """
        if not retrieved_chunks:
            return {
                'answer': "I don't have enough information to answer that question based on Josh's content.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Limit context to top chunks
        context_chunks = retrieved_chunks[:max_context_chunks]
        
        # Build context string
        context_parts = []
        sources = []
        
        for idx, chunk in enumerate(context_chunks, 1):
            # Extract relevant info
            chunk_text = chunk.get('chunk_text', '')
            source_type = chunk.get('source_type', 'unknown')
            title = chunk.get('title', 'Unknown')
            url = chunk.get('url', '')
            similarity = chunk.get('similarity', 0.0)
            
            # Add to context
            context_parts.append(f"[Source {idx}] {chunk_text}")
            
            # Track sources
            sources.append({
                'title': title,
                'type': source_type,
                'url': url,
                'relevance': similarity
            })
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        if self.verbose:
            print(f"\n[*] Generating answer for: {query}")
            print(f"[*] Using {len(context_chunks)} context chunks")
        
        # Generate answer
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            answer = response.text.strip()
            
            # Calculate overall confidence based on retrieval scores
            avg_similarity = sum(c.get('similarity', 0) for c in context_chunks) / len(context_chunks)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': avg_similarity
            }
            
        except Exception as e:
            if self.verbose:
                print(f"[!] Error generating answer: {e}")
            
            return {
                'answer': f"I encountered an error generating the answer: {str(e)}",
                'sources': sources,
                'confidence': 0.0
            }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the prompt for answer generation.
        
        Args:
            query: User's question
            context: Retrieved context chunks
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are Josh, a laptop reviewer and tech expert. Answer the user's question based ONLY on the provided context from your reviews, videos, and articles.

IMPORTANT GUIDELINES:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so honestly
3. Be conversational and helpful, as if speaking directly to the user
4. Cite specific products and details when relevant
5. If comparing products, be balanced and mention pros/cons
6. Don't make up information or use knowledge outside the provided context

CONTEXT FROM YOUR CONTENT:
{context}

USER QUESTION:
{query}

YOUR ANSWER:"""
        
        return prompt
    
    def generate_with_product_focus(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        product_name: str,
        max_context_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Generate an answer focused on a specific product.
        
        Args:
            query: User's question
            retrieved_chunks: List of ranked chunks from retriever
            product_name: Name of the product to focus on
            max_context_chunks: Maximum number of chunks to include
        
        Returns:
            Dictionary containing answer, sources, and confidence
        """
        # Filter chunks that mention the product
        product_chunks = [
            c for c in retrieved_chunks
            if product_name.lower() in c.get('chunk_text', '').lower()
        ]
        
        # If we have product-specific chunks, prioritize them
        if product_chunks:
            context_chunks = product_chunks[:max_context_chunks]
        else:
            context_chunks = retrieved_chunks[:max_context_chunks]
        
        # Build context
        context_parts = []
        sources = []
        
        for idx, chunk in enumerate(context_chunks, 1):
            chunk_text = chunk.get('chunk_text', '')
            source_type = chunk.get('source_type', 'unknown')
            title = chunk.get('title', 'Unknown')
            url = chunk.get('url', '')
            similarity = chunk.get('similarity', 0.0)
            
            context_parts.append(f"[Source {idx}] {chunk_text}")
            sources.append({
                'title': title,
                'type': source_type,
                'url': url,
                'relevance': similarity
            })
        
        context = "\n\n".join(context_parts)
        
        # Build product-focused prompt
        prompt = f"""You are Josh, a laptop reviewer and tech expert. Answer the user's question about the {product_name} based ONLY on the provided context from your reviews, videos, and articles.

IMPORTANT GUIDELINES:
1. Focus specifically on the {product_name}
2. Only use information from the provided context
3. If the context doesn't contain enough information about this product, say so honestly
4. Be conversational and helpful
5. Mention specific pros, cons, and use cases when relevant
6. Don't make up information or use knowledge outside the provided context

CONTEXT FROM YOUR CONTENT:
{context}

USER QUESTION:
{query}

YOUR ANSWER ABOUT THE {product_name}:"""
        
        if self.verbose:
            print(f"\n[*] Generating product-focused answer for: {product_name}")
            print(f"[*] Using {len(context_chunks)} context chunks")
        
        # Generate answer
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            answer = response.text.strip()
            avg_similarity = sum(c.get('similarity', 0) for c in context_chunks) / len(context_chunks)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': avg_similarity,
                'product_focus': product_name
            }
            
        except Exception as e:
            if self.verbose:
                print(f"[!] Error generating answer: {e}")
            
            return {
                'answer': f"I encountered an error generating the answer: {str(e)}",
                'sources': sources,
                'confidence': 0.0,
                'product_focus': product_name
            }

    def generate_stream(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_context_chunks: int = 5
    ):
        """
        Stream the answer generation token by token.
        
        Args:
            query: User's question
            retrieved_chunks: List of ranked chunks from retriever
            max_context_chunks: Maximum number of chunks to include
        
        Yields:
            String tokens of the answer
        """
        if not retrieved_chunks:
            yield "I don't have enough information to answer that question based on Josh's content."
            return
        
        # Limit context
        context_chunks = retrieved_chunks[:max_context_chunks]
        
        # Build context
        context_parts = []
        for idx, chunk in enumerate(context_chunks, 1):
            # Handle object vs dict
            chunk_data = chunk.to_dict() if hasattr(chunk, 'to_dict') else chunk
            if not isinstance(chunk_data, dict):
                 # Fallback if it's an object without to_dict but has attributes (like dataclass)
                 if hasattr(chunk, 'chunk_text'):
                     chunk_data = {'chunk_text': chunk.chunk_text}
                 else:
                     chunk_data = {}
            
            context_parts.append(f"[Source {idx}] {chunk_data.get('chunk_text', '')}")
        
        context = "\n\n".join(context_parts)
        prompt = self._build_prompt(query, context)
        
        if self.verbose:
            print(f"\n[*] Streaming answer for: {query}")
        
        try:
            # stream=True enables streaming
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={'response_mime_type': 'text/plain'}
            )
            
            # Re-implementation with correct streaming call pattern
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt
            ):
                yield chunk.text
                
        except Exception as e:
            if self.verbose:
                print(f"[!] Error streaming answer: {e}")
            yield f"Error: {str(e)}"
