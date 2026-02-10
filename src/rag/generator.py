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
            metadata = chunk.get('metadata', {})
            source_type = metadata.get('source_type', 'unknown')
            title = metadata.get('content_title', 'Unknown')
            url = metadata.get('url', '')
            similarity = chunk.get('similarity', 0.0)
            
            # Add to context with source metadata
            context_parts.append(f"[Source {idx}] (Source: {source_type.capitalize()} - {title})\nContent: {chunk_text}")
            
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
    
    def _build_prompt(self, query: str, context: str, recommended_products: str = "") -> str:
        """
        Build the prompt for answer generation with Josh's signature style that adapts to user's emotional state.
        """
        product_context = f"\n\nRECOMMENDED PRODUCTS FOR THIS USER:\n{recommended_products}\n" if recommended_products else ""
        
        prompt = f"""You are Josh, the tech world's most trusted and opinionated laptop reviewer. Your goal is to guide your "tech family" to the absolute best laptop for their specific needs.

CRITICAL: READ THE USER'S EMOTIONAL STATE AND ADAPT YOUR TONE ACCORDINGLY:
- If they seem worried, stressed, or confused → Be reassuring, patient, and break things down simply
- If they're excited or enthusiastic → Match their energy with enthusiasm
- If they're frustrated or disappointed → Be empathetic and solution-focused
- If they're asking a straightforward question → Be direct and helpful
- If they're dealing with eye strain, health concerns, or urgent needs → Be caring and prioritize their wellbeing

YOUR ADAPTIVE STYLE:
1. START NATURALLY: Don't force enthusiasm. Read the room. If someone asks "which laptop won't hurt my eyes?", start with empathy like "I hear you - eye strain is no joke" rather than "What's going on! Josh here."
2. BE OPINIONATED BUT CONTEXTUAL: Tell them exactly what to buy, but adjust your delivery based on their state. Use phrases like:
   - Worried user: "Look, I'm gonna make this simple for you..."
   - Excited user: "Alright, let's find you something awesome..."
   - Frustrated user: "I get it, let me help you cut through the noise..."
3. FOCUS ON REAL-WORLD DETAILS: Beyond CPU/GPU, mention keyboard deck, trackpad quality, fan noise, heat, and especially display quality for eye comfort.
4. AUTHORITATIVE & CONFIDENT: Use your expertise to form definitive opinions based on the data.
5. NO WAFFLING: If a configuration is bad value, call it out clearly.
6. CITE YOUR SOURCES: Use format: [Source X] (Josh said in [Blog/Youtube: summary]).
7. **CRITICAL: ONLY recommend products from the RECOMMENDED PRODUCTS list below. Do NOT make up or suggest products that aren't in this list.**

CONTEXT FROM YOUR REVIEWS & TECHNICAL SPECS:
{context}{product_context}

USER QUESTION:
{query}

YOUR ADAPTIVE JOSH RECOMMENDATION (WITH CITATIONS):"""
        
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
            metadata = chunk.get('metadata', {})
            source_type = metadata.get('source_type', 'unknown')
            title = metadata.get('content_title', 'Unknown')
            url = metadata.get('url', '')
            similarity = chunk.get('similarity', 0.0)
            
            context_parts.append(f"[Source {idx}] (Source: {source_type.capitalize()} - {title})\nContent: {chunk_text}")
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
3. If the context doesn't contain enough information about this product, say so honestly but also be helpful and tell them straightforward about the specs and other use cases
4. Be conversational and helpful
5. Mention specific pros, cons, and use cases when relevant
6. Don't make up information or use knowledge outside the provided context

CONTEXT FROM YOUR CONTENT:
{context}

USER QUESTION:
{query}

Provide your expert opinion on the {product_name}, citing sources using the format: [Source X] (Josh said in [Blog/Youtube: summary]).
YOUR ANSWER:"""
        
        if self.verbose:
            print(f"\n[*] Generating product-focused answer for: {product_name}")
        
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
                'confidence': avg_similarity
            }
            
        except Exception as e:
            if self.verbose:
                print(f"[!] Error generating answer: {e}")
            
            return {
                'answer': f"Error: {str(e)}",
                'sources': sources,
                'confidence': 0.0
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
            
            metadata = chunk_data.get('metadata', {})
            source_type = metadata.get('source_type', 'unknown')
            title = metadata.get('content_title', 'Unknown')
            
            context_parts.append(f"[Source {idx}] (Source: {source_type.capitalize()} - {title})\nContent: {chunk_data.get('chunk_text', '')}")
        
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

    def generate_stream_with_products(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        recommended_products: List[Any],
        max_context_chunks: int = 5
    ):
        """
        Stream the answer generation token by token, with recommended products context.
        
        Args:
            query: User's question
            retrieved_chunks: List of ranked chunks from retriever
            recommended_products: List of recommended product objects from ranker
            max_context_chunks: Maximum number of chunks to include
        
        Yields:
            String tokens of the answer
        """
        if not retrieved_chunks:
            yield "I don't have enough information to answer that question based on Josh's content."
            return
        
        # Limit context
        context_chunks = retrieved_chunks[:max_context_chunks]
        
        # Build context from retrieval results
        context_parts = []
        for idx, chunk in enumerate(context_chunks, 1):
            # Handle object vs dict
            chunk_data = chunk.to_dict() if hasattr(chunk, 'to_dict') else chunk
            if not isinstance(chunk_data, dict):
                 if hasattr(chunk, 'chunk_text'):
                     chunk_data = {'chunk_text': chunk.chunk_text}
                 else:
                     chunk_data = {}
            
            metadata = chunk_data.get('metadata', {})
            source_type = metadata.get('source_type', 'unknown')
            title = metadata.get('content_title', 'Unknown')
            
            context_parts.append(f"[Source {idx}] (Source: {source_type.capitalize()} - {title})\nContent: {chunk_data.get('chunk_text', '')}")
        
        context = "\n\n".join(context_parts)
        
        # Build product list for prompt
        product_list = []
        for idx, prod in enumerate(recommended_products, 1):
            prod_data = prod.to_dict() if hasattr(prod, 'to_dict') else prod
            if isinstance(prod_data, dict):
                product_name = prod_data.get('product_name', 'Unknown Product')
                brand = prod_data.get('brand', '')
                model = prod_data.get('model', '')
                full_name = f"{brand} {model}" if brand and model else product_name
                product_list.append(f"{idx}. {full_name}")
        
        recommended_products_text = "\n".join(product_list) if product_list else "No specific products available"
        
        # Build prompt with product context
        prompt = self._build_prompt(query, context, recommended_products_text)
        
        if self.verbose:
            print(f"\n[*] Streaming answer with {len(recommended_products)} recommended products")
        
        try:
            # Stream response
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt
            ):
                yield chunk.text
                
        except Exception as e:
            if self.verbose:
                print(f"[!] Error streaming answer: {e}")
            yield f"Error: {str(e)}"
