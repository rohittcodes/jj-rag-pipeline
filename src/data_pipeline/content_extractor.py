"""
Content extractor and chunker for all content types.
Extracts structured content, metadata, and chunks text for embedding.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import os


class ContentExtractor:
    """Extracts and processes content from various sources (blogs, YouTube, PDFs)."""
    
    def __init__(self):
        self.base_url = "https://justjosh.tech/blogs"
    
    def extract_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract content from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self.extract_from_json(data, file_path.stem)
    
    def extract_from_json(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """
        Extract content from parsed JSON data.
        
        Args:
            data: Parsed JSON data from Sanity CMS
            filename: Original filename (used for URL if slug missing)
            
        Returns:
            Dictionary with extracted content and metadata
        """
        # Extract metadata
        title = data.get('title', '')
        slug = data.get('slug', {}).get('current', filename)
        url = f"{self.base_url}/{slug}"
        
        # Parse publish date
        published_at = data.get('publishedAt')
        publish_date = None
        if published_at:
            try:
                publish_date = datetime.fromisoformat(published_at.replace('Z', '+00:00')).date()
            except Exception:
                pass
        
        # Extract category as tag
        category = data.get('category', '')
        tags = [category] if category else []
        
        # Extract and process body content
        body_blocks = data.get('body', [])
        raw_content, structured_sections = self._process_body(body_blocks)
        
        # Build structured data
        structured_data = {
            'sections': structured_sections,
            'meta_description': data.get('metaDescription', ''),
            'meta_title': data.get('metaTitle', ''),
            'read_count': data.get('readCount', 0),
            'main_image': data.get('mainImage', {}),
        }
        
        return {
            'content_type': 'blog',
            'title': title,
            'url': url,
            'publish_date': publish_date,
            'raw_content': raw_content,
            'structured_data': structured_data,
            'tags': tags,
        }
    
    def _process_body(self, blocks: List[Dict]) -> tuple[str, List[Dict]]:
        """
        Process body blocks into raw text and structured sections.
        
        Args:
            blocks: List of Sanity CMS body blocks
            
        Returns:
            Tuple of (raw_content_text, structured_sections)
        """
        raw_parts = []
        sections = []
        current_section = None
        
        for block in blocks:
            block_type = block.get('_type')
            
            if block_type == 'block':
                text = self._extract_block_text(block)
                style = block.get('style', 'normal')
                
                # Handle headings - create new sections
                if style in ['h2', 'h3', 'h4']:
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        'heading': text,
                        'level': style,
                        'content': [],
                        'config_ids': []
                    }
                    raw_parts.append(f"\n\n## {text}\n")
                
                # Handle normal text and lists
                elif style == 'normal':
                    list_item = block.get('listItem')
                    if list_item == 'bullet':
                        formatted_text = f"• {text}"
                        raw_parts.append(formatted_text)
                    elif list_item == 'number':
                        formatted_text = f"- {text}"
                        raw_parts.append(formatted_text)
                    else:
                        raw_parts.append(text)
                    
                    if current_section:
                        current_section['content'].append(text)
                
            elif block_type == 'image':
                # Handle image blocks
                alt_text = block.get('alt', '')
                if alt_text:
                    raw_parts.append(f"[Image: {alt_text}]")
            
            elif block_type == 'configEmbed':
                # Handle product configuration embeds
                configs = block.get('configs', [])
                config_ids = [int(cfg.get('key')) for cfg in configs if cfg.get('key')]
                
                if config_ids and current_section:
                    current_section['config_ids'].extend(config_ids)
                    raw_parts.append(f"[Product configs: {', '.join(map(str, config_ids))}]")
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        raw_content = '\n'.join(raw_parts)
        return raw_content, sections
    
    def _extract_block_text(self, block: Dict) -> str:
        """
        Extract text from a Sanity block.
        
        Args:
            block: Sanity block dictionary
            
        Returns:
            Extracted text string
        """
        children = block.get('children', [])
        text_parts = []
        
        for child in children:
            if child.get('_type') == 'span':
                text = child.get('text', '')
                marks = child.get('marks', [])
                
                # Apply text formatting based on marks
                if 'strong' in marks:
                    text = f"**{text}**"
                if 'em' in marks:
                    text = f"*{text}*"
                if 'code' in marks:
                    text = f"`{text}`"
                
                text_parts.append(text)
        
        return ''.join(text_parts)
    
    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Split content into overlapping chunks using recursive character splitting.
        
        Args:
            content: Text content to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not content:
            return []
            
        separators = ["\n\n", "\n", " ", ""]
        
        # Helper to split text recursively
        def _split_text(text: str, separators: List[str]) -> List[str]:
            final_chunks = []
            separator = separators[-1]
            new_separators = []
            
            for i, _s in enumerate(separators):
                if _s == "":
                    separator = _s
                    break
                if _s in text:
                    separator = _s
                    new_separators = separators[i+1:]
                    break
            
            _splits = text.split(separator) if separator else list(text)
            good_splits = []
            
            for s in _splits:
                if len(s) < chunk_size:
                    good_splits.append(s)
                else:
                    if good_splits:
                        merged_text = self._merge_splits(good_splits, separator, chunk_size, overlap)
                        final_chunks.extend(merged_text)
                        good_splits = []
                    if not new_separators:
                        final_chunks.append(s)
                    else:
                        other_splits = _split_text(s, new_separators)
                        final_chunks.extend(other_splits)
            
            if good_splits:
                merged_text = self._merge_splits(good_splits, separator, chunk_size, overlap)
                final_chunks.extend(merged_text)
                
            return final_chunks

        text_chunks = _split_text(content, separators)
        
        # Format chunks with metadata
        return [
            {
                'text': chunk,
                'index': i,
                'char_count': len(chunk)
            }
            for i, chunk in enumerate(text_chunks)
        ]

    def _merge_splits(self, splits: List[str], separator: str, chunk_size: int, overlap: int) -> List[str]:
        """Merge smaller splits into chunks of max size with overlap."""
        docs = []
        current_doc = []
        total = 0
        separator_len = len(separator)
        
        for d in splits:
            _len = len(d)
            if total + _len + (separator_len if current_doc else 0) > chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc)
                    
                    # Keep valid overlap
                    while total > overlap or (total + _len + separator_len > chunk_size and total > 0):
                        total -= len(current_doc[0]) + separator_len
                        current_doc.pop(0)
            
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
            
        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc)
                
        return docs

    def extract_and_chunk(self, content: str, title: str = "", chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Extract and chunk content.
        
        Args:
            content: Raw text content
            title: Content title (optional)
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap
            
        Returns:
            List of chunk dictionaries
        """
        return self.chunk_content(content, chunk_size, overlap)
    
    def extract_product_mentions(self, text: str) -> List[str]:
        """
        Extract product mentions from text using regex patterns (Fallback).
        """
        import re
        
        # Common laptop product patterns
        patterns = [
            r'(?:MacBook|ThinkPad|XPS|Spectre|Envy|Pavilion|IdeaPad|Yoga|VivoBook|ZenBook|ROG|TUF|Legion|Nitro|Aspire|Swift|Predator|Victus|Omen|ProBook|EliteBook|Latitude|Precision|Surface|Galaxy Book|Gram|Blade|Aero|Creator|Prestige|Modern|Summit|Stealth|Scar|Zephyrus|Flow|Strix|Raider|Vector|Katana|Cyborg|Pulse|Bravo|Alpha|Delta|Crosshair|Slim|Pro|Air|Max|Plus|Ultra)\s+(?:\w+\s+)?(?:\d+(?:[a-zA-Z0-9-]+)?)',
            r'(?:HP|Dell|Lenovo|Asus|Acer|MSI|Razer|Apple|Samsung|LG|Microsoft|Gigabyte|Alienware)\s+(?:\w+\s+)?(?:\d+(?:[a-zA-Z0-9-]+)?)',
        ]
        
        mentions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.extend(matches)
        
        return list(set([m.strip() for m in mentions if m.strip()]))

    def extract_product_mentions_llm(self, text: str, openai_client=None) -> List[str]:
        """
        Extract product mentions using LLM for higher accuracy.
        
        Args:
            text: Text chunk
            openai_client: Initialized OpenAI client (optional, will create one if None)
            
        Returns:
            List of product names
        """
        from openai import OpenAI
        import os
        
        if not text or len(text) < 10:
            return []
            
        client = openai_client
        if not client:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                # Fallback silently or log if needed, but user wants LLM. 
                # If key missing, regex is better than nothing.
                return self.extract_product_mentions(text)
            client = OpenAI(api_key=api_key)
            
        system_prompt = """You are a precise entity extractor. 
Extract all distinct laptop product names mentioned in the text.
Return JSON list of strings.
Example: ["MacBook Pro 14", "Dell XPS 13"]
Rules:
- Only extract LAPTOP names
- Include model numbers and series (e.g., "Lenovo Legion 7i", not just "Lenovo")
- Filter out general terms like "my laptop" or "the computer"
- If no products found, return []
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            # Clean markdown if present
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            import json
            products = json.loads(content)
            
            if isinstance(products, list):
                return [str(p) for p in products]
            return []
            
        except Exception as e:
            print(f"[-] LLM extraction error: {e}")
            return self.extract_product_mentions(text)  # Fallback


def extract_all_blogs(blogs_dir: Path) -> List[Dict[str, Any]]:
    """Extract content from all JSON files in a directory."""
    extractor = ContentExtractor()
    extracted_blogs = []
    
    json_files = list(blogs_dir.glob('*.json'))
    print(f"[*] Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            content = extractor.extract_from_file(json_file)
            extracted_blogs.append(content)
            print(f"[+] {content['title']}")
        except Exception as e:
            print(f"[-] Error: {json_file.name}: {e}")
    
    return extracted_blogs


if __name__ == "__main__":
    from pathlib import Path
    
    blogs_dir = Path(__file__).parent.parent.parent / "raw" / "blogs"
    blogs = extract_all_blogs(blogs_dir)
    
    print(f"[+] Extracted {len(blogs)} blogs")
    for blog in blogs:
        print(f"  - {blog['title']}")
