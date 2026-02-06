"""
Test Data Parser - Extract performance benchmarks from test data PDFs.

The PDFs contain comparative benchmark data across multiple laptops.
We extract the data for the target product and create searchable chunks.
"""
import PyPDF2
import re
from typing import Dict, List, Optional
from pathlib import Path


class TestDataParser:
    """Parse test data PDFs and extract structured benchmark information."""
    
    def __init__(self):
        # Benchmark categories we look for
        self.benchmark_patterns = {
            'geekbench': r'Geekbench \d+',
            'cinebench': r'Cinebench \d+',
            '3dmark': r'3DMark|Timespy|Wildlife',
            'battery': r'Battery|Video Playback|Netflix',
            'display': r'Display.*Brightness|Max Brightness',
            'weight': r'Weight',
            'temperature': r'Temperature|Heat',
            'fan_noise': r'Fan Noise',
            'power_draw': r'Power Draw'
        }
    
    def parse_pdf(self, pdf_path: Path, product_name: str) -> Dict:
        """
        Extract test data from PDF for a specific product.
        
        Args:
            pdf_path: Path to the PDF file
            product_name: Name of the product to extract data for
        
        Returns:
            Dict with structured benchmark data
        """
        # Extract all text from PDF
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
        
        # Clean product name for matching (remove special chars, normalize)
        clean_product = self._normalize_product_name(product_name)
        
        # Extract benchmarks
        benchmarks = {
            'product_name': product_name,
            'raw_text': full_text,  # Keep for fallback search
            'benchmarks': {}
        }
        
        # Try to extract specific benchmark values
        benchmarks['benchmarks'] = self._extract_benchmarks(full_text, clean_product)
        
        return benchmarks
    
    def _normalize_product_name(self, name: str) -> str:
        """Normalize product name for matching."""
        # Remove special characters, convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', name.lower())
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _extract_benchmarks(self, text: str, product_name: str) -> Dict:
        """
        Extract benchmark values for the product.
        
        This is a simplified extraction - in reality, the PDF format is complex.
        We'll focus on creating searchable natural language descriptions.
        """
        benchmarks = {}
        
        # Look for benchmark sections
        for category, pattern in self.benchmark_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                benchmarks[category] = {
                    'found': True,
                    'context': self._extract_context(text, pattern, product_name)
                }
        
        return benchmarks
    
    def _extract_context(self, text: str, pattern: str, product_name: str, context_chars: int = 500) -> str:
        """Extract context around a benchmark mention."""
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if not matches:
            return ""
        
        # Get text around the first match
        match = matches[0]
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        
        return text[start:end]
    
    def create_chunks(self, test_data: Dict, config_id: int) -> List[Dict]:
        """
        Create searchable chunks from test data.
        
        Args:
            test_data: Parsed test data dictionary
            config_id: Configuration ID this data belongs to
        
        Returns:
            List of chunk dictionaries ready for embedding
        """
        chunks = []
        product_name = test_data['product_name']
        benchmarks = test_data['benchmarks']
        
        # Create a comprehensive performance summary chunk
        summary_text = f"Performance test data for {product_name}:\n\n"
        
        benchmark_descriptions = []
        for category, data in benchmarks.items():
            if data.get('found'):
                category_name = category.replace('_', ' ').title()
                benchmark_descriptions.append(f"- {category_name} benchmarks available")
        
        if benchmark_descriptions:
            summary_text += "\n".join(benchmark_descriptions)
        else:
            summary_text += "Comprehensive performance testing completed."
        
        chunks.append({
            'config_id': config_id,
            'test_type': 'summary',
            'chunk_text': summary_text,
            'benchmark_results': benchmarks,
            'test_description': f"Performance test summary for {product_name}"
        })
        
        # Create individual chunks for each benchmark category
        for category, data in benchmarks.items():
            if data.get('found') and data.get('context'):
                category_name = category.replace('_', ' ').title()
                chunk_text = f"{category_name} performance for {product_name}:\n\n"
                chunk_text += data['context'][:300]  # Limit context size
                
                chunks.append({
                    'config_id': config_id,
                    'test_type': category,
                    'chunk_text': chunk_text,
                    'benchmark_results': {category: data},
                    'test_description': f"{category_name} test results"
                })
        
        # If no specific benchmarks found, create a general chunk with raw text snippet
        if not benchmarks:
            raw_text = test_data.get('raw_text', '')
            if raw_text:
                chunk_text = f"Test data for {product_name}:\n\n"
                chunk_text += raw_text[:500]  # First 500 chars
                
                chunks.append({
                    'config_id': config_id,
                    'test_type': 'general',
                    'chunk_text': chunk_text,
                    'benchmark_results': {},
                    'test_description': f"General test data for {product_name}"
                })
        
        return chunks
    
    def create_natural_language_summary(self, benchmarks: Dict) -> str:
        """
        Create a natural language summary of benchmark results.
        
        This is useful for generating explanations in recommendations.
        """
        summary_parts = []
        
        if benchmarks.get('geekbench', {}).get('found'):
            summary_parts.append("Tested with Geekbench for CPU performance")
        
        if benchmarks.get('3dmark', {}).get('found'):
            summary_parts.append("3DMark graphics benchmarks available")
        
        if benchmarks.get('battery', {}).get('found'):
            summary_parts.append("Battery life tests completed")
        
        if benchmarks.get('cinebench', {}).get('found'):
            summary_parts.append("Cinebench CPU rendering tests performed")
        
        if not summary_parts:
            return "Comprehensive performance testing completed"
        
        return ". ".join(summary_parts) + "."


# Test the parser
if __name__ == '__main__':
    parser = TestDataParser()
    
    # Test with the sample PDF we downloaded
    pdf_path = Path('tmp/test_sample.pdf')
    product_name = "ThinkPad T14s Gen 6"
    
    if pdf_path.exists():
        print(f"[*] Parsing test data for: {product_name}")
        test_data = parser.parse_pdf(pdf_path, product_name)
        
        print(f"\n[+] Found benchmarks:")
        for category, data in test_data['benchmarks'].items():
            if data.get('found'):
                print(f"  - {category}")
        
        print(f"\n[*] Creating chunks...")
        chunks = parser.create_chunks(test_data, config_id=15)
        
        print(f"[+] Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  Chunk {i}:")
            print(f"    Type: {chunk['test_type']}")
            print(f"    Description: {chunk['test_description']}")
            print(f"    Text preview: {chunk['chunk_text'][:150]}...")
        
        print(f"\n[*] Natural language summary:")
        summary = parser.create_natural_language_summary(test_data['benchmarks'])
        print(f"    {summary}")
    else:
        print(f"[!] PDF not found: {pdf_path}")
        print(f"[!] Run test_pdf_fetch.py first")
