"""
Test Data Parser - Extract performance benchmarks from test data PDFs using Gemini Flash.

This parser uses Google's Gemini Flash multimodal AI to accurately extract
benchmark data from visual PDF charts. Benefits over regex-based parsing:
1. Gemini can see the actual visual layout of charts
2. It can understand table structures and bar charts
3. It can handle N/A values contextually
4. No need for complex regex patterns
5. More accurate score extraction from visual elements
"""
import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from io import BytesIO
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class TestDataParser:
    """Parse test data PDFs using Gemini Flash vision capabilities."""
    
    def __init__(self):
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
    
    def parse_pdf(self, pdf_source: BytesIO, product_name: str) -> Dict:
        """
        Parse PDF using Gemini Flash to extract benchmark data.
        
        Args:
            pdf_source: BytesIO object containing PDF data
            product_name: Product name to extract (e.g., "Zenbook A14")
        
        Returns:
            Dict with parsed benchmarks
        """
        # Upload PDF to Gemini
        print(f"[*] Uploading PDF to Gemini...")
        
        # Save to temp file for upload
        temp_path = "temp_test_data.pdf"
        with open(temp_path, 'wb') as f:
            f.write(pdf_source.getvalue())
        
        # Upload file
        uploaded_file = self.client.files.upload(file=temp_path)
        print(f"[+] Uploaded")
        
        # Create prompt
        prompt = self._create_extraction_prompt(product_name)
        
        # Generate response
        print(f"[*] Analyzing PDF with Gemini Flash...")
        response = self.client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=[
                types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type),
                prompt
            ]
        )
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            print(f"[+] Parsed {len(result.get('benchmarks', []))} benchmarks")
            
            return result
        except json.JSONDecodeError as e:
            print(f"[!] Failed to parse JSON response: {e}")
            print(f"[!] Raw response: {response.text}")
            return {
                'product_name': product_name,
                'benchmarks': [],
                'error': str(e)
            }
    
    def _create_extraction_prompt(self, product_name: str) -> str:
        """Create prompt for Gemini to extract benchmark data."""
        return f"""
You are analyzing a test data PDF that contains benchmark results for laptop computers.

**Target Product:** {product_name}

**Your Task:**
Extract ALL benchmark scores for the product "{product_name}" from this PDF.

**PDF Structure:**
- Each page typically shows one benchmark test (Geekbench, Cinebench, 3DMark, Battery, etc.)
- Products are listed in a comparison chart (usually as bars or in a table)
- The target product "{product_name}" appears in the product list
- Some benchmarks may show "N/A" or "-" if the product wasn't tested

**Instructions:**
1. Go through EVERY page of the PDF
2. For each page with benchmark data:
   - Identify the benchmark name (e.g., "Geekbench 6", "Cinebench 2024", "3DMark Timespy")
   - Find "{product_name}" in the product list
   - Extract the numeric scores for this product
   - If the product shows "N/A", "-", or no data, SKIP that benchmark entirely
3. Return the data in the JSON format specified below

**Important:**
- Only include benchmarks where "{product_name}" has actual numeric scores
- Do NOT include benchmarks with N/A or missing data
- Extract ALL metrics for each benchmark (e.g., single-core AND multi-core for Geekbench)
- Be precise with numbers - include commas if present (e.g., 10,677)

**Output Format (JSON):**
{{
  "product_name": "{product_name}",
  "total_pages_analyzed": <number>,
  "benchmarks_found": <number>,
  "benchmarks": [
    {{
      "page_number": <int>,
      "benchmark_name": "<full name, e.g., 'Geekbench 6'>",
      "benchmark_type": "<type: geekbench|cinebench|3dmark|battery|weight|display>",
      "scores": {{
        "<metric_name>": <numeric_value>,
        ...
      }}
    }},
    ...
  ]
}}

**Example scores object:**
- Geekbench: {{"single_core": 10677, "multi_core": 2119}}
- Cinebench: {{"single_core": 142, "multi_core": 695}}
- 3DMark Timespy: {{"timespy": 2047, "graphics": 3878, "cpu": 8000}}
- 3DMark Wildlife: {{"wildlife": 3254}}
- Battery: {{"duration_minutes": 720}}
- Weight: {{"grams": 980}}

Return ONLY the JSON object, no additional text or explanation.
"""
    
    def create_chunks(self, test_data: Dict, config_id: int) -> List[Dict]:
        """
        Create semantic chunks from parsed benchmark data.
        
        Args:
            test_data: Output from parse_pdf()
            config_id: Configuration ID
        
        Returns:
            List of chunks ready for embedding and storage
        """
        chunks = []
        product_name = test_data['product_name']
        benchmarks = test_data.get('benchmarks', [])
        
        if not benchmarks:
            return chunks
        
        # Create one chunk per benchmark
        for benchmark in benchmarks:
            if not benchmark.get('scores'):
                continue
            
            # Natural language description
            chunk_text = self._create_benchmark_description(
                product_name, 
                benchmark['benchmark_name'],
                benchmark['benchmark_type'],
                benchmark['scores']
            )
            
            # Structured benchmark results
            benchmark_results = {
                benchmark['benchmark_type']: {
                    'found': True,
                    'benchmark_name': benchmark['benchmark_name'],
                    'scores': benchmark['scores']
                }
            }
            
            chunks.append({
                'config_id': config_id,
                'test_type': benchmark['benchmark_type'],
                'test_description': f"{benchmark['benchmark_name']} performance",
                'chunk_text': chunk_text,
                'benchmark_results': benchmark_results,
                'source_file': f"page_{benchmark['page_number']}"
            })
        
        # Create summary chunk
        if chunks:
            summary_text = self._create_summary_description(product_name, benchmarks)
            
            # Aggregate all benchmark results
            all_results = {}
            for benchmark in benchmarks:
                if benchmark.get('scores'):
                    bench_type = benchmark['benchmark_type']
                    all_results[bench_type] = {
                        'found': True,
                        'benchmark_name': benchmark['benchmark_name'],
                        'scores': benchmark['scores']
                    }
            
            chunks.insert(0, {
                'config_id': config_id,
                'test_type': 'summary',
                'test_description': f"Complete performance testing for {product_name}",
                'chunk_text': summary_text,
                'benchmark_results': all_results,
                'source_file': 'summary'
            })
        
        return chunks
    
    def _create_benchmark_description(
        self,
        product_name: str,
        benchmark_name: str,
        benchmark_type: str,
        scores: Dict[str, int]
    ) -> str:
        """Create natural language description of benchmark results."""
        parts = [f"{product_name} tested in {benchmark_name}."]
        
        if benchmark_type in ['geekbench', 'cinebench']:
            if 'single_core' in scores and 'multi_core' in scores:
                parts.append(f"Single-core score: {scores['single_core']}, multi-core score: {scores['multi_core']}.")
            elif 'multi_core' in scores:
                parts.append(f"Multi-core score: {scores['multi_core']}.")
        
        elif benchmark_type == '3dmark':
            if 'timespy' in scores:
                parts.append(f"3DMark Timespy score: {scores['timespy']}.")
            if 'wildlife' in scores:
                parts.append(f"Wildlife Extreme score: {scores['wildlife']}.")
            if 'graphics' in scores:
                parts.append(f"Graphics score: {scores['graphics']}.")
        
        elif benchmark_type == 'battery':
            if 'duration_minutes' in scores:
                hours = scores['duration_minutes'] / 60
                parts.append(f"Battery life: {hours:.1f} hours ({scores['duration_minutes']} minutes).")
        
        elif benchmark_type == 'weight':
            if 'grams' in scores:
                pounds = scores['grams'] / 453.592
                parts.append(f"Weight: {scores['grams']}g ({pounds:.2f} lbs).")
        
        parts.append("Tested in highest performance mode.")
        
        return " ".join(parts)
    
    def _create_summary_description(self, product_name: str, benchmarks: List[Dict]) -> str:
        """Create comprehensive summary of all benchmarks."""
        parts = [f"Complete performance testing results for {product_name}."]
        
        # Categorize benchmarks
        cpu_info = []
        gpu_info = []
        portability_info = []
        
        for bench in benchmarks:
            scores = bench.get('scores', {})
            bench_type = bench['benchmark_type']
            
            if bench_type in ['geekbench', 'cinebench']:
                if 'multi_core' in scores:
                    cpu_info.append(f"{bench['benchmark_name']}: {scores['multi_core']} multi-core")
            
            elif bench_type == '3dmark':
                if 'timespy' in scores:
                    gpu_info.append(f"3DMark Timespy: {scores['timespy']}")
            
            elif bench_type == 'battery':
                if 'duration_minutes' in scores:
                    hours = scores['duration_minutes'] / 60
                    portability_info.append(f"Battery: {hours:.1f}h")
            
            elif bench_type == 'weight':
                if 'grams' in scores:
                    portability_info.append(f"Weight: {scores['grams']}g")
        
        if cpu_info:
            parts.append(f"CPU performance: {', '.join(cpu_info)}.")
        if gpu_info:
            parts.append(f"GPU performance: {', '.join(gpu_info)}.")
        if portability_info:
            parts.append(f"Portability: {', '.join(portability_info)}.")
        
        return " ".join(parts)
