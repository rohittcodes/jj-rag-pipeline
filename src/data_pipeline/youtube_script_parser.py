"""
YouTube Script Parser - Extract text from .docx video script files.

Parses Word documents containing YouTube video scripts and extracts
clean text content for ingestion into the RAG pipeline.
"""
from pathlib import Path
from typing import Optional, Dict
from docx import Document


class YouTubeScriptParser:
    """
    Parse YouTube video scripts from .docx files.
    
    Handles:
    - Text extraction from Word documents
    - Metadata extraction from folder structure
    - File selection (main scripts vs. B-roll/editing notes)
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize the parser."""
        self.verbose = verbose
    
    def parse_script(self, file_path: Path) -> Optional[Dict]:
        """
        Parse a YouTube script .docx file.
        
        Args:
            file_path: Path to the .docx file
            
        Returns:
            Dictionary with:
            - text: Full script text
            - video_id: Extracted video ID (e.g., "T223")
            - title: Video title from folder name
            - file_name: Original file name
        """
        try:
            # Extract metadata from path
            folder_name = file_path.parent.name
            video_id = self._extract_video_id(folder_name)
            title = self._extract_title(folder_name)
            
            # Parse document
            doc = Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            if not paragraphs:
                if self.verbose:
                    print(f"[!] No content in {file_path.name}")
                return None
            
            # Join paragraphs with double newline
            full_text = '\n\n'.join(paragraphs)
            
            if self.verbose:
                print(f"[+] Parsed {file_path.name}: {len(paragraphs)} paragraphs, {len(full_text)} chars")
            
            return {
                'text': full_text,
                'video_id': video_id,
                'title': title,
                'file_name': file_path.name,
                'paragraph_count': len(paragraphs)
            }
            
        except Exception as e:
            if self.verbose:
                print(f"[!] Error parsing {file_path}: {e}")
            return None
    
    def _extract_video_id(self, folder_name: str) -> str:
        """
        Extract video ID from folder name.
        
        Examples:
        - "T223 Omnibook 7 Aero" -> "T223"
        - "T207 Legion 7i" -> "T207"
        """
        parts = folder_name.split()
        if parts and parts[0].startswith('T') and parts[0][1:].isdigit():
            return parts[0]
        return folder_name
    
    def _extract_title(self, folder_name: str) -> str:
        """
        Extract video title from folder name.
        
        Examples:
        - "T223 Omnibook 7 Aero" -> "Omnibook 7 Aero"
        - "T207 Legion 7i" -> "Legion 7i"
        """
        parts = folder_name.split(maxsplit=1)
        if len(parts) > 1:
            return parts[1]
        return folder_name
    
    def select_main_script(self, folder_path: Path) -> Optional[Path]:
        """
        Select the main script file from a folder, ignoring B-roll and editing notes.
        
        Priority:
        1. File matching folder name exactly (e.g., "T223 OmniBook 7 Aero.docx")
        2. File with "Script" in name (highest version number)
        3. Shortest filename (likely the main script)
        
        Excludes:
        - Files with "B Roll" or "B-Roll"
        - Files with "Editing"
        - Files with "notes"
        - Files with "VO" (voice over notes)
        
        Args:
            folder_path: Path to the video folder
            
        Returns:
            Path to the main script file, or None if not found
        """
        docx_files = list(folder_path.glob('*.docx'))
        
        if not docx_files:
            return None
        
        # Filter out B-roll, editing, and notes files
        exclude_keywords = ['b roll', 'b-roll', 'editing', 'notes', 'vo', 'old']
        main_scripts = [
            f for f in docx_files
            if not any(keyword in f.name.lower() for keyword in exclude_keywords)
        ]
        
        if not main_scripts:
            if self.verbose:
                print(f"[!] No main script found in {folder_path.name} (only B-roll/editing files)")
            return None
        
        # Priority 1: Exact folder name match
        folder_name = folder_path.name
        for script in main_scripts:
            if script.stem == folder_name:
                return script
        
        # Priority 2: File with "Script" and highest version
        script_files = [f for f in main_scripts if 'script' in f.name.lower()]
        if script_files:
            # Sort by version number (v3 > v2 > v1) or by name
            script_files.sort(key=lambda f: f.name, reverse=True)
            return script_files[0]
        
        # Priority 3: Shortest filename (likely the main script)
        main_scripts.sort(key=lambda f: len(f.name))
        return main_scripts[0]


if __name__ == "__main__":
    """Test the parser on sample files."""
    from pathlib import Path
    
    parser = YouTubeScriptParser(verbose=True)
    
    # Test on sample folders
    youtube_dir = Path(__file__).parent.parent.parent / 'raw' / 'youtube'
    
    test_folders = [
        youtube_dir / '2025' / 'T223 Omnibook 7 Aero',
        youtube_dir / '2025' / 'T207 Legion 7i',
    ]
    
    for folder in test_folders:
        if folder.exists():
            print(f"\n{'='*80}")
            print(f"Testing: {folder.name}")
            print('='*80)
            
            main_script = parser.select_main_script(folder)
            if main_script:
                print(f"Selected: {main_script.name}")
                
                result = parser.parse_script(main_script)
                if result:
                    print(f"Video ID: {result['video_id']}")
                    print(f"Title: {result['title']}")
                    print(f"Paragraphs: {result['paragraph_count']}")
                    print(f"Text preview: {result['text'][:200]}...")
