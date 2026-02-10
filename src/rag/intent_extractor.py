"""
LLM-based intent extraction for natural language queries.
Converts free-text prompts into structured quiz responses.
"""
import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class IntentExtractor:
    """Extract structured intent from natural language using LLM."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", verbose: bool = False):
        """
        Initialize intent extractor.
        
        Args:
            model: OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
            verbose: Print debug information
        """
        self.model = model
        self.verbose = verbose
        
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize OpenAI client (v1.0+ syntax)
        self.client = OpenAI(api_key=api_key)
        
        if verbose:
            print(f"[*] Intent Extractor initialized with model: {model}")
    
    def extract_intent(self, user_prompt: str, existing_intent: Optional[Dict] = None) -> Dict:
        """
        Extract structured intent from natural language prompt, optionally refining existing intent.
        
        Args:
            user_prompt: Natural language query from user
            existing_intent: Optional previously extracted intent to refine
            
        Returns:
            Structured quiz response dictionary
        """
        if self.verbose:
            print(f"\n[*] Extracting intent from prompt:")
            print(f"    '{user_prompt[:100]}...'")
            if existing_intent:
                print(f"    Refining existing intent: {json.dumps(existing_intent)}")
        
        # System prompt for intent extraction
        system_prompt = """You are an expert at understanding laptop requirements from natural language.
Your job is to understand what the user wants to DO with their laptop and map it to appropriate use cases.

AVAILABLE USE CASES (you MUST use ONLY these):
- basic_use: General everyday tasks, web browsing, email, streaming (Everyday Laptop Users)
- school: Student work, note-taking, research, assignments (Student)
- gaming: Playing video games, casual or competitive (Gamer)
- programming: Writing code, software development, using IDEs (Programmer)
- engineering: CAD, 3D modeling, simulations, technical software (Engineer)
- video_editing: Editing videos, rendering, motion graphics (Video Editor)
- trading: Stock trading, financial analysis, multiple monitors (Trader or Investor)
- data_science: Data analysis, ML, Python, Jupyter, statistical software (Data Scientist)
- interior_design: Interior design, architecture, CAD, 3D visualization (Interior Design & Architects)
- graphic_design: Graphic design, illustration, branding (Graphic Designers)
- photography: Photo editing, RAW processing, color grading (Photographers)
- music_production: Audio editing, DAW software, music creation (Music Producers & Audio Engineers)
- corporate: Business/enterprise use, productivity, office work (Corporate Buyers)

OTHER AVAILABLE OPTIONS:
- profession: ["student", "gamer", "programmer", "engineer", "video_editor", "trader", "data_scientist", "interior_designer", "graphic_designer", "photographer", "music_producer", "corporate", "everyday_user"]
- budget: ["budget", "value", "premium", "flagship"]
  * budget: under $800
  * value: $800-$1500
  * premium: $1500-$2500
  * flagship: $2500+
- portability: "ultraportable" | "light" | "balanced" | "performance"
- screen_size: ["13 inches or smaller", "14 inches", "15-16 inches", "17+ inches"]

HOW TO MAP SOFTWARE/ACTIVITIES TO USE CASES:
Think about what the software/activity DOES and WHO uses it:

ENGINEERING & TECHNICAL:
- CAD software (AutoCAD, SolidWorks, Rhino, SketchUp, Fusion 360, CATIA) → engineering
- Engineering simulations (ANSYS, MATLAB, Simulink) → engineering
- Architecture software (Revit, ArchiCAD, SketchUp) → interior_design

CREATIVE WORK:
- Video editors (Premiere, DaVinci, Final Cut, After Effects) → video_editing
- Photo editors (Photoshop, Lightroom, Capture One) → photography
- Graphic design (Illustrator, InDesign, Figma, Canva) → graphic_design
- 3D rendering for interiors (3ds Max, V-Ray, Lumion) → interior_design
- Music/Audio (Ableton, FL Studio, Logic Pro, Pro Tools) → music_production

DEVELOPMENT & DATA:
- Code editors (VS Code, PyCharm, IntelliJ, Sublime) → programming
- Data science (Python, Jupyter, R, Pandas, TensorFlow) → data_science
- Trading platforms (Bloomberg Terminal, ThinkorSwim, TradingView) → trading

GENERAL USE:
- Games (Fortnite, Valorant, Cyberpunk, any game) → gaming
- Office apps (Word, Excel, PowerPoint, Google Docs) → corporate OR school (depends on context)
- Browsing, Netflix, YouTube, email → basic_use

CONTEXT-AWARE MAPPING (understand the USER):
- "Computer science student + AutoCAD" → ["school", "engineering"] (student doing CAD)
- "Mechanical engineering student + MATLAB" → ["school", "engineering"] (student + technical work)
- "CS student learning to code" → ["school", "programming"] (student + coding)
- "Graphic designer + Photoshop" → ["graphic_design"] (professional designer)
- "Photographer editing photos" → ["photography"] (photo professional)
- "YouTuber + Premiere Pro" → ["video_editing"] (content creator)
- "Data scientist + Python" → ["data_science"] (data professional)
- "Day trader" → ["trading"] (financial professional)
- "Interior designer + SketchUp" → ["interior_design"] (design professional)
- "Just browsing and Netflix" → ["basic_use"] (everyday user)

Return ONLY valid JSON.

STATEFUL REFINEMENT RULES:
If an 'existing_intent' is provided, your goal is to UPDATE it with NEW information from the 'latest_message'.
1. If the user changes their mind (e.g., "Actually, my budget is higher"), OVERWRITE the relevant field.
2. If the user adds new needs (e.g., "I also want to do video editing"), APPEND to the lists.
3. If the user clarifies (e.g., "I'm a college student"), refine the 'profession' field.
4. If the latest message is just a thank you or unrelated, return the 'existing_intent' UNCHANGED.

CRITICAL RULES:
1. ONLY use use_case values from the list above - NEVER create new ones
2. Think about WHO the user is and WHAT they want to accomplish
3. For students: always include "school" + any specialized use case (engineering, programming, data_science, etc.)
4. List use cases in order of importance (most critical first)
5. Match profession to use_case logically
6. Omit fields if not mentioned
7. Always include at least profession and use_case
8. Infer hardware needs based on use case
9. **BRAND PREFERENCES**:
   - Extract any brand preferences into "preferred_brands" array
   - Extract any brand EXCLUSIONS into "excluded_brands" array
   - Use lowercase brand names: "apple", "dell", "lenovo", "hp", "asus", "acer", "microsoft", "razer", "msi", "lg", "samsung", etc.
   - For "no Macs" or "no MacBooks" → excluded_brands: ["apple"]
   - For "only Lenovo" → preferred_brands: ["lenovo"]

EXAMPLE 1 (Initial Extraction - CS student, no Macs, prefers Lenovo):
USER: "I'm a computer science student. I need something for coding and maybe some light gaming. My budget is around $1200. I don't want a Mac, I prefer Lenovo."
OUTPUT:
{
  "profession": ["student"],
  "use_case": ["school", "programming", "gaming"],
  "budget": ["value"],
  "portability": "balanced",
  "screen_size": ["14 inches", "15-16 inches"],
  "preferred_brands": ["lenovo"],
  "excluded_brands": ["apple"],
  "extracted_requirements": {
    "min_ram": 16,
    "min_storage": 512,
    "needs_gpu": true,
    "needs_long_battery": true,
    "other_notes": "CS student who prefers Lenovo ThinkPads or Legions, explicitly excluded Apple"
  }
}

EXAMPLE 2 (Refinement - User changes mind):
EXISTING PROFILE: {"profession": ["student"], "use_case": ["school"], "budget": ["budget"]}
LATEST MESSAGE: "Actually, I decided to start video editing, so I can go up to $2000 for something powerful."
OUTPUT:
{
  "profession": ["student", "video_editor"],
  "use_case": ["school", "video_editing"],
  "budget": ["premium"],
  "portability": "performance",
  "extracted_requirements": {
    "needs_gpu": true,
    "min_ram": 16,
    "other_notes": "User pivoted from budget student needs to premium video editing requirements"
  }
}
"""
        
        user_content = user_prompt
        if existing_intent:
            user_content = f"Existing User Profile: {json.dumps(existing_intent)}\n\nLatest User Message: {user_prompt}\n\nPlease update the profile based on this message."

        try:
            # Call OpenAI API (v1.0+ syntax)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,  # Even lower for refinement
                max_tokens=500
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            if self.verbose:
                print(f"[*] LLM response:")
                print(f"    {content}")
            
            # Parse JSON
            # Sometimes LLM wraps JSON in markdown code blocks, so clean it
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            intent = json.loads(content)
            
            # Validate and clean the response
            intent = self._validate_intent(intent)
            
            if self.verbose:
                print(f"[+] Extracted intent:")
                print(f"    Profession: {intent.get('profession')}")
                print(f"    Use Case: {intent.get('use_case')}")
                print(f"    Budget: {intent.get('budget')}")
                print(f"    Portability: {intent.get('portability')}")
            
            return intent
            
        except json.JSONDecodeError as e:
            print(f"[-] Error parsing LLM response as JSON: {e}")
            print(f"    Response: {content}")
            raise ValueError(f"Failed to parse LLM response: {e}")
        
        except Exception as e:
            print(f"[-] Error calling OpenAI API: {e}")
            raise
    
    def _validate_intent(self, intent: Dict) -> Dict:
        """
        Validate and clean extracted intent.
        
        Args:
            intent: Raw intent from LLM
            
        Returns:
            Cleaned and validated intent
        """
        # Ensure lists are lists
        for field in ['profession', 'use_case', 'budget', 'screen_size']:
            if field in intent and not isinstance(intent[field], list):
                intent[field] = [intent[field]]
        
        # Ensure portability is a string
        if 'portability' in intent and isinstance(intent['portability'], list):
            intent['portability'] = intent['portability'][0]
        
        # Remove empty fields
        intent = {k: v for k, v in intent.items() if v}
        
        return intent
