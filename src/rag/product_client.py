"""
Config Database Client - Read-only access to production config database.

Provides access to product configurations with:
- Specs (CPU, GPU, RAM, storage, etc.)
- Test data (performance metrics)
- Pricing and ratings
"""
import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

load_dotenv()


class ConfigDatabaseClient:
    """
    Client for accessing config specifications from local database.
    
    Uses locally synced config data for fast, offline access.
    Production sync is handled by scripts/sync_products.py.
    """
    
    def __init__(self):
        """Initialize config database client (uses local synced copy)."""
        # Local database (synced copy)
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'josh_rag'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
    
    def get_config_by_id(self, config_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch config specifications by config_id.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            Dictionary with config specs, test data, and metadata or None if not found
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query configs table
            cursor.execute("""
                SELECT 
                    config_id,
                    public_config_id,
                    product_id,
                    product_name,
                    brand,
                    model,
                    specs,
                    test_data,
                    price,
                    final_rating,
                    classification,
                    model_year,
                    image,
                    sku,
                    updated_at
                FROM configs
                WHERE config_id = %s;
            """, (config_id,))
            
            config = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return dict(config) if config else None
            
        except Exception as e:
            print(f"[!] Error fetching config {config_id}: {e}")
            return None
    
    # Alias for backward compatibility
    def get_product_by_config_id(self, config_id: int) -> Optional[Dict[str, Any]]:
        """Alias for get_config_by_id for backward compatibility."""
        return self.get_config_by_id(config_id)
    
    def get_configs_by_ids(self, config_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Fetch multiple configs by config_ids (batch query).
        
        Args:
            config_ids: List of configuration IDs
            
        Returns:
            Dictionary mapping config_id -> config data
        """
        if not config_ids:
            return {}
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    config_id,
                    public_config_id,
                    product_id,
                    product_name,
                    brand,
                    model,
                    specs,
                    test_data,
                    price,
                    final_rating,
                    classification,
                    model_year,
                    image,
                    sku,
                    updated_at
                FROM configs
                WHERE config_id = ANY(%s);
            """, (config_ids,))
            
            configs = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {c['config_id']: dict(c) for c in configs}
            
        except Exception as e:
            print(f"[!] Error fetching configs: {e}")
            return {}
    
    # Alias for backward compatibility
    def get_products_by_config_ids(self, config_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Alias for get_configs_by_ids for backward compatibility."""
        return self.get_configs_by_ids(config_ids)
    
    # Known leading brands in content; DB often stores "Model Brand" so we try "Model" as fallback
    _LEADING_BRANDS = frozenset(
        {"dell", "asus", "hp", "lenovo", "acer", "msi", "razer", "lg", "microsoft", "gigabyte", "eluktronics"}
    )

    def search_configs_by_name(self, product_name: str) -> List[Dict[str, Any]]:
        """
        Find all configs whose product_name contains the given string (case-insensitive).
        Tries exact phrase first; if no match, tries without leading brand (DB often has "Model Brand").
        """
        if not product_name or not product_name.strip():
            return []
        name = product_name.strip()
        # Try full name first, then model-only (content says "Dell XPS 13", DB has "XPS 13 Dell")
        to_try = [name]
        parts = name.split(None, 1)
        if len(parts) == 2 and parts[0].lower() in self._LEADING_BRANDS:
            model_part = parts[1]
            to_try.append(model_part)
            # If model part is long (e.g. "TUF F16 sometimes goes on sale..."), try first 2 words
            model_words = model_part.split()
            if len(model_words) > 2:
                to_try.append(" ".join(model_words[:2]))
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            for query in to_try:
                pattern = f"%{query}%"
                cursor.execute("""
                    SELECT config_id, product_name, brand, model, specs, price, final_rating
                    FROM configs
                    WHERE product_name ILIKE %s
                    ORDER BY config_id;
                """, (pattern,))
                rows = cursor.fetchall()
                if rows:
                    cursor.close()
                    conn.close()
                    return [dict(r) for r in rows]
            cursor.close()
            conn.close()
            return []
        except Exception as e:
            print(f"[!] Error searching configs by name: {e}")
            return []

    def find_best_config_for_product(
        self,
        product_name: str,
        quiz_response: Dict,
        chunk_context: Optional[str] = None
    ) -> Optional[int]:
        """
        Find best matching config_id for a product name when multiple configs exist.

        Strategy:
        1. Query all configs matching the product name.
        2. If none or one, return that result.
        3. Otherwise score each config: Josh's context (0.5), user requirements (0.3), rating (0.2).
        4. Return config_id with highest score.
        """
        candidates = self.search_configs_by_name(product_name)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]["config_id"]

        josh_weight = 0.5
        user_weight = 0.3
        rating_weight = 0.2

        best_id = None
        best_score = -1.0

        for config in candidates:
            josh_score = self._score_josh_context(config, chunk_context)
            user_score = self._score_user_requirements(config, quiz_response)
            rating_score = self._score_rating(config)
            total = josh_weight * josh_score + user_weight * user_score + rating_weight * rating_score
            if total > best_score:
                best_score = total
                best_id = config["config_id"]

        return best_id

    def _score_josh_context(self, config: Dict, chunk_context: Optional[str]) -> float:
        """Score how well config matches what Josh mentioned in the chunk (e.g. price, specs)."""
        if not chunk_context:
            return 0.5
        score = 0.0
        config_price = config.get("price")
        if config_price is not None:
            try:
                price = float(config_price)
            except (TypeError, ValueError):
                price = None
        else:
            price = None

        if price is not None:
            # Extract dollar amounts from chunk (e.g. $1,999 or $1999)
            mentioned = re.findall(r"\$[\d,]+(?:\.[\d]+)?", chunk_context)
            for m in mentioned:
                try:
                    val = float(m.replace("$", "").replace(",", ""))
                    if val > 0:
                        if abs(val - price) / max(val, 1) <= 0.2:
                            return 1.0
                        if abs(val - price) / max(val, 1) <= 0.35:
                            score = max(score, 0.7)
                except ValueError:
                    pass

        if not score and price is not None:
            # No explicit price match; check "under $X" or "around $X"
            under = re.search(r"under\s+\$[\d,]+", chunk_context, re.IGNORECASE)
            if under:
                score = max(score, 0.5)
        return score if score > 0 else 0.5

    def _score_user_requirements(self, config: Dict, quiz_response: Dict) -> float:
        """Score how well config matches user's budget, use case, portability."""
        parts = []

        price = config.get("price")
        if price is not None:
            try:
                price = float(price)
            except (TypeError, ValueError):
                price = None
        specs = config.get("specs") or {}

        budgets = quiz_response.get("budget", [])
        if budgets and price is not None:
            if price < 800 and "budget" in budgets:
                parts.append(1.0)
            elif 800 <= price < 1500 and "value" in budgets:
                parts.append(1.0)
            elif price >= 1500 and "premium" in budgets:
                parts.append(1.0)
            else:
                parts.append(0.3)

        use_cases = quiz_response.get("use_case", [])
        if use_cases:
            has_gpu = (specs.get("Dedicated Graphics (Yes/No)") or "").strip().lower() == "yes"
            ram_str = specs.get("Memory Amount", "") or ""
            ram_num = self._extract_number(ram_str)
            cpu = (specs.get("Processor") or "").lower()
            uc_score = 0.0
            for uc in use_cases:
                if uc == "gaming" and has_gpu:
                    uc_score = 1.0
                    break
                if uc == "programming" and (ram_num >= 8 or "m3" in cpu or "i7" in cpu or "ryzen" in cpu):
                    uc_score = max(uc_score, 1.0 if ram_num >= 16 else 0.7)
                if uc == "video_editing" and (has_gpu or ram_num >= 16):
                    uc_score = max(uc_score, 1.0 if (has_gpu and ram_num >= 16) else 0.7)
            parts.append(uc_score if uc_score > 0 else 0.4)

        portability = quiz_response.get("portability", "")
        if portability:
            weight_str = specs.get("Weight (lbs)", "") or ""
            screen_str = specs.get("Display Size", "") or ""
            weight = self._extract_number(weight_str)
            screen = self._extract_number(screen_str)
            if portability == "light" and weight > 0 and weight < 4.0:
                parts.append(1.0)
            elif portability == "performance" and screen >= 15:
                parts.append(1.0)
            elif portability == "somewhat" and 14 <= screen <= 16:
                parts.append(1.0)
            else:
                parts.append(0.4)

        return sum(parts) / len(parts) if parts else 0.5

    def _score_rating(self, config: Dict) -> float:
        """Normalize final_rating to 0-1 for tiebreaker."""
        rating = config.get("final_rating")
        if rating is None:
            return 0.5
        try:
            r = float(rating)
            if r <= 0:
                return 0.5
            if r <= 5:
                return min(1.0, r / 5.0)
            return min(1.0, r / 10.0)
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _extract_number(text: str) -> float:
        """Extract first number from text."""
        if not text:
            return 0.0
        match = re.search(r"(\d+(?:\.\d+)?)", str(text))
        return float(match.group(1)) if match else 0.0

    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection successful
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return result[0] == 1
            
        except Exception as e:
            print(f"[!] Connection test failed: {e}")
            return False
    
    def get_config_count(self) -> int:
        """
        Get total number of configs in database.
        
        Returns:
            Config count
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM configs;")
            count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return count
            
        except Exception as e:
            print(f"[!] Error getting config count: {e}")
            return 0
    
    # Alias for backward compatibility
    def get_product_count(self) -> int:
        """Alias for get_config_count for backward compatibility."""
        return self.get_config_count()


if __name__ == "__main__":
    """Test the config client."""
    
    print("="*80)
    print("Testing Config Database Client")
    print("="*80)
    
    # Test local connection
    print("\n[*] Testing local database connection...")
    client = ConfigDatabaseClient()
    
    if client.test_connection():
        print("[+] Connection successful")
        count = client.get_config_count()
        print(f"[+] Configs in database: {count}")
        
        if count > 0:
            print("\n[*] Sample config (config_id=1):")
            config = client.get_config_by_id(1)
            if config:
                print(f"    - Name: {config.get('product_name')}")
                print(f"    - Brand: {config.get('brand')}")
                print(f"    - Price: ${config.get('price')}")
                print(f"    - Rating: {config.get('final_rating')}")
                print(f"    - Specs: {list(config.get('specs', {}).keys()) if config.get('specs') else 'None'}")
                print(f"    - Test Data: {list(config.get('test_data', {}).keys()) if config.get('test_data') else 'None'}")
        else:
            print("\n[!] No configs found. Run sync:")
            print("    uv run python scripts/sync_products.py --full")
    else:
        print("[-] Connection failed")
        print("    Check database is running: docker ps")
    
    print("\n" + "="*80)
