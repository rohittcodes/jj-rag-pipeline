"""
Production Database Client - Read-only access to production database.

Provides direct access to:
- Products and configs (with all properties and property groups)
- Test data and performance metrics
- Pricing, ratings, and classifications
- Property groups and property values

This client reads DIRECTLY from production database (read-only).
No data is synced or written locally. Local DB is only for RAG content.
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
    Client for accessing product and config data from PRODUCTION database.
    
    **IMPORTANT**: This client connects directly to the production database
    for real-time, accurate product data. It is READ-ONLY - no writes allowed.
    
    The local database is ONLY used for RAG-specific data:
    - josh_content, josh_chunks (blog content)
    - youtube_content, youtube_chunks (video transcripts)
    - test_data_chunks (PDF benchmark data)
    - sync_cursors (sync state tracking)
    - configs (synced product names for fast lookups)
    """
    
    def __init__(self):
        """Initialize production database client (READ-ONLY)."""
        # Local database (for synced product names - fast lookups)
        self.local_conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'josh_rag'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres')
        )
        
        # Production database (READ-ONLY access)
        prod_host = os.getenv('PROD_DB_HOST')
        
        if not prod_host:
            raise ValueError(
                "PROD_DB_HOST not configured. Please set production database credentials in .env:\n"
                "PROD_DB_HOST=your_rds_host.rds.amazonaws.com\n"
                "PROD_DB_PORT=5432\n"
                "PROD_DB_NAME=your_prod_db_name\n"
                "PROD_DB_USER=postgres\n"
                "PROD_DB_PASSWORD=your_prod_password"
            )
        
        self.db_config = {
            'host': prod_host,
            'port': os.getenv('PROD_DB_PORT', '5432'),
            'database': os.getenv('PROD_DB_NAME'),
            'user': os.getenv('PROD_DB_USER', 'postgres'),
            'password': os.getenv('PROD_DB_PASSWORD')
        }
    
    def get_product_name(self, config_id: int) -> Optional[str]:
        """
        Get product name for a config_id from LOCAL database (fast).
        Uses synced data from the configs table.
        """
        try:
            with self.local_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT product_name
                    FROM configs
                    WHERE config_id = %s
                """, (config_id,))
                
                result = cur.fetchone()
                return result['product_name'] if result else None
        except Exception as e:
            print(f"[!] Error fetching product name from local DB: {e}")
            return None
    
    def get_product_names_batch(self, config_ids: List[int]) -> Dict[int, str]:
        """
        Get product names for multiple config_ids from LOCAL database (fast batch fetch).
        Returns dict mapping config_id -> product_name.
        """
        if not config_ids:
            return {}
        
        try:
            with self.local_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT config_id, product_name
                    FROM configs
                    WHERE config_id = ANY(%s)
                """, (list(config_ids),))
                
                results = cur.fetchall()
                return {row['config_id']: row['product_name'] for row in results}
        except Exception as e:
            print(f"[!] Error fetching product names from local DB: {e}")
            return {}
    
    def get_config_by_id(self, config_id: int, include_properties: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetch config specifications by config_id from PRODUCTION database.
        
        Args:
            config_id: Configuration ID
            include_properties: If True, includes property groups and values
            
        Returns:
            Dictionary with config data, optionally including property groups
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query configs with product info
            cursor.execute("""
                SELECT 
                    c.id as config_id,
                    c.public_config_id,
                    c.product_id,
                    c.classification,
                    c.fallback_msrp as price,
                    c.final_rating,
                    c.model_year,
                    c.image,
                    c.sku,
                    c.upc,
                    c.testing_status,
                    c.updated_at,
                    p.title as product_name,
                    p.brand,
                    p.slug as product_slug,
                    p.description as product_description,
                    p.image as product_image,
                    p.yt_review_video_id,
                    p.test_data_pdf_url,
                    p.test_data_pdf_key
                FROM configs c
                JOIN products p ON c.product_id = p.id
                WHERE c.id = %s AND c.is_archived = false;
            """, (config_id,))
            
            config = cursor.fetchone()
            
            if not config:
                cursor.close()
                conn.close()
                return None
            
            result = dict(config)
            
            # Fetch property groups and values if requested
            if include_properties:
                result['property_groups'] = self._fetch_config_properties(cursor, config_id)
            
            cursor.close()
            conn.close()
            
            return result
            
        except Exception as e:
            print(f"[!] Error fetching config {config_id}: {e}")
            return None
    
    def _fetch_config_properties(self, cursor, config_id: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch property groups and values for a config.
        
        Returns:
            Dictionary mapping property group names to lists of property values
            Example: {
                "Performance": [
                    {"property": "Processor", "value": "Intel Core i7-13700H"},
                    {"property": "RAM", "value": "16GB DDR5"}
                ],
                "Display": [
                    {"property": "Screen Size", "value": "15.6 inches"},
                    {"property": "Resolution", "value": "1920x1080"}
                ]
            }
        """
        cursor.execute("""
            SELECT 
                ptpg.name as group_name,
                ptp.title as property_name,
                ptpv.value as property_value,
                ptpg.id as group_id,
                ptp.id as property_id
            FROM config_properties cp
            JOIN product_type_property_values ptpv ON cp.value_id = ptpv.id
            JOIN product_type_properties ptp ON ptpv.product_type_property_id = ptp.id
            JOIN product_type_property_groups ptpg ON ptp.group_id = ptpg.id
            WHERE cp.config_id = %s
            ORDER BY ptpg.id, ptp.id;
        """, (config_id,))
        
        rows = cursor.fetchall()
        
        # Group by property group name
        groups = {}
        for row in rows:
            group_name = row['group_name']
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append({
                'property': row['property_name'],
                'value': row['property_value']
            })
        
        return groups
    
    # Alias for backward compatibility
    def get_product_by_config_id(self, config_id: int) -> Optional[Dict[str, Any]]:
        """Alias for get_config_by_id for backward compatibility."""
        return self.get_config_by_id(config_id)
    
    def get_configs_by_ids(self, config_ids: List[int], include_properties: bool = False, include_links: bool = False, location_id: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
        """
        Fetch multiple configs by config_ids (batch query) from PRODUCTION database.
        
        Args:
            config_ids: List of configuration IDs
            include_properties: If True, includes property groups and values
            include_links: If True, includes affiliate links and store info
            location_id: Optional location ID to filter affiliate links
            
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
                    c.id as config_id,
                    c.public_config_id,
                    c.product_id,
                    c.classification,
                    c.fallback_msrp as price,
                    c.final_rating,
                    c.model_year,
                    c.image as config_image,
                    c.sku,
                    c.upc,
                    c.testing_status,
                    c.updated_at,
                    p.title as product_name,
                    p.brand,
                    p.slug as product_slug,
                    p.description as product_description,
                    p.image as product_image,
                    p.yt_review_video_id,
                    p.test_data_pdf_url,
                    p.test_data_pdf_key
                FROM configs c
                JOIN products p ON c.product_id = p.id
                WHERE c.id = ANY(%s) AND c.is_archived = false;
            """, (config_ids,))
            
            configs = cursor.fetchall()
            result = {c['config_id']: dict(c) for c in configs}
            
            # Fetch properties for all configs if requested
            if include_properties:
                for config_id in result.keys():
                    result[config_id]['property_groups'] = self._fetch_config_properties(cursor, config_id)
            
            # Fetch affiliate links for all configs if requested
            if include_links:
                for config_id in result.keys():
                    result[config_id]['affiliate_links'] = self._fetch_affiliate_links(cursor, config_id, location_id)
            
            cursor.close()
            conn.close()
            
            return result
            
        except Exception as e:
            print(f"[!] Error fetching configs: {e}")
            return {}

    def _fetch_affiliate_links(self, cursor, config_id: int, location_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch affiliate links and store info for a config."""
        query = """
            SELECT 
                al.id,
                al.url,
                al.current_price,
                al.msrp,
                al.out_of_stock,
                s.name as store_name,
                s.logo as store_logo
            FROM affiliate_links_v2 al
            JOIN affiliate_programs ap ON al.affiliate_program_id = ap.id
            JOIN stores s ON ap.store_id = s.id
            WHERE al.config_id = %s 
              AND al.is_archived = false 
              AND (al.out_of_stock = false OR al.out_of_stock IS NULL)
        """
        params = [config_id]
        
        if location_id is not None:
            query += " AND ap.location_id = %s"
            params.append(location_id)
            
        query += " ORDER BY al.current_price ASC;"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    # Alias for backward compatibility
    def get_products_by_config_ids(self, config_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Alias for get_configs_by_ids for backward compatibility."""
        return self.get_configs_by_ids(config_ids)
    
    def get_config_use_cases(self, config_ids: List[int]) -> Dict[int, List[int]]:
        """
        Fetch manual use case tags for multiple configs from PRODUCTION database.
        
        Args:
            config_ids: List of configuration IDs
            
        Returns:
            Dictionary mapping config_id -> list of use_case_ids
        """
        if not config_ids:
            return {}
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT config_id, use_case_id 
                FROM config_use_case_relation 
                WHERE config_id = ANY(%s);
            """, (list(config_ids),))
            
            rows = cursor.fetchall()
            
            result = {}
            for row in rows:
                cid = row['config_id']
                uid = row['use_case_id']
                if cid not in result:
                    result[cid] = []
                result[cid].append(uid)
                
            cursor.close()
            conn.close()
            
            return result
            
        except Exception as e:
            print(f"[!] Error fetching config use cases: {e}")
            return {}
    
    # Known leading brands in content; DB often stores "Model Brand" so we try "Model" as fallback
    _LEADING_BRANDS = frozenset(
        {"dell", "asus", "hp", "lenovo", "acer", "msi", "razer", "lg", "microsoft", "gigabyte", "eluktronics"}
    )

    def search_configs_by_name(self, product_name: str) -> List[Dict[str, Any]]:
        """
        Find all configs whose product title contains the given string (case-insensitive).
        Searches in PRODUCTION database.
        """
        if not product_name or not product_name.strip():
            return []
        name = product_name.strip()
        # Try full name first, then model-only (content says "Dell XPS 13", DB has "XPS 13")
        to_try = [name]
        parts = name.split(None, 1)
        if len(parts) == 2 and parts[0].lower() in self._LEADING_BRANDS:
            model_part = parts[1]
            to_try.append(model_part)
            # If model part is long, try first 2 words
            model_words = model_part.split()
            if len(model_words) > 2:
                to_try.append(" ".join(model_words[:2]))
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            for query in to_try:
                pattern = f"%{query}%"
                cursor.execute("""
                    SELECT 
                        c.id as config_id,
                        c.public_config_id,
                        c.product_id,
                        c.classification,
                        c.fallback_msrp as price,
                        c.final_rating,
                        p.title as product_name,
                        p.brand,
                        p.slug as product_slug
                    FROM configs c
                    JOIN products p ON c.product_id = p.id
                    WHERE p.title ILIKE %s AND c.is_archived = false
                    ORDER BY c.id;
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
        """
        Score how well config matches user's budget, use case, portability.
        Now uses property_groups if available, falls back to legacy specs field.
        """
        parts = []

        price = config.get("price")
        if price is not None:
            try:
                price = float(price)
            except (TypeError, ValueError):
                price = None

        # Extract specs from property_groups or legacy specs field
        specs = self._extract_specs_from_config(config)

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
            has_gpu = specs.get("has_gpu", False)
            ram_num = specs.get("ram_gb", 0)
            cpu = specs.get("processor", "").lower()
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
            weight = specs.get("weight_lbs", 0)
            screen = specs.get("screen_size", 0)
            if portability == "light" and weight > 0 and weight < 4.0:
                parts.append(1.0)
            elif portability == "performance" and screen >= 15:
                parts.append(1.0)
            elif portability == "somewhat" and 14 <= screen <= 16:
                parts.append(1.0)
            else:
                parts.append(0.4)

        return sum(parts) / len(parts) if parts else 0.5
    
    def _extract_specs_from_config(self, config: Dict) -> Dict[str, Any]:
        """
        Extract normalized specs from config (either from property_groups or legacy specs).
        
        Returns:
            Dictionary with normalized keys:
            - has_gpu: bool
            - ram_gb: float
            - processor: str
            - weight_lbs: float
            - screen_size: float
        """
        result = {
            "has_gpu": False,
            "ram_gb": 0,
            "processor": "",
            "weight_lbs": 0,
            "screen_size": 0
        }
        
        # Try property_groups first (new schema)
        if "property_groups" in config:
            for group_name, properties in config["property_groups"].items():
                for prop in properties:
                    prop_name = prop["property"].lower()
                    prop_value = str(prop["value"]).lower()
                    
                    if "gpu" in prop_name or "graphics" in prop_name:
                        result["has_gpu"] = "yes" in prop_value or "dedicated" in prop_value
                    elif "ram" in prop_name or "memory" in prop_name:
                        result["ram_gb"] = self._extract_number(prop_value)
                    elif "processor" in prop_name or "cpu" in prop_name:
                        result["processor"] = prop_value
                    elif "weight" in prop_name:
                        result["weight_lbs"] = self._extract_number(prop_value)
                    elif "screen" in prop_name or "display" in prop_name:
                        result["screen_size"] = self._extract_number(prop_value)
        
        # Fall back to legacy specs field if available
        elif "specs" in config and config["specs"]:
            specs = config["specs"]
            if isinstance(specs, dict):
                result["has_gpu"] = (specs.get("Dedicated Graphics (Yes/No)") or "").strip().lower() == "yes"
                result["ram_gb"] = self._extract_number(specs.get("Memory Amount", ""))
                result["processor"] = (specs.get("Processor") or "").lower()
                result["weight_lbs"] = self._extract_number(specs.get("Weight (lbs)", ""))
                result["screen_size"] = self._extract_number(specs.get("Display Size", ""))
        
        return result

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
        Test PRODUCTION database connection (READ-ONLY).
        
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
            
            print("[+] Production database connection successful (READ-ONLY)")
            return result[0] == 1
            
        except Exception as e:
            print(f"[!] Production database connection failed: {e}")
            print("    Please check PROD_DB_* credentials in .env")
            return False
    
    def get_config_count(self) -> int:
        """
        Get total number of active (non-archived) configs in PRODUCTION database.
        
        Returns:
            Config count
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM configs WHERE is_archived = false;")
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
    """Test the production database client."""
    
    print("="*80)
    print("Testing Production Database Client (READ-ONLY)")
    print("="*80)
    
    # Test production connection
    print("\n[*] Testing PRODUCTION database connection...")
    try:
        client = ConfigDatabaseClient()
        
        if client.test_connection():
            count = client.get_config_count()
            print(f"[+] Active configs in production: {count}")
            
            if count > 0:
                # Get first available config ID
                conn = psycopg2.connect(**client.db_config)
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM configs WHERE is_archived = false ORDER BY id LIMIT 1;")
                first_config_id = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                print(f"\n[*] Sample config (config_id={first_config_id}):")
                config = client.get_config_by_id(first_config_id, include_properties=False)
                if config:
                    print(f"    - Name: {config.get('product_name')}")
                    print(f"    - Brand: {config.get('brand')}")
                    print(f"    - Price: ${config.get('price')}")
                    print(f"    - Rating: {config.get('final_rating')}")
                    print(f"    - Classification: {config.get('classification')}")
                    print(f"    - Testing Status: {config.get('testing_status')}")
                else:
                    print(f"    [!] Config {first_config_id} not found")
                
                print(f"\n[*] Sample config WITH property groups (config_id={first_config_id}):")
                config_with_props = client.get_config_by_id(first_config_id, include_properties=True)
                if config_with_props and 'property_groups' in config_with_props:
                    print(f"    Property Groups:")
                    for group_name, properties in config_with_props['property_groups'].items():
                        print(f"      {group_name}:")
                        for prop in properties[:3]:  # Show first 3 properties per group
                            print(f"        - {prop['property']}: {prop['value']}")
                else:
                    print(f"    [!] No property groups found for config {first_config_id}")
            else:
                print("\n[!] No configs found in production database")
        else:
            print("[-] Connection failed")
            print("    Please configure PROD_DB_* credentials in .env")
    
    except ValueError as e:
        print(f"[-] Configuration error: {e}")
    
    print("\n" + "="*80)
