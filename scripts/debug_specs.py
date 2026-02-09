
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'josh_rag'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}


def check_specs(config_ids):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        placeholders = ', '.join(['%s'] * len(config_ids))
        query = f"SELECT config_id, product_name, specs FROM configs WHERE config_id IN ({placeholders})"
        
        with open('debug_results.log', 'w', encoding='utf-8') as f:
            f.write(f"Querying for Config IDs: {config_ids}\n")
            cur.execute(query, config_ids)
            rows = cur.fetchall()
            
            found_ids = set()
            for row in rows:
                found_ids.add(row['config_id'])
                f.write(f"\n[Config ID: {row['config_id']}] {row['product_name']}\n")
                specs = row.get('specs')
                
                if not specs:
                    f.write("  ! specs column is None or empty\n")
                    continue
                    
                if isinstance(specs, str):
                    try:
                        specs = json.loads(specs)
                    except:
                        f.write(f"  ! Failed to parse spec string: {specs[:50]}...\n")
                
                # Check for Graphics keys
                gpu_keys = [k for k in specs.keys() if 'graphic' in k.lower() or 'gpu' in k.lower()]
                f.write(f"  Graphics/GPU Keys found: {gpu_keys}\n")
                
                for k in gpu_keys:
                    f.write(f"    - {k}: {specs[k]}\n")

                # Specific check for the key used in ranker.py
                target_key = 'Dedicated Graphics (Yes/No)'
                if target_key in specs:
                     f.write(f"  -> EXACT MATCH '{target_key}': '{specs[target_key]}'\n")
                else:
                     f.write(f"  -> MISSING '{target_key}'\n")

            missing = set(config_ids) - found_ids
            if missing:
                f.write(f"\n[!] WARNING: Config IDs not found in DB: {missing}\n")
            else:
                 f.write(f"\n[+] All Config IDs found in DB.\n")

        cur.close()
        conn.close()

    except Exception as e:
        with open('debug_results.log', 'a', encoding='utf-8') as f:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    # IDs from the failed RAG test
    # Blade 16 Razer (526), Legion 7i 16 Lenovo (593), ROG Zephyrus G16 Asus (337)
    check_ids = [526, 593, 337]
    check_specs(check_ids)
