
import os
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

def check_structure():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # 1. Get Column Names for 'configs'
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'configs';")
        columns = [row['column_name'] for row in cur.fetchall()]
        
        with open('debug_structure.log', 'w', encoding='utf-8') as f:
            f.write(f"Columns in 'configs' table: {columns}\n\n")
            
            if 'use_case' in columns:
                f.write("[!] Found 'use_case' column!\n")
            else:
                f.write("[-] No 'use_case' column found in 'configs'.\n")

            # 2. Check sample data
            f.write("\nSample Row Data (limit 1):\n")
            cur.execute("SELECT * FROM configs LIMIT 1")
            row = cur.fetchone()
            if row:
                for k, v in row.items():
                    if k == 'specs':
                        f.write(f"  {k}: <JSONB with keys: {list(v.keys()) if v else 'None'}>\n")
                    else:
                        f.write(f"  {k}: {v}\n")
            
    except Exception as e:
        with open('debug_structure.log', 'a') as f:
            f.write(f"Error: {e}\n")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    check_structure()
