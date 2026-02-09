
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('PROD_DB_HOST'),
    'port': os.getenv('PROD_DB_PORT', '5432'),
    'database': os.getenv('PROD_DB_NAME'),
    'user': os.getenv('PROD_DB_USER', 'postgres'),
    'password': os.getenv('PROD_DB_PASSWORD')
}

if not DB_CONFIG['host']:
    print("[-] Error: PROD_DB_HOST not set. Please check your .env file.")
    exit(1)

def check_use_cases():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if table exists
        cur.execute("SELECT to_regclass('public.config_use_case_relation');")
        exists = cur.fetchone()['to_regclass']
        
        with open('debug_use_cases.log', 'w', encoding='utf-8') as f:
            if exists:
                f.write("[+] Table 'config_use_case_relation' EXISTS.\n")
                
                # Check count
                cur.execute("SELECT count(*) as count FROM config_use_case_relation;")
                count = cur.fetchone()['count']
                f.write(f"    - Row Count: {count}\n")
                
                if count > 0:
                    cur.execute("SELECT * FROM config_use_case_relation LIMIT 5;")
                    rows = cur.fetchall()
                    f.write("\n    - Sample Data:\n")
                    for row in rows:
                        f.write(f"      {row}\n")
            else:
                f.write("[-] Table 'config_use_case_relation' does NOT exist.\n")

    except Exception as e:
        with open('debug_use_cases.log', 'a', encoding='utf-8') as f:
            f.write(f"Error: {e}\n")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    check_use_cases()
