
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

def check_use_case_table():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT * FROM use_cases;")
        rows = cur.fetchall()
        
        for row in rows:
            print(f"ID: {row['id']} - Name: {row.get('name') or row.get('title') or row}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    check_use_case_table()
