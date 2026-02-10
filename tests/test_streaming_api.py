import requests
import json
import time
import sys

def test_streaming_api():
    url = "http://localhost:8000/stream-rag"
    
    payload = {
        "prompt": "I am a computer science student looking for a laptop for coding and some light gaming like Cyberpunk 2077. My budget is around $1200.",
        "top_k": 5
    }
    
    print(f"[*] Testing Streaming API: {url}")
    print(f"[*] Payload: {json.dumps(payload, indent=2)}")
    print("-" * 50)
    
    headers = {
        "Authorization": "Bearer testing_is_fun"
    }
    
    try:
        start_time = time.time()
        # Stream=True is important here
        with requests.post(url, json=payload, stream=True, headers=headers) as response:
            if response.status_code != 200:
                print(f"[!] Error: {response.status_code} - {response.text}")
                return
            
            print("[*] Stream started...")
            full_text = ""
            json_data = None
            
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    text_chunk = chunk.decode('utf-8')
                    
                    # Check for JSON block
                    if "__JSON_DATA__" in text_chunk:
                        parts = text_chunk.split("__JSON_DATA__")
                        print(parts[0], end="", flush=True)
                        full_text += parts[0]
                        
                        try:
                            json_str = parts[1]
                            json_data = json.loads(json_str)
                            print("\n\n[+] Received JSON Data Block!")
                        except Exception as e:
                            print(f"\n[!] Error parsing JSON data: {e}")
                    elif "__JSON_ERROR__" in text_chunk:
                        print(f"\n[!] Stream Error: {text_chunk}")
                    else:
                        print(text_chunk, end="", flush=True)
                        full_text += text_chunk
            
            total_time = time.time() - start_time
            print(f"\n\n[*] Stream finished in {total_time:.2f}s")
            
            if json_data:
                print("-" * 50)
                print("JSON DATA ANALYSIS:")
                recs = json_data.get('recommendations', [])
                print(f"[+] Recommendations: {len(recs)}")
                for i, rec in enumerate(recs, 1):
                    print(f"  {i}. {rec.get('product_name')}")
                    print(f"     Price: {rec.get('price')}")
                    print(f"     Image: {rec.get('image_url')}")
                    print(f"     Link: {rec.get('product_link')}")
                
                proc_time = json_data.get('processing_time', 0)
                print(f"[+] Server Processing Time: {proc_time:.2f}s")
            else:
                print("\n[!] No JSON data received at end of stream")
                
    except Exception as e:
        print(f"\n[!] Connection Error: {e}")
        print("Make sure the API server is running! (uv run src/api/main.py)")

if __name__ == "__main__":
    test_streaming_api()
