import requests
import json
import time
import sys

def test_streaming_api(prompt_index=0):
    url = "http://localhost:8000/stream-rag"
    
    prompts = [
        # 1. OLED vs IPS
        "I have seen many posts talking about how oled screens is hard on the eye for long hours and its text clarity is worse than IPS, also staring at some static texts all day is generally not good for oled. I found some really good laptops that meet my requirements, but almost all of them have oled :frowning:",
        
        # 2. Intel Core Ultra 7 vs Ultra 9 (HP OmniBook)
        "Any thoughts on the Intel Core Ultra 7 258V vs the Intel Core Ultra 9 288V? Specifically in the 14” HP OmniBook Ultra Flip 2-in-1. Those are the 2 highest CPU options, and the only 2 options with 32 GB of RAM. I will be using it as a personal and work laptop. Mostly for heavy web-based productivity tasks, multitasking, and browsing. It will also spend much of its time “docked” to an external 4k monitor. Is the higher clock speed/extra performance of the Ultra 9 worth it? I really want a quiet laptop with very little excessive heat and decent battery life, so I am leaning more towards the Ultra 7.",
        
        # 3. ThinkPad T vs P (Gen 7/8)
        "I am an College student. I found the Thinkpad T1g or P1. Currently there are many options within the lenovo website which confused me. Considerations: Lenovo ThinkPad T1g Gen 8 (5070 and 3k screen), Lenovo ThinkPad P1 Gen 8 (RTX PRO 2000 Blackwell and 1920x1200), Lenovo ThinkPad P1 Gen 7 (RTX PRO 2000 Blackwell and 2560x1600). Standard studying, assignments, and light gaming.",
        
        # 4. Budget 1.5 Lakh INR (ThinkPad only)
        "My budget is 1.5 Lakh Inr ($1,690) , so i need a laptop with good battery life and decent performance, with good build quality and display, no macs please. I am choosing options from lenevo only, because i love thinkpads."
    ]
    
    payload = {
        "prompt": prompts[prompt_index],
        "top_k": 5
    }
    
    print(f"\n[*] Testing Prompt {prompt_index + 1}:")
    print(f"[*] Prompt text: {prompts[prompt_index][:100]}...")
    print("-" * 50)
    
    headers = {
        "Authorization": "Bearer testing_is_fun"
    }
    
    try:
        start_time = time.time()
        with requests.post(url, json=payload, stream=True, headers=headers) as response:
            if response.status_code != 200:
                print(f"[!] Error: {response.status_code} - {response.text}")
                return
            
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    text_chunk = chunk.decode('utf-8')
                    if "__JSON_DATA__" in text_chunk:
                        parts = text_chunk.split("__JSON_DATA__")
                        print(parts[0], end="", flush=True)
                        try:
                            data = json.loads(parts[1])
                            if "sources" in data:
                                print("\n\n--- CITED SOURCES (for UI/Debug) ---")
                                for i, src in enumerate(data["sources"], 1):
                                    snippet = src.get("text", "")[:100] + "..." if len(src.get("text", "")) > 100 else src.get("text", "")
                                    print(f"[Source {i}] {src['title']} ({src['type']})")
                                    print(f"    Content: {snippet}")
                        except:
                            pass
                    elif "__JSON_ERROR__" in text_chunk:
                        print(f"\n[!] Stream Error: {text_chunk}")
                    else:
                        print(text_chunk, end="", flush=True)
            
            print(f"\n\n[*] Total Time: {time.time() - start_time:.2f}s")
                
    except Exception as e:
        print(f"\n[!] Error: {e}")

if __name__ == "__main__":
    # Test only the two most relevant prompts
    for i in [0, 1]:
        test_streaming_api(i)
        print("\n" + "="*80 + "\n")
