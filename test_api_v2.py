
import urllib.request
import urllib.parse
import json
import sys

def post_json(url, data):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def test_compare():
    base_url = 'http://127.0.0.1:5000/api'
    
    print("Loading sample data...")
    try:
        post_json(f'{base_url}/load-sample', {})
    except Exception as e:
        print(f"Failed to load sample: {e}")
    
    print("Comparing algorithms...")
    payload = {
        'query': 'machine learning',
        'top_k': 10
    }
    
    try:
        data = post_json(f'{base_url}/compare-algorithms', payload)
        print(json.dumps(data, indent=2))
        
        if data.get('success'):
            print("\nVerification:")
            comps = data.get('comparisons', [])
            best = data.get('best', {})
            
            for comp in comps:
                name = comp['algorithm']
                print(f"Algorithm: {name}")
                print(f"  Precision: {comp['precision']}")
                print(f"  Speed: {comp.get('search_time_ms')}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_compare()
