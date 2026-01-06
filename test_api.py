
import requests
import json

def test_compare():
    url = 'http://127.0.0.1:5000/api/compare-algorithms'
    
    # First ensure we have data
    requests.post('http://127.0.0.1:5000/api/load-sample')
    
    payload = {
        'query': 'machine learning',
        'top_k': 10
    }
    
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        print(json.dumps(data, indent=2))
        
        if data.get('success'):
            print("\nVerification:")
            comps = data.get('comparisons', [])
            best = data.get('best', {})
            
            for comp in comps:
                name = comp['algorithm']
                print(f"Algorithm: {name}")
                print(f"  Precision: {comp['precision']}")
                
                if best.get('precision') == name:
                    print(f"  MATCHES BEST PRECISION")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_compare()
