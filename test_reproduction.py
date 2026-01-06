import requests
import time
import os

BASE_URL = 'http://127.0.0.1:5000/api'

def test_external_file_upload_and_search():
    print("Testing external file upload...")
    
    # Create a dummy file
    filename = 'test_external_doc.txt'
    unique_term = 'antigravity_unique_term_xyz'
    content = f"This is an external document containing the unique term {unique_term} for testing."
    
    with open(filename, 'w') as f:
        f.write(content)
        
    try:
        # Upload
        with open(filename, 'rb') as f:
            files = {'files': (filename, f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            
        print("Upload response:", response.json())
        if not response.json().get('success'):
            print("FAILED: Upload failed")
            return

        # Search
        print(f"Searching for {unique_term}...")
        response = requests.post(f"{BASE_URL}/search", json={
            'query': unique_term,
            'top_k': 10
        })
        data = response.json()
        print("Search response:", data)
        
        results = data.get('results', [])
        found = any(r['title'] == 'test_external_doc' for r in results)
        
        if found:
            print("SUCCESS: Found uploaded external document.")
            doc_id = results[0]['doc_id']
            
            # Test metrics update
            print("Testing metrics update...")
            # Mark relevant
            requests.post(f"{BASE_URL}/set-relevance", json={
                'query': unique_term,
                'relevant_ids': [doc_id]
            })
            
            # Search again to see metrics
            response = requests.post(f"{BASE_URL}/search-with-evaluation", json={
                'query': unique_term,
                'relevant_ids': [doc_id] # Simulating what the frontend sends after sync
            })
            metrics = response.json().get('evaluation', {})
            print("Metrics:", metrics)
            
            if metrics.get('precision', 0) > 0:
                print("SUCCESS: Metrics updated correctly.")
            else:
                print("FAILED: Metrics still 0.0%")
                
        else:
            print("FAILED: Could not find uploaded document.")

    finally:
        if os.path.exists(filename):
            os.remove(filename)

def test_top_k():
    print("\nTesting top_k functionality...")
    # Using 'learning' which we know has many results in sample data
    # (Assuming sample data is loaded)
    requests.post(f"{BASE_URL}/upload", json={}) # Ensure data? No, need to trigger sample load if empty
    
    # Let's just create 5 docs
    for i in range(5):
        requests.post(f"{BASE_URL}/add-document", json={
            'title': f'TopK Doc {i}',
            'content': 'common_term_for_top_k'
        })
        
    response = requests.post(f"{BASE_URL}/search", json={
        'query': 'common_term_for_top_k',
        'top_k': 3
    })
    
    results = response.json().get('results', [])
    print(f"Requested top_k=3, got {len(results)} results.")
    
    if len(results) == 3:
        print("SUCCESS: top_k working correctly.")
    else:
        print(f"FAILED: top_k not working (got {len(results)}).")

if __name__ == '__main__':
    test_external_file_upload_and_search()
    test_top_k()
