import requests
import json

BASE_URL = 'http://127.0.0.1:5000/api'

def verify_system():
    print("Verifying system functionality...")
    
    # 1. Verify Top K
    print("\n1. Verifying Top K...")
    # Add 12 documents with a common unique term
    term = "strawberry"
    for i in range(12):
        requests.post(f"{BASE_URL}/add-document", json={
            'title': f'Strawberry Doc {i}',
            'content': f'This document mentions {term} multiple times. {term} {term}.'
        })
    
    # Search with top_k=5
    resp = requests.post(f"{BASE_URL}/search", json={'query': term, 'top_k': 5})
    results = resp.json().get('results', [])
    print(f"Top 5 Search: Got {len(results)} results.")
    
    if len(results) == 5:
        print("SUCCESS: Top K limit working.")
    else:
        print(f"FAILED: Top K limit not respected (got {len(results)}).")

    # 2. Verify Metrics Update for External File (simulated via add-document)
    print("\n2. Verifying Metrics Update...")
    # We use the first result from above
    if results:
        doc_id = results[0]['doc_id']
        print(f"Marking Doc ID {doc_id} as relevant for '{term}'...")
        
        # Initial check
        resp = requests.post(f"{BASE_URL}/search-with-evaluation", json={'query': term, 'top_k': 5})
        prec = resp.json().get('evaluation', {}).get('precision', 0)
        print(f"Initial Precision: {prec}")
        
        # Mark relevant
        requests.post(f"{BASE_URL}/set-relevance", json={
            'query': term,
            'relevant_ids': [doc_id]
        })
        
        # Check update
        resp = requests.post(f"{BASE_URL}/search-with-evaluation", json={'query': term, 'top_k': 5})
        eval_data = resp.json().get('evaluation', {})
        new_prec = eval_data.get('precision', 0)
        print(f"Updated Precision: {new_prec}")
        print(f"Correctness: {eval_data}")
        
        if new_prec > 0:
            print("SUCCESS: Metric updated.")
        else:
            print("FAILED: Metric did not update.")
            
if __name__ == '__main__':
    verify_system()
