import requests

BASE_URL = 'http://127.0.0.1:5000/api'

def debug_system_state():
    # Check documents
    response = requests.get(f"{BASE_URL}/documents")
    data = response.json()
    print(f"Total documents in API: {len(data.get('documents', []))}")
    for doc in data.get('documents', [])[-3:]:
        print(f"Doc ID: {doc['id']}, Title: {doc['title']}, Length: {doc['length']}")
        # We can't see content here because of potential truncation if I hadn't fixed it, 
        # but I did fix truncation in the previous step.
        print(f"Content Preview: {doc['content'][:50]}...")

    # Upload again just to be sure
    filename = 'debug_upload.txt'
    with open(filename, 'w') as f:
        f.write("banana orange apple pineapple")
    
    with open(filename, 'rb') as f:
        requests.post(f"{BASE_URL}/upload", files={'files': (filename, f, 'text/plain')})
    
    # Check documents again
    response = requests.get(f"{BASE_URL}/documents")
    data = response.json()
    print(f"Total documents after upload: {len(data.get('documents', []))}")
    
    # Search for banana
    print("Searching for 'banana'...")
    response = requests.post(f"{BASE_URL}/search", json={'query': 'banana'})
    search_data = response.json()
    print("Search Result Count:", search_data.get('total_results'))
    print("Results:", search_data.get('results'))

if __name__ == '__main__':
    debug_system_state()
