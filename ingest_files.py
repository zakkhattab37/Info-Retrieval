import os
import requests

UPLOAD_FOLDER = 'uploads'
API_URL = 'http://127.0.0.1:5000/api/upload'

def ingest_files():
    if not os.path.exists(UPLOAD_FOLDER):
        print(f"Error: Directory '{UPLOAD_FOLDER}' not found.")
        return

    files_to_upload = []
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath):
            # Check extension
            ext = filename.split('.')[-1].lower()
            if ext in ['txt', 'pdf', 'docx', 'doc', 'csv', 'json']:
                files_to_upload.append(('files', (filename, open(filepath, 'rb'), 'application/octet-stream')))

    if not files_to_upload:
        print("No valid files found in uploads directory.")
        return

    print(f"Uploading {len(files_to_upload)} files...")
    try:
        response = requests.post(API_URL, files=files_to_upload)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"Success! {result.get('message')}")
                print("Stats:", result.get('stats'))
            else:
                print(f"Failed: {result.get('error')}")
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        # Close files
        for _, (name, f, _) in files_to_upload:
            f.close()

if __name__ == '__main__':
    ingest_files()
