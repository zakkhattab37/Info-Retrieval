# Information Retrieval System

A Flask-based Information Retrieval (IR) web application with a modular core. It supports multiple ranking algorithms (Boolean, Vector Space Model variants, and BM25), interactive search, relevance-based evaluation (Precision, Recall, F1, Accuracy), and a clean API.

## Features
- Upload or add documents (TXT, PDF, DOCX, CSV, JSON)
- Build an inverted index with tokenization, stopword removal, and optional stemming
- Multiple search/ranking algorithms:
  - Boolean Model (AND/OR/NOT)
  - Vector Space Model (TF, TF-IDF, Log TF-IDF)
  - BM25
- Compare algorithms with metrics and timing
- Simple, responsive web UI (Flask + HTML)
- REST API for automation

## Quick Start

- Requirements: Python 3.10+ (3.12 recommended) on Windows
- Optional: Use a virtual environment

### 1) Install dependencies
1. Open a command prompt in the project folder.
2. Install requirements:
   - Create/activate a virtual environment (recommended)
   - Install Python dependencies from `requirements.txt`

### 2) Run the web app
- Option A (recommended): Run `run.py` which starts the Flask server on 127.0.0.1:5000.
- Option B: Double-click `RUN_APP.bat` (Windows) to start `app.py` directly.

Then open your browser at: http://127.0.0.1:5000

## Project Structure

- `app.py` — Flask application (routes and API)
- `run.py` — Convenience runner for the Flask server
- `RUN_APP.bat` — Windows launcher script
- `ir_core/` — Core logic (package)
  - `dataset_loader.py` — Load documents (TXT, PDF, DOCX, CSV, JSON) and standard dataset structures
  - `document_loader.py` — Document entity and loader helpers
  - `preprocessor.py` — Text preprocessing (tokenization, stopwords, stemming)
  - `pipeline.py` — Tokenizer, InvertedIndex, QueryProcessor, Rankers (Boolean, VSM, BM25)
  - `evaluation.py` — Metrics (Precision, Recall, F1, Accuracy, AP, P@K, NDCG, MRR)
  - `search_algorithms.py` — Alternative implementations (kept for reference/CLI use if needed)
- `templates/index.html` — Web UI
- `uploads/` — Uploaded files directory (auto-created)
- `requirements.txt` — Python dependencies

Removed redundant files for clarity: `demo.py`, `main.py`, `HOW_TO_USE.md`, `README_WEB_APP.md`, `USER_GUIDE.md`, `__pycache__/`.

## How to Use (Web UI)
1. Start the server and open the web UI.
2. Load sample data or upload your documents (TXT, PDF, DOCX, or CSV/JSON with configurable fields).
3. Choose a search algorithm.
4. Enter a query and view ranked results with matched terms and word frequencies.
5. Optionally provide relevant document IDs to compute Precision, Recall, F1, and Accuracy, or use built-in defaults for known sample queries.
6. Use the Compare feature to evaluate all algorithms side-by-side.

## API Reference (JSON)
Base URL: `http://127.0.0.1:5000`

- `GET /api/status` — Returns system status and index stats.
- `POST /api/load-sample` — Loads built-in sample documents and builds the index.
- `POST /api/upload` — Multipart upload for files (supports txt, pdf, docx, doc, csv, json). Optional form fields for CSV/JSON columns.
- `POST /api/add-document` — JSON: `{ "title": str, "content": str }` to add one document.
- `GET /api/documents` — Lists all documents with previews.
- `POST /api/set-algorithm` — JSON: `{ "algorithm": "boolean"|"vsm_tf"|"vsm_tfidf"|"vsm_log"|"bm25" }`.
- `POST /api/search` — JSON: `{ "query": str, "algorithm"?: str, "top_k"?: int }`.
- `POST /api/search-with-evaluation` — JSON: `{ "query": str, "algorithm"?: str, "top_k"?: int, "relevant_ids"?: int[] }`.
- `POST /api/compare-algorithms` — JSON: `{ "query": str, "top_k"?: int, "relevant_ids"?: int[] }`.
- `POST /api/clear` — Clears all in-memory state.
- `POST /api/set-relevance` — JSON: `{ "query": str, "relevant_ids": int[] }` to set judgments.

## Algorithms Overview

- Boolean Model
  - Matching: exact term presence with AND/OR/NOT.
  - Best case: few terms with highly selective posting lists; runs fast.
  - Worst case: common terms (stopwords not filtered) lead to large intersections.
  - Complexity: roughly proportional to posting list scans; set operations on posting sets.

- Vector Space Model (TF, TF-IDF, Log TF-IDF)
  - Matching: cosine similarity between query and document vectors.
  - Best case: sparse vectors with few overlaps; efficient comparisons.
  - Worst case: very high-dimensional vocabularies; requires computing similarities to many docs without pruning.
  - Complexity: O(N · avg_overlap) for N documents considered; with precomputed vectors.

- BM25
  - Probabilistic scoring with term frequency saturation and length normalization.
  - Best case: discriminative terms with moderate document frequency.
  - Worst case: extremely common terms (very low IDF) and very long documents.
  - Complexity: O(sum of posting lengths for query terms) with straightforward implementation.

Notes on speed: The app reports search and total time (ms) per request. For larger corpora, consider caching document vectors, pruning by IDF thresholds, or using specialized indexes.

## Evaluation Metrics
- Precision = relevant ∩ retrieved / retrieved
- Recall = relevant ∩ retrieved / relevant
- F1 = 2PR/(P+R)
- Accuracy = (TP + TN)/Total
- Advanced: P@K, Average Precision, NDCG, MRR

If you see 0.0% across metrics, ensure you provided relevant document IDs or used one of the built-in sample queries (the app auto-supplies defaults for common queries like “machine learning”).

## Troubleshooting
- Import errors after cleanup: ensure you run from the project root so `ir_core` is importable.
- PDF/DOCX parsing: requires optional libs from requirements; for unusual files, convert to TXT.
- Large CSV/JSON: specify correct content/title field names when uploading.
- Stemming requires NLTK; if not installed, the system gracefully disables stemming.


## MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions

