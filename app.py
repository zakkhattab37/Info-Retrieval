"""
Information Retrieval System - Web Application
Flask-based web interface with modern UI
"""

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
import json
from typing import Dict, List, Set

from ir_core.dataset_loader import DatasetLoader, Document, IRDataset
from ir_core.pipeline import IRPipeline, RankedResult
from ir_core.evaluation import Evaluator, format_evaluation_result

app = Flask(__name__)
app.secret_key = 'ir_system_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'csv', 'json'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


class IRSystemState:
    """Global state for the IR system"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.dataset_loader = DatasetLoader()
        self.pipeline = IRPipeline()
        self.evaluator = Evaluator()
        self.is_indexed = False
        self.relevance_judgments: Dict[str, Set[int]] = {}
        self._setup_default_judgments()

    def _setup_default_judgments(self):
        """Setup default relevance judgments for sample data"""
        self.relevance_judgments = {
            'machine learning': {1, 2, 7},
            'machine learning algorithms': {1, 2, 7},
            'deep learning': {2, 1},
            'deep learning neural networks': {2, 1},
            'natural language processing': {3},
            'information retrieval': {4},
            'database': {5},
            'database sql': {5},
            'web development': {6},
            'data science': {7, 1, 2},
            'cloud computing': {8},
            'cybersecurity': {9},
            'python': {10, 6, 7},
            'python programming': {10, 6, 7},
            'artificial intelligence': {1, 2, 3},
        }


# Global IR system instance
ir_system = IRSystemState()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get system status"""
    docs = ir_system.dataset_loader.get_all_documents()
    return jsonify({
        'document_count': len(docs),
        'is_indexed': ir_system.is_indexed,
        'current_algorithm': ir_system.pipeline.current_ranker.name if ir_system.is_indexed else 'None',
        'available_algorithms': ir_system.pipeline.get_available_rankers(),
        'index_stats': {
            'unique_terms': ir_system.pipeline.index.stats.unique_terms,
            'total_terms': ir_system.pipeline.index.stats.total_terms,
            'avg_doc_length': round(ir_system.pipeline.index.stats.avg_document_length, 2),
        } if ir_system.is_indexed else None
    })


@app.route('/api/load-sample', methods=['POST'])
def load_sample():
    """Load sample documents"""
    ir_system.reset()
    dataset = ir_system.dataset_loader.create_sample_dataset()

    # Index documents
    stats = ir_system.pipeline.index_documents(ir_system.dataset_loader.get_all_documents())
    ir_system.is_indexed = True

    return jsonify({
        'success': True,
        'message': f'Loaded {len(dataset.documents)} sample documents',
        'documents': [{'id': d.doc_id, 'title': d.title, 'length': len(d.content)}
                     for d in dataset.documents],
        'stats': {
            'total_documents': stats.total_documents,
            'unique_terms': stats.unique_terms,
            'avg_doc_length': round(stats.avg_document_length, 2),
            'index_time_ms': round(stats.index_build_time * 1000, 2)
        }
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process files"""
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files provided'})

    files = request.files.getlist('files')
    loaded_docs = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load based on file type
            ext = filename.rsplit('.', 1)[1].lower()

            if ext in ['txt', 'pdf', 'docx', 'doc']:
                doc = ir_system.dataset_loader.load_single_file(filepath)
                if doc:
                    loaded_docs.append({'id': doc.doc_id, 'title': doc.title, 'length': len(doc.content)})
            elif ext == 'csv':
                content_col = request.form.get('content_column', 'content')
                title_col = request.form.get('title_column', 'title')
                docs = ir_system.dataset_loader.load_csv_dataset(filepath, content_col, title_col)
                loaded_docs.extend([{'id': d.doc_id, 'title': d.title, 'length': len(d.content)} for d in docs])
            elif ext == 'json':
                content_field = request.form.get('content_field', 'content')
                title_field = request.form.get('title_field', 'title')
                docs = ir_system.dataset_loader.load_json_dataset(filepath, content_field, title_field)
                loaded_docs.extend([{'id': d.doc_id, 'title': d.title, 'length': len(d.content)} for d in docs])

    # Re-index if documents were loaded
    if loaded_docs:
        stats = ir_system.pipeline.index_documents(ir_system.dataset_loader.get_all_documents())
        ir_system.is_indexed = True

        return jsonify({
            'success': True,
            'message': f'Loaded {len(loaded_docs)} documents',
            'documents': loaded_docs,
            'stats': {
                'total_documents': stats.total_documents,
                'unique_terms': stats.unique_terms,
                'avg_doc_length': round(stats.avg_document_length, 2),
                'index_time_ms': round(stats.index_build_time * 1000, 2)
            }
        })

    return jsonify({'success': False, 'error': 'No valid documents loaded'})


@app.route('/api/add-document', methods=['POST'])
def add_document():
    """Add a document manually"""
    data = request.json
    title = data.get('title', '').strip()
    content = data.get('content', '').strip()

    if not title or not content:
        return jsonify({'success': False, 'error': 'Title and content are required'})

    doc = ir_system.dataset_loader.add_document(title, content)

    # Re-index
    stats = ir_system.pipeline.index_documents(ir_system.dataset_loader.get_all_documents())
    ir_system.is_indexed = True

    return jsonify({
        'success': True,
        'message': f'Added document: {title}',
        'document': {'id': doc.doc_id, 'title': doc.title, 'length': len(doc.content)},
        'stats': {
            'total_documents': stats.total_documents,
            'unique_terms': stats.unique_terms,
            'index_time_ms': round(stats.index_build_time * 1000, 2)
        }
    })


@app.route('/api/documents')
def get_documents():
    """Get all documents"""
    docs = ir_system.dataset_loader.get_all_documents()
    return jsonify({
        'documents': [
            {
                'id': d.doc_id,
                'title': d.title,
                'content': d.content[:500] + '...' if len(d.content) > 500 else d.content,
                'length': len(d.content)
            }
            for d in docs
        ]
    })


@app.route('/api/set-algorithm', methods=['POST'])
def set_algorithm():
    """Set the search algorithm"""
    data = request.json
    algorithm = data.get('algorithm', 'bm25')

    if ir_system.pipeline.set_ranker(algorithm):
        return jsonify({
            'success': True,
            'algorithm': ir_system.pipeline.current_ranker.name
        })

    return jsonify({'success': False, 'error': 'Unknown algorithm'})


@app.route('/api/search', methods=['POST'])
def search():
    """Perform search"""
    if not ir_system.is_indexed:
        return jsonify({'success': False, 'error': 'No documents indexed'})

    data = request.json
    query = data.get('query', '').strip()
    algorithm = data.get('algorithm', 'bm25')
    top_k = data.get('top_k', 10)

    if not query:
        return jsonify({'success': False, 'error': 'Query is required'})

    # Set algorithm if specified
    if algorithm and algorithm in ir_system.pipeline.rankers:
        ir_system.pipeline.set_ranker(algorithm)

    # Perform search - pass ranker_name as keyword argument
    results, timing = ir_system.pipeline.search(query, ranker_name=algorithm, top_k=top_k)

    # Calculate word frequencies for matched terms
    from collections import Counter
    query_terms = set(ir_system.pipeline.tokenizer.tokenize_simple(query))

    result_list = []
    for r in results:
        # Count word frequency in document
        if r.document:
            doc_content = r.document.content.lower()
            word_frequencies = {}
            for term in query_terms:
                # Count occurrences of the term in document
                count = doc_content.count(term.lower())
                if count > 0:
                    word_frequencies[term] = count
        else:
            word_frequencies = {}

        result_list.append({
            'rank': r.rank,
            'doc_id': r.doc_id,
            'title': r.document.title if r.document else f'Document {r.doc_id}',
            'score': round(r.score, 4),
            'content': r.document.content if r.document else '',
            'matched_terms': r.matched_terms,
            'word_frequencies': word_frequencies
        })

    return jsonify({
        'success': True,
        'query': query,
        'algorithm': ir_system.pipeline.current_ranker.name if ir_system.is_indexed else 'BM25',
        'results': result_list,
        'timing': {
            'search_time_ms': round(timing.get('ranking', 0) * 1000, 4),
            'total_time_ms': round(timing.get('total', 0) * 1000, 4)
        },
        'total_results': len(result_list)
    })


@app.route('/api/search-with-evaluation', methods=['POST'])
def search_with_evaluation():
    """Search with evaluation metrics"""
    if not ir_system.is_indexed:
        return jsonify({'success': False, 'error': 'No documents indexed'})

    data = request.json
    query = data.get('query', '').strip()
    algorithm = data.get('algorithm', 'bm25')
    top_k = data.get('top_k', 10)
    relevant_ids = set(data.get('relevant_ids', []))

    if not query:
        return jsonify({'success': False, 'error': 'Query is required'})

    # Set algorithm if specified
    if algorithm and algorithm in ir_system.pipeline.rankers:
        ir_system.pipeline.set_ranker(algorithm)

    # Get default relevance judgments if not provided
    if not relevant_ids:
        relevant_ids = ir_system.relevance_judgments.get(query.lower(), set())

    # Perform search
    results, timing = ir_system.pipeline.search(query, ranker_name=algorithm, top_k=top_k)

    # Calculate word frequencies for matched terms
    query_terms = set(ir_system.pipeline.tokenizer.tokenize_simple(query))

    result_list = []
    for r in results:
        # Count word frequency in document
        if r.document:
            doc_content = r.document.content.lower()
            word_frequencies = {}
            for term in query_terms:
                count = doc_content.count(term.lower())
                if count > 0:
                    word_frequencies[term] = count
        else:
            word_frequencies = {}

        result_list.append({
            'rank': r.rank,
            'doc_id': r.doc_id,
            'title': r.document.title if r.document else f'Document {r.doc_id}',
            'score': round(r.score, 4),
            'content': r.document.content if r.document else '',
            'matched_terms': r.matched_terms,
            'is_relevant': r.doc_id in relevant_ids,
            'word_frequencies': word_frequencies
        })

    # Calculate metrics
    retrieved_ids = {r.doc_id for r in results}
    total_docs = len(ir_system.dataset_loader.get_all_documents())
    all_doc_ids = set(range(1, total_docs + 1))

    # Calculate metrics properly
    if relevant_ids:
        # User provided relevant IDs, calculate standard metrics
        relevant_retrieved = retrieved_ids & relevant_ids
        precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy calculation
        tp = len(relevant_retrieved)
        fp = len(retrieved_ids - relevant_ids)
        fn = len(relevant_ids - retrieved_ids)
        tn = total_docs - tp - fp - fn
        accuracy = (tp + tn) / total_docs if total_docs > 0 else 0.0
    else:
        # No relevant IDs provided - cannot calculate meaningful metrics
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        accuracy = 0.0

    return jsonify({
        'success': True,
        'query': query,
        'algorithm': ir_system.pipeline.current_ranker.name if ir_system.is_indexed else 'BM25',
        'results': result_list,
        'evaluation': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'accuracy': round(accuracy, 4),
            'retrieved_count': len(retrieved_ids),
            'relevant_count': len(relevant_ids),
            'relevant_retrieved': len(retrieved_ids & relevant_ids) if relevant_ids else 0
        },
        'timing': {
            'search_time_ms': round(timing.get('ranking', 0) * 1000, 4),
            'total_time_ms': round(timing.get('total', 0) * 1000, 4)
        },
        'relevant_ids': list(relevant_ids),
        'has_relevant_ids': len(relevant_ids) > 0
    })


@app.route('/api/compare-algorithms', methods=['POST'])
def compare_algorithms():
    """Compare all algorithms"""
    if not ir_system.is_indexed:
        return jsonify({'success': False, 'error': 'No documents indexed'})

    data = request.json
    query = data.get('query', '').strip()
    top_k = data.get('top_k', 10)
    relevant_ids = set(data.get('relevant_ids', []))

    if not query:
        return jsonify({'success': False, 'error': 'Query is required'})

    # Get default relevance judgments if not provided
    if not relevant_ids:
        relevant_ids = ir_system.relevance_judgments.get(query.lower(), set())

    total_docs = len(ir_system.dataset_loader.documents)
    all_doc_ids = set(range(1, total_docs + 1))

    comparisons = []

    for algo_key, algo_name in ir_system.pipeline.get_available_rankers().items():
        results, timing = ir_system.pipeline.search(query, ranker_name=algo_key, top_k=top_k)
        retrieved_ids = {r.doc_id for r in results}

        # Calculate metrics
        if retrieved_ids:
            relevant_retrieved = retrieved_ids & relevant_ids
            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
            recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0

        tp = len(retrieved_ids & relevant_ids)
        tn = len((all_doc_ids - relevant_ids) & (all_doc_ids - retrieved_ids))
        accuracy = (tp + tn) / total_docs if total_docs > 0 else 0

        comparisons.append({
            'algorithm': algo_name,
            'algorithm_key': algo_key,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'accuracy': round(accuracy, 4),
            'search_time_ms': round(timing['ranking'] * 1000, 4),
            'results_count': len(results)
        })

    # Find best for each metric
    best = {
        'precision': max(comparisons, key=lambda x: x['precision'])['algorithm'],
        'recall': max(comparisons, key=lambda x: x['recall'])['algorithm'],
        'f1_score': max(comparisons, key=lambda x: x['f1_score'])['algorithm'],
        'accuracy': max(comparisons, key=lambda x: x['accuracy'])['algorithm'],
        'speed': min(comparisons, key=lambda x: x['search_time_ms'])['algorithm']
    }

    return jsonify({
        'success': True,
        'query': query,
        'relevant_ids': list(relevant_ids),
        'comparisons': comparisons,
        'best': best
    })


@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear all data"""
    ir_system.reset()
    return jsonify({'success': True, 'message': 'All data cleared'})


@app.route('/api/set-relevance', methods=['POST'])
def set_relevance():
    """Set relevance judgments for a query"""
    data = request.json
    query = data.get('query', '').strip().lower()
    relevant_ids = set(data.get('relevant_ids', []))

    if query:
        ir_system.relevance_judgments[query] = relevant_ids
        return jsonify({'success': True, 'message': f'Set {len(relevant_ids)} relevant documents for query'})

    return jsonify({'success': False, 'error': 'Query is required'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
