"""
Information Retrieval Pipeline Module
Handles the complete IR pipeline: Tokenization -> Indexing -> Query Processing -> Ranking
"""

import math
import time
import re
import string
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Token:
    """Represents a token with its position and metadata"""
    term: str
    position: int
    original: str  # Original form before preprocessing


@dataclass
class PostingEntry:
    """Entry in a posting list"""
    doc_id: int
    term_frequency: int
    positions: List[int] = field(default_factory=list)


@dataclass
class IndexStatistics:
    """Statistics about the index"""
    total_documents: int = 0
    total_terms: int = 0
    unique_terms: int = 0
    avg_document_length: float = 0.0
    index_build_time: float = 0.0


class Tokenizer:
    """
    Tokenization Module
    Handles text tokenization with various options
    """

    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
        'here', 'there', 'then', 'once', 'if', 'because', 'about', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'between',
        'under', 'again', 'further', 'while', 'any', 'being', 'having', 'doing',
        # Arabic Stopwords
        'في', 'من', 'على', 'إلى', 'عن', 'مع', 'ما', 'لا', 'أن', 'كان', 'هو', 'هي', 'هم',
        'هذا', 'هذه', 'تلك', 'ذلك', 'الذي', 'التي', 'الذين', 'كل', 'عند', 'أو', 'ثم',
        'حيث', 'كيف', 'متى', 'لماذا', 'كم', 'أي', 'يا', 'بين', 'فوق', 'تحت', 'بعد',
        'وقد', 'حتى', 'كما', 'ومن', 'فإن', 'لو', 'إذا', 'غير', 'ولكن'
    }

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True,
                 remove_stopwords: bool = True, use_stemming: bool = True,
                 min_token_length: int = 2):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.min_token_length = min_token_length
        self.stemmer = None

        self.stemmer = None
        self.arabic_stemmer = None

        if use_stemming:
            try:
                from nltk.stem import PorterStemmer
                from nltk.stem.isri import ISRIStemmer
                self.stemmer = PorterStemmer()
                self.arabic_stemmer = ISRIStemmer()
            except ImportError:
                print("NLTK not installed. Stemming disabled.")
                self.use_stemming = False

    def tokenize(self, text: str, keep_positions: bool = True) -> List[Token]:
        """
        Tokenize text into a list of Token objects
        """
        if not text:
            return []

        # Store original for reference
        original_text = text

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Split into words while tracking positions
        tokens = []
        position = 0

        # Use regex to find words - Updated to include Arabic
        # \w matches [a-zA-Z0-9_] plus unicode characters (including Arabic) in Python 3
        # We use \w+ to catch everything, and if strict punctuation removal is needed we filter later
        # OR we can just use \w+ which effectively removes punctuation that isn't part of a word.
        # The previous r'\b[a-zA-Z0-9]+\b' was too restrictive for Arabic.
        pattern = r'\b\w+\b' 

        for match in re.finditer(pattern, text):
            term = match.group(0)
            
            # If we want strict alphanumeric (excluding underscores etc) we can check here
            # But usually \w is fine. If remove_punctuation is True, we might want to filter out tokens that are JUST underscores?
            # For now, \w+ is a good standard for "words".
            
            original_term_form = original_text[match.start():match.end()]
            
            # --- Processing Pipeline for Term ---
            
            # 1. Filter by length
            if len(term) < self.min_token_length:
                continue

            # 2. Stopword Removal
            if self.remove_stopwords and term in self.STOPWORDS:
                continue

            # 3. Stemming
            if self.use_stemming:
                 # Check if the term has Arabic characters
                 is_arabic = any('\u0600' <= char <= '\u06FF' for char in term)
                 
                 if is_arabic and self.arabic_stemmer:
                     term = self.arabic_stemmer.stem(term)
                 elif not is_arabic and self.stemmer:
                     term = self.stemmer.stem(term)

            tokens.append(Token(
                term=term,
                position=position if not keep_positions else match.start(), # Fix: match.start() is char pos, position is token index. 
                # Wait, Token definition says "position: int". Usually this is token index (0, 1, 2...). 
                # Previous code used 'position' variable.
                # Let's stick to 'position' which increments.
                # However, original code 'position=position' is correct for token index.
                original=original_term_form
            ))
            position += 1

        return tokens

    def tokenize_simple(self, text: str) -> List[str]:
        """Simple tokenization returning just terms"""
        tokens = self.tokenize(text, keep_positions=False)
        return [t.term for t in tokens]

    def get_term_frequencies(self, tokens: List[Token]) -> Dict[str, int]:
        """Get term frequency from tokens"""
        tf = Counter(t.term for t in tokens)
        return dict(tf)


class InvertedIndex:
    """
    Indexing Module
    Creates and manages an inverted index for document retrieval
    """

    def __init__(self, tokenizer: Tokenizer = None):
        self.tokenizer = tokenizer or Tokenizer()

        # Main index: term -> list of PostingEntry
        self.index: Dict[str, List[PostingEntry]] = defaultdict(list)

        # Document information
        self.doc_lengths: Dict[int, int] = {}  # doc_id -> document length
        self.doc_vectors: Dict[int, Dict[str, float]] = {}  # doc_id -> term weights

        # Statistics
        self.stats = IndexStatistics()
        self.document_frequency: Dict[str, int] = defaultdict(int)  # term -> df
        self.vocabulary: Set[str] = set()

    def build_index(self, documents: List[Any]) -> IndexStatistics:
        """
        Build the inverted index from documents

        Pipeline Step: INDEXING
        """
        start_time = time.time()

        # Clear existing index
        self.index.clear()
        self.doc_lengths.clear()
        self.doc_vectors.clear()
        self.document_frequency.clear()
        self.vocabulary.clear()

        total_terms = 0

        for doc in documents:
            # Tokenize document
            tokens = self.tokenizer.tokenize(doc.content)

            # Store document length
            self.doc_lengths[doc.doc_id] = len(tokens)
            total_terms += len(tokens)

            # Get term frequencies and positions
            term_positions: Dict[str, List[int]] = defaultdict(list)
            term_freqs: Dict[str, int] = defaultdict(int)

            for token in tokens:
                term_positions[token.term].append(token.position)
                term_freqs[token.term] += 1
                self.vocabulary.add(token.term)

            # Update document frequency and index
            for term, positions in term_positions.items():
                self.document_frequency[term] += 1

                posting = PostingEntry(
                    doc_id=doc.doc_id,
                    term_frequency=term_freqs[term],
                    positions=positions
                )
                self.index[term].append(posting)

        # Calculate statistics
        self.stats.total_documents = len(documents)
        self.stats.total_terms = total_terms
        self.stats.unique_terms = len(self.vocabulary)
        self.stats.avg_document_length = total_terms / len(documents) if documents else 0
        self.stats.index_build_time = time.time() - start_time

        return self.stats

    def get_postings(self, term: str) -> List[PostingEntry]:
        """Get posting list for a term"""
        return self.index.get(term, [])

    def get_document_frequency(self, term: str) -> int:
        """Get document frequency for a term"""
        return self.document_frequency.get(term, 0)

    def get_idf(self, term: str) -> float:
        """Calculate IDF for a term"""
        df = self.get_document_frequency(term)
        if df == 0:
            return 0.0
        return math.log(self.stats.total_documents / df)

    def get_index_info(self) -> str:
        """Get index information as string"""
        info = f"""
Index Statistics:
  - Total Documents: {self.stats.total_documents}
  - Total Terms: {self.stats.total_terms}
  - Unique Terms: {self.stats.unique_terms}
  - Avg Document Length: {self.stats.avg_document_length:.2f}
  - Index Build Time: {self.stats.index_build_time*1000:.2f} ms
"""
        return info


class QueryProcessor:
    """
    Query Processing Module
    Parses and processes user queries
    """

    def __init__(self, tokenizer: Tokenizer = None):
        self.tokenizer = tokenizer or Tokenizer()

    def parse_boolean_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a boolean query with AND, OR, NOT operators

        Examples:
        - "machine learning" -> AND query
        - "machine OR learning" -> OR query
        - "machine NOT learning" -> NOT query
        - "machine AND learning OR deep" -> Mixed query
        """
        query = query.strip()

        # Detect query type
        has_or = ' OR ' in query.upper()
        has_not = ' NOT ' in query.upper()
        has_and = ' AND ' in query.upper()

        result = {
            'type': 'boolean',
            'and_terms': [],
            'or_terms': [],
            'not_terms': [],
            'original_query': query
        }

        # Process NOT terms first
        if has_not:
            parts = re.split(r'\s+NOT\s+', query, flags=re.IGNORECASE)
            query = parts[0]
            for part in parts[1:]:
                not_tokens = self.tokenizer.tokenize_simple(part)
                result['not_terms'].extend(not_tokens)

        # Process OR terms
        if has_or:
            or_parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
            for part in or_parts:
                tokens = self.tokenizer.tokenize_simple(part)
                result['or_terms'].extend(tokens)
        elif has_and:
            # Explicit AND
            and_parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
            for part in and_parts:
                tokens = self.tokenizer.tokenize_simple(part)
                result['and_terms'].extend(tokens)
        else:
            # Default: treat as AND query
            tokens = self.tokenizer.tokenize_simple(query)
            result['and_terms'] = tokens

        return result

    def parse_vector_query(self, query: str) -> Dict[str, float]:
        """
        Parse query for vector space model
        Returns term weights for the query
        """
        tokens = self.tokenizer.tokenize_simple(query)
        tf = Counter(tokens)

        # Normalize by max tf
        max_tf = max(tf.values()) if tf else 1
        query_vector = {term: count / max_tf for term, count in tf.items()}

        return query_vector

    def expand_query(self, query: str, synonyms: Dict[str, List[str]] = None) -> str:
        """
        Expand query with synonyms (optional query expansion)
        """
        if not synonyms:
            return query

        tokens = self.tokenizer.tokenize_simple(query)
        expanded_terms = set(tokens)

        for token in tokens:
            if token in synonyms:
                expanded_terms.update(synonyms[token])

        return ' OR '.join(expanded_terms)


@dataclass
class RankedResult:
    """Represents a ranked search result"""
    doc_id: int
    score: float
    rank: int
    document: Any = None
    matched_terms: List[str] = field(default_factory=list)


class Ranker(ABC):
    """Abstract base class for ranking algorithms"""

    @abstractmethod
    def rank(self, query_terms: List[str], index: InvertedIndex,
             documents: Dict[int, Any], top_k: int = 10) -> List[RankedResult]:
        pass


class BooleanRanker(Ranker):
    """
    Boolean Retrieval Model

    Implements exact matching with AND, OR, NOT operators
    No ranking - documents either match or don't
    """

    def __init__(self, query_processor: QueryProcessor = None):
        self.query_processor = query_processor or QueryProcessor()
        self.name = "Boolean Model"

    def rank(self, query: str, index: InvertedIndex,
             documents: Dict[int, Any], top_k: int = 10) -> List[RankedResult]:
        """
        Perform boolean retrieval
        """
        # Parse query
        parsed = self.query_processor.parse_boolean_query(query)

        all_doc_ids = set(documents.keys())
        result_docs = None

        # Process AND terms
        if parsed['and_terms']:
            for term in parsed['and_terms']:
                postings = index.get_postings(term)
                doc_ids = {p.doc_id for p in postings}

                if result_docs is None:
                    result_docs = doc_ids
                else:
                    result_docs = result_docs.intersection(doc_ids)

        # Process OR terms
        if parsed['or_terms']:
            or_docs = set()
            for term in parsed['or_terms']:
                postings = index.get_postings(term)
                or_docs.update(p.doc_id for p in postings)

            if result_docs is None:
                result_docs = or_docs
            else:
                result_docs = result_docs.union(or_docs)

        # Process NOT terms
        if parsed['not_terms']:
            not_docs = set()
            for term in parsed['not_terms']:
                postings = index.get_postings(term)
                not_docs.update(p.doc_id for p in postings)

            if result_docs is not None:
                result_docs = result_docs - not_docs

        if result_docs is None:
            result_docs = set()

        # Create results (no ranking, all have score 1.0)
        results = []
        for rank, doc_id in enumerate(sorted(result_docs)[:top_k], 1):
            results.append(RankedResult(
                doc_id=doc_id,
                score=1.0,
                rank=rank,
                document=documents.get(doc_id),
                matched_terms=parsed['and_terms'] + parsed['or_terms']
            ))

        return results


class VectorSpaceRanker(Ranker):
    """
    Vector Space Model

    Implements TF-IDF weighting with cosine similarity
    Supports different weighting schemes:
    - TF: Raw term frequency
    - TF-IDF: Term frequency * Inverse document frequency
    - Log TF-IDF: Log normalized TF * IDF
    """

    def __init__(self, query_processor: QueryProcessor = None,
                 weighting: str = 'tfidf'):
        self.query_processor = query_processor or QueryProcessor()
        self.weighting = weighting  # 'tf', 'tfidf', 'log_tfidf'
        self.name = f"Vector Space Model ({weighting})"

        # Cache for document vectors
        self._doc_vectors: Dict[int, Dict[str, float]] = {}

    def _calculate_term_weight(self, tf: int, idf: float) -> float:
        """Calculate term weight based on weighting scheme"""
        if self.weighting == 'tf':
            return tf
        elif self.weighting == 'tfidf':
            return tf * idf
        elif self.weighting == 'log_tfidf':
            return (1 + math.log(tf)) * idf if tf > 0 else 0
        else:
            return tf * idf

    def build_document_vectors(self, index: InvertedIndex):
        """Pre-compute document vectors for faster retrieval"""
        self._doc_vectors.clear()

        for term, postings in index.index.items():
            idf = index.get_idf(term)

            for posting in postings:
                if posting.doc_id not in self._doc_vectors:
                    self._doc_vectors[posting.doc_id] = {}

                weight = self._calculate_term_weight(posting.term_frequency, idf)
                self._doc_vectors[posting.doc_id][term] = weight

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors"""
        common_terms = set(vec1.keys()) & set(vec2.keys())

        if not common_terms:
            return 0.0

        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def rank(self, query: str, index: InvertedIndex,
             documents: Dict[int, Any], top_k: int = 10) -> List[RankedResult]:
        """
        Rank documents using vector space model
        """
        # Build document vectors if not cached
        if not self._doc_vectors:
            self.build_document_vectors(index)

        # Parse query and create query vector
        query_terms = self.query_processor.parse_vector_query(query)

        # Apply IDF weighting to query
        query_vector = {}
        for term, tf in query_terms.items():
            idf = index.get_idf(term)
            query_vector[term] = self._calculate_term_weight(tf, idf) if idf > 0 else tf

        if not query_vector:
            return []

        # Calculate similarity with all documents
        scores = []
        for doc_id, doc_vector in self._doc_vectors.items():
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                matched = [t for t in query_terms.keys() if t in doc_vector]
                scores.append((doc_id, similarity, matched))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Create results
        results = []
        for rank, (doc_id, score, matched) in enumerate(scores[:top_k], 1):
            results.append(RankedResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                document=documents.get(doc_id),
                matched_terms=matched
            ))

        return results


class BM25Ranker(Ranker):
    """
    BM25 (Best Matching 25) Ranking Model

    Probabilistic model that improves on TF-IDF with:
    - Term frequency saturation
    - Document length normalization
    """

    def __init__(self, query_processor: QueryProcessor = None,
                 k1: float = 1.5, b: float = 0.75):
        self.query_processor = query_processor or QueryProcessor()
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization
        self.name = "BM25"

    def rank(self, query: str, index: InvertedIndex,
             documents: Dict[int, Any], top_k: int = 10) -> List[RankedResult]:
        """
        Rank documents using BM25
        """
        query_terms = list(self.query_processor.parse_vector_query(query).keys())

        if not query_terms:
            return []

        N = index.stats.total_documents
        avgdl = index.stats.avg_document_length

        scores: Dict[int, float] = defaultdict(float)
        matched_terms: Dict[int, List[str]] = defaultdict(list)

        for term in query_terms:
            postings = index.get_postings(term)
            df = index.get_document_frequency(term)

            # BM25 IDF formula
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1) if df > 0 else 0

            for posting in postings:
                doc_id = posting.doc_id
                tf = posting.term_frequency
                dl = index.doc_lengths.get(doc_id, avgdl)

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (dl / avgdl))

                scores[doc_id] += idf * (numerator / denominator)
                matched_terms[doc_id].append(term)

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Create results
        results = []
        for rank, (doc_id, score) in enumerate(sorted_scores[:top_k], 1):
            results.append(RankedResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                document=documents.get(doc_id),
                matched_terms=matched_terms[doc_id]
            ))

        return results


class IRPipeline:
    """
    Complete Information Retrieval Pipeline

    Orchestrates: Tokenization -> Indexing -> Query Processing -> Ranking
    """

    def __init__(self):
        # Pipeline components
        self.tokenizer = Tokenizer()
        self.index = InvertedIndex(self.tokenizer)
        self.query_processor = QueryProcessor(self.tokenizer)

        # Available rankers
        self.rankers = {
            'boolean': BooleanRanker(self.query_processor),
            'vsm_tf': VectorSpaceRanker(self.query_processor, 'tf'),
            'vsm_tfidf': VectorSpaceRanker(self.query_processor, 'tfidf'),
            'vsm_log': VectorSpaceRanker(self.query_processor, 'log_tfidf'),
            'bm25': BM25Ranker(self.query_processor),
        }

        self.current_ranker = self.rankers['bm25']
        self.documents: Dict[int, Any] = {}

        # Pipeline timing
        self.timing = {
            'tokenization': 0.0,
            'indexing': 0.0,
            'query_processing': 0.0,
            'ranking': 0.0,
            'total': 0.0
        }

    def index_documents(self, documents: List[Any]) -> IndexStatistics:
        """
        Index documents through the pipeline

        Pipeline: Documents -> Tokenization -> Indexing
        """
        start = time.time()

        # Store documents
        self.documents = {doc.doc_id: doc for doc in documents}

        # Build index (includes tokenization)
        stats = self.index.build_index(documents)

        # Rebuild document vectors for VSM rankers
        for ranker in self.rankers.values():
            if isinstance(ranker, VectorSpaceRanker):
                ranker.build_document_vectors(self.index)

        self.timing['indexing'] = time.time() - start
        return stats

    def search(self, query: str, ranker_name: str = None,
               top_k: int = 10) -> Tuple[List[RankedResult], Dict[str, float]]:
        """
        Search through the complete pipeline

        Pipeline: Query -> Tokenization -> Query Processing -> Ranking -> Results
        """
        total_start = time.time()

        # Select ranker
        if ranker_name and ranker_name in self.rankers:
            self.current_ranker = self.rankers[ranker_name]

        # Query processing and ranking
        query_start = time.time()
        results = self.current_ranker.rank(query, self.index, self.documents, top_k)
        self.timing['ranking'] = time.time() - query_start

        self.timing['total'] = time.time() - total_start

        return results, self.timing.copy()

    def set_ranker(self, ranker_name: str) -> bool:
        """Set the current ranking algorithm"""
        if ranker_name in self.rankers:
            self.current_ranker = self.rankers[ranker_name]
            return True
        return False

    def get_available_rankers(self) -> Dict[str, str]:
        """Get available ranking algorithms"""
        return {key: ranker.name for key, ranker in self.rankers.items()}

    def get_pipeline_info(self) -> str:
        """Get pipeline information"""
        info = f"""
IR Pipeline Information:
========================
Tokenizer Settings:
  - Lowercase: {self.tokenizer.lowercase}
  - Remove Stopwords: {self.tokenizer.remove_stopwords}
  - Stemming: {self.tokenizer.use_stemming}
  - Min Token Length: {self.tokenizer.min_token_length}

{self.index.get_index_info()}

Current Ranker: {self.current_ranker.name}

Available Rankers:
{chr(10).join(f'  - {k}: {v.name}' for k, v in self.rankers.items())}
"""
        return info

