import re
from typing import List, Tuple
from langchain_core.documents import Document

class BM25Searcher:
    """
    A wrapper around the rank_bm25 library for BM25 keyword scoring.
    """
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.corpus_size = len(documents)
        
        if self.corpus_size > 0:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError:
                print("Error: rank-bm25 package is not installed. Please run: pip install -r requirements.txt")
                raise
                
            tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if self.corpus_size == 0 or not self.bm25:
            return [(doc, 0.0) for doc in self.documents[:k]]
            
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(doc, 0.0) for doc in self.documents[:k]]
            
        # Get score for each document in the corpus
        scores = self.bm25.get_scores(query_tokens)
        
        # Zip documents with scores
        scored_docs = list(zip(self.documents, scores))
        
        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:k]
