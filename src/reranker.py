import os
from typing import List, Tuple
from langchain_core.documents import Document

# Lazy load the flashrank Ranker to avoid overhead when importing the module.
_ranker = None

def get_ranker():
    global _ranker
    if _ranker is None:
        try:
            from flashrank import Ranker
            # By default, loads the lightweight "ms-marco-MiniLM-L-6-v2" model (~80 MB)
            _ranker = Ranker()
        except ImportError:
            print("Error: flashrank package is not installed. Please run: pip install -r requirements.txt")
            raise
    return _ranker

def rerank_documents(query: str, doc_score_tuples: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    """
    Applies FlashRank transformer-based cross-encoder reranking.
    doc_score_tuples: a list of (Document, score) tuples.
    Returns sorted doc_score_tuples with the new cross-encoder scores.
    """
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Re-ranker] [{os.path.basename(__file__)}] Re-scoring {len(doc_score_tuples)} candidates using FlashRank Cross-Encoder\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    if not doc_score_tuples:
        return []
        
    try:
        from flashrank import RerankRequest
        
        # Prepare passages for FlashRank
        passages = []
        doc_map = {}
        for i, (doc, _) in enumerate(doc_score_tuples):
            passages.append({
                "id": i,
                "text": doc.page_content
            })
            doc_map[i] = doc
            
        # Get ranker and rerank
        ranker = get_ranker()
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)
        
        # Reconstruct the ranked doc list with the new scores
        scored_list = []
        for item in results:
            doc_id = item["id"]
            score = item["score"]
            doc = doc_map[doc_id]
            scored_list.append((doc, score))
            
        return scored_list
        
    except Exception as e:
        print(f"Reranking failed: {e}. Falling back to original input list order.")
        return doc_score_tuples
