import os

def rerank_documents(query: str, doc_score_tuples: list) -> list:
    """
    Applies a local heuristic re-ranking by combining semantic similarity with
    keyword match density.
    doc_score_tuples is a list of (Document, score) tuples.
    Returns sorted doc_score_tuples.
    """
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Re-ranker] [{os.path.basename(__file__)}] Re-scoring {len(doc_score_tuples)} candidate document chunks\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    query_tokens = set(query.lower().split())
    scored_list = []
    
    for doc, dist_score in doc_score_tuples:
        # Convert distance to similarity
        similarity_score = 1.0 - dist_score if dist_score <= 1.0 else 0.0
        
        # Calculate term overlap match density
        doc_tokens = doc.page_content.lower().split()
        if not doc_tokens:
            overlap_ratio = 0.0
        else:
            matches = sum(1 for token in doc_tokens if token in query_tokens)
            overlap_ratio = matches / len(doc_tokens)
            
        # Combine: 70% semantic similarity + 30% word match density
        final_score = (0.7 * similarity_score) + (0.3 * overlap_ratio)
        scored_list.append((doc, final_score))
        
    # Sort descending by final score
    scored_list.sort(key=lambda x: x[1], reverse=True)
    return scored_list
