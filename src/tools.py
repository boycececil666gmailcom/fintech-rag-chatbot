from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import src.vector_db as db
from src.reranker import rerank_documents

# Setup search tool
search_tool = DuckDuckGoSearchRun()

@tool
def retrieve_local_documents(query: str) -> str:
    """Retrieve semantically relevant document chunks from the local vector database.
    Use this tool when the query refers to private documentation, internal guidelines,
    project names (like 'Supernova'), or local workspace facts."""
    if db.vector_store is None:
        return "Error: Local Vector database is not initialized."
    try:
        # Query database for raw candidates
        docs = db.vector_store.similarity_search_with_score(query, k=5)
        if not docs:
            return "No matching local documents found."
        
        # Apply re-ranking
        reranked_docs = rerank_documents(query, docs)
        
        # Format top 2 chunks as output context
        context_list = []
        for doc, score in reranked_docs[:2]:
            context_list.append(f"[Match Score: {score:.3f}] Content: {doc.page_content}")
            
        return "\n\n".join(context_list)
    except Exception as e:
        return f"Error querying local documents: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Search the public internet for current events, today's weather, real-time news,
    active figures, or public facts."""
    try:
        return search_tool.invoke(query)
    except Exception as e:
        return f"Error performing web search: {str(e)}"
