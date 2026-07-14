import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app
from src.reranker import rerank_documents
from src.config import OLLAMA_MODEL
from langchain_core.documents import Document

client = TestClient(app)

def test_health_check():
    with patch("src.vector_db.vector_store") as mock_vector:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model"] == OLLAMA_MODEL

@patch("src.vector_db.vector_store")
def test_ingest_endpoint(mock_vector_store):
    # Mock Chroma add_documents
    mock_vector_store.add_documents.return_value = None
    
    response = client.post(
        "/ingest",
        json={
            "text": "Hello World. This is project Supernova 9.",
            "metadata": {"source": "test"}
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["chunk_count"] > 0
    mock_vector_store.add_documents.assert_called_once()

def test_rerank_documents():
    query = "project Supernova"
    doc1 = Document(page_content="This is project Supernova 9.", metadata={})
    doc2 = Document(page_content="This is a completely unrelated file.", metadata={})
    
    # doc1 has high semantic similarity (dist 0.1 -> sim 0.9) and high term overlap
    # doc2 has low similarity (dist 0.9 -> sim 0.1) and low overlap
    docs_with_scores = [
        (doc1, 0.1),
        (doc2, 0.9)
    ]
    
    reranked = rerank_documents(query, docs_with_scores)
    assert len(reranked) == 2
    # doc1 should rank first
    assert reranked[0][0].page_content == "This is project Supernova 9."

@patch("src.main.llm_with_tools")
def test_query_no_tools(mock_llm_with_tools):
    mock_res = MagicMock()
    mock_res.content = "Paris is the capital of France."
    mock_res.tool_calls = []
    mock_llm_with_tools.invoke.return_value = mock_res
    
    response = client.post("/query", json={"message": "What is the capital of France?", "history": []})
    assert response.status_code == 200
    assert response.json()["response"] == "Paris is the capital of France."
    assert response.json()["tool_calls_executed"] == []
    assert response.json()["fallback_triggered"] == False

@patch("src.vector_db.vector_store")
@patch("src.main.llm_with_tools")
def test_query_local_vector_retrieval(mock_llm_with_tools, mock_vector_store):
    # First invoke returns tool call to local db search
    mock_res_tool = MagicMock()
    mock_res_tool.content = ""
    mock_res_tool.tool_calls = [{
        "name": "retrieve_local_documents",
        "args": {"query": "Supernova 9"},
        "id": "call_1"
    }]
    
    # Second invoke returns final compiled response
    mock_res_answer = MagicMock()
    mock_res_answer.content = "Supernova 9 is our internal pipeline."
    mock_res_answer.tool_calls = []
    
    mock_llm_with_tools.invoke.side_effect = [mock_res_tool, mock_res_answer]
    
    # Mock Vector search results
    doc = Document(page_content="This is project Supernova 9.", metadata={})
    mock_vector_store.similarity_search_with_score.return_value = [(doc, 0.1)]
    
    response = client.post("/query", json={"message": "What is Supernova 9?", "history": []})
    assert response.status_code == 200
    assert response.json()["response"] == "Supernova 9 is our internal pipeline."
    assert "retrieve_local_documents" in response.json()["tool_calls_executed"]
    assert response.json()["fallback_triggered"] == False

@patch("src.main.llm_with_tools")
def test_query_tool_hallucination_safeguard(mock_llm_with_tools):
    # Simulate a hallucinated tool call name 'non_existent_tool'
    mock_res_tool = MagicMock()
    mock_res_tool.content = ""
    mock_res_tool.tool_calls = [{
        "name": "non_existent_tool",
        "args": {"query": "something"},
        "id": "call_fake"
    }]
    
    # Since tool call is invalid, it triggers safeguard fallback and re-invokes LLM
    mock_res_answer = MagicMock()
    mock_res_answer.content = "This is a direct response output."
    mock_res_answer.tool_calls = []
    
    mock_llm_with_tools.invoke.side_effect = [mock_res_tool, mock_res_answer]
    
    response = client.post("/query", json={"message": "Can you run this?", "history": []})
    assert response.status_code == 200
    assert response.json()["response"] == "This is a direct response output."
    assert response.json()["tool_calls_executed"] == []
    assert response.json()["fallback_triggered"] == True
