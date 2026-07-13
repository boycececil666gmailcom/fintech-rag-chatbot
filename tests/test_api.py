import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "qwen2.5:1.5b"

@patch("src.main.llm")
def test_query_no_search(mock_llm):
    # Mock LLM behavior:
    # First invocation is classification prompt -> returns 'NO'
    # Second invocation is direct query prompt -> returns mock answer
    mock_res_classification = MagicMock()
    mock_res_classification.content = "NO"
    
    mock_res_answer = MagicMock()
    mock_res_answer.content = "Paris is the capital of France."
    
    mock_llm.invoke.side_effect = [mock_res_classification, mock_res_answer]
    
    response = client.post("/query", json={"message": "What is the capital of France?"})
    
    assert response.status_code == 200
    assert response.json()["response"] == "Paris is the capital of France."
    assert mock_llm.invoke.call_count == 2

@patch("src.main.search_tool")
@patch("src.main.llm")
def test_query_with_search(mock_llm, mock_search_tool):
    # Mock LLM behavior:
    # First invocation is classification -> returns 'YES'
    # Second invocation is query with search context -> returns search-based answer
    mock_res_classification = MagicMock()
    mock_res_classification.content = "YES"
    
    mock_res_answer = MagicMock()
    mock_res_answer.content = "The current US president is Donald John Trump."
    
    mock_llm.invoke.side_effect = [mock_res_classification, mock_res_answer]
    
    # Mock search results
    mock_search_tool.run.return_value = "Donald John Trump assumed office in 2025."
    
    response = client.post("/query", json={"message": "Who is the current USA president?"})
    
    assert response.status_code == 200
    assert response.json()["response"] == "The current US president is Donald John Trump."
    mock_search_tool.run.assert_called_once_with("Who is the current USA president?")
    assert mock_llm.invoke.call_count == 2
