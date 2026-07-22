import pytest
from src.theme_based_rag_backend.jira_exporter_adapter import get_mock_jira_tickets, convert_jira_ticket_to_document
from src.theme_based_rag_backend.agent_flow.state import AgentState
from src.theme_based_rag_backend.agent_flow.nodes.routing import routing_node
from src.theme_based_rag_backend.agent_flow.nodes.jira_bug_qa import jira_bug_qa_node

def test_jira_ticket_conversion():
    mock_tickets = get_mock_jira_tickets()
    assert len(mock_tickets) > 0
    
    doc = convert_jira_ticket_to_document(mock_tickets[0])
    # Verify searchable content contains summary and description ONLY
    assert "Summary: Qdrant vector store connection timeout" in doc.page_content
    assert "Description:" in doc.page_content
    assert doc.metadata["key"] == "KANZI-101"

def test_routing_jira_bug():
    state: AgentState = {
        "message": "Error log: qdrant connection timed out on port 6333",
        "history": [],
        "category": "refuse",
        "retrieved_documents": None,
        "jira_tickets": [],
        "horizontal_repair_advice": None,
        "draft_response": "",
        "critique_feedback": None,
        "attempts": 0
    }
    
    result = routing_node(state)
    assert result["category"] == "jira_bug"

def test_jira_bug_qa_node_fallback():
    state: AgentState = {
        "message": "qdrant_client.http.exceptions.ResponseHandlingException timeout on port 6333",
        "history": [],
        "category": "jira_bug",
        "retrieved_documents": None,
        "jira_tickets": [],
        "horizontal_repair_advice": None,
        "draft_response": "",
        "critique_feedback": None,
        "attempts": 0
    }
    
    result = jira_bug_qa_node(state)
    assert "jira_tickets" in result
    assert "draft_response" in result
    assert len(result["jira_tickets"]) > 0
