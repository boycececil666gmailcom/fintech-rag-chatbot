import importlib.util
from pathlib import Path
from typing import List
from langchain_core.documents import Document

JIRA_DEFINITION_FILE = Path(r"C:\Users\boyce\OneDrive\Desktop\JIRA-exporter\src\definition.py")

JiraTicket = None
JiraConfig = None
JiraClient = None
JiraExporter = None

if JIRA_DEFINITION_FILE.exists():
    try:
        spec = importlib.util.spec_from_file_location("jira_exporter_definition", JIRA_DEFINITION_FILE)
        jira_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(jira_module)
        JiraTicket = getattr(jira_module, "JiraTicket", None)
        JiraConfig = getattr(jira_module, "JiraConfig", None)
        JiraClient = getattr(jira_module, "JiraClient", None)
        JiraExporter = getattr(jira_module, "JiraExporter", None)
    except Exception as e:
        print(f"Warning: Failed to load JIRA-exporter definition module: {e}")



def convert_jira_ticket_to_document(ticket) -> Document:
    """
    Converts a JiraTicket instance to a LangChain Document.
    Strictly uses ONLY summary and description for the searchable page_content.
    Stores all other attributes in metadata.
    """
    summary = getattr(ticket, "summary", "") or ""
    description = getattr(ticket, "description", "") or ""
    
    # Page content strictly built from summary & description ONLY
    page_content = f"Summary: {summary}\nDescription: {description}".strip()

    metadata = {
        "key": getattr(ticket, "key", ""),
        "status": getattr(ticket, "status", ""),
        "priority": getattr(ticket, "priority", ""),
        "issue_type": getattr(ticket, "issue_type", ""),
        "project": getattr(ticket, "project", ""),
        "components": getattr(ticket, "components", []),
        "fix_versions": getattr(ticket, "fix_versions", []),
        "reporter": getattr(ticket, "reporter", ""),
        "assignee": getattr(ticket, "assignee", ""),
    }

    return Document(page_content=page_content, metadata=metadata)


def get_mock_jira_tickets() -> List:
    """Returns sample mock JiraTicket objects for offline testing and PoC."""
    if JiraTicket is None:
        return []

    return [
        JiraTicket(
            key="KANZI-101",
            id=101,
            summary="Qdrant vector store connection timeout during high concurrency",
            description=(
                "Error log: qdrant_client.http.exceptions.ResponseHandlingException: "
                "Timed out waiting for connection on port 6333.\n"
                "Root cause: Default connection pool size was set to 5 which exhausted under high load.\n"
                "Fix: Increased max_connections to 50 in vector_db.py and enabled keepalive."
            ),
            status="Closed",
            priority="High",
            issue_type="Bug",
            project="KANZI",
            reporter="QA Engineer",
            assignee="Senior Developer",
            created="2026-07-20T10:00:00Z",
            updated="2026-07-21T15:30:00Z",
            components=["VectorDB", "Backend"],
            fix_versions=["v1.2.0"]
        ),
        JiraTicket(
            key="KANZI-102",
            id=102,
            summary="Gemini API Key missing or invalid HTTP 401 Unauthorized error",
            description=(
                "Error log: google.api_core.exceptions.Unauthenticated: 401 API key not valid.\n"
                "Root cause: GEMINI_API_KEY environment variable was not loaded properly in Docker container.\n"
                "Fix: Updated Kubernetes Secrets manifest to inject GEMINI_API_KEY into backend pod env vars."
            ),
            status="Closed",
            priority="Critical",
            issue_type="Bug",
            project="KANZI",
            reporter="DevOps Engineer",
            assignee="Backend Developer",
            created="2026-07-18T09:00:00Z",
            updated="2026-07-19T11:20:00Z",
            components=["Authentication", "Backend"],
            fix_versions=["v1.2.1"]
        ),
        JiraTicket(
            key="KANZI-103",
            id=103,
            summary="Streamlit Gateway UI CORS error when fetching RAG API response",
            description=(
                "Error log: Access to fetch at 'http://localhost:8000' from origin 'http://localhost:8080' "
                "has been blocked by CORS policy.\n"
                "Root cause: FastAPI backend middleware did not allow origin http://localhost:8080.\n"
                "Fix: Added CORSMiddleware to FastAPI app in main.py with allow_origins=['*']."
            ),
            status="Resolved",
            priority="Medium",
            issue_type="Bug",
            project="KANZI",
            reporter="Frontend Lead",
            assignee="Gateway Developer",
            created="2026-07-15T14:00:00Z",
            updated="2026-07-16T16:45:00Z",
            components=["Gateway", "UI"],
            fix_versions=["v1.3.0"]
        )
    ]

