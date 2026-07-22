import logging
from langchain_core.messages import SystemMessage, HumanMessage
from src.theme_based_rag_backend.agent_flow.state import AgentState
from src.theme_based_rag_backend.vector_db import search_jira_tickets, add_jira_tickets
from src.theme_based_rag_backend.jira_exporter_adapter import get_mock_jira_tickets

logger = logging.getLogger(__name__)


def jira_bug_qa_node(state: AgentState) -> dict:
    """
    Executes Qdrant Hybrid Search (Dense Gemini + FastEmbed BM25 Sparse)
    over indexed JIRA tickets (summary & description) and generates structured
    horizontal bug repair advice (横展修復提案).
    """
    from src.theme_based_rag_backend.agent_flow import llm

    query = state["message"]

    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Agent Flow] Processing JIRA Bug Analysis Node\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")

    # 1. Search Qdrant for matching JIRA tickets
    try:
        tickets = search_jira_tickets(query, k=3)
        if not tickets:
            print("No indexed JIRA tickets found in Qdrant. Seeding mock tickets...")
            add_jira_tickets(get_mock_jira_tickets())
            tickets = search_jira_tickets(query, k=3)
    except Exception as e:
        logger.warning(f"Qdrant search error ({e}). Falling back to mock dataset...")
        mock_tickets = get_mock_jira_tickets()
        tickets = [
            {
                "content": f"Summary: {t.summary}\nDescription: {t.description}",
                "metadata": {"key": t.key, "components": t.components, "status": t.status}
            }
            for t in mock_tickets
        ]

    print(f"Retrieved {len(tickets)} matching JIRA bug ticket(s).")

    # 2. Format context for LLM
    context_blocks = []
    for i, t in enumerate(tickets, 1):
        meta = t.get("metadata", {})
        key = meta.get("key", f"JIRA-{i}")
        status = meta.get("status", "Closed")
        components = ", ".join(meta.get("components", []))
        context_blocks.append(
            f"### Ticket [{key}] (Status: {status}, Components: {components})\n"
            f"{t.get('content', '')}\n"
        )

    context_str = "\n".join(context_blocks)

    # 3. Construct LLM System Prompt for Horizontal Repair Advice
    system_prompt = (
        "You are an expert AI Software Engineer assisting with 'Horizontal Bug Repair & Development' (横展開発サポート).\n"
        "Analyze the user's reported bug/error alongside the retrieved historical JIRA tickets provided below.\n\n"
        "Retrieved Historical JIRA Tickets:\n"
        f"{context_str}\n\n"
        "Generate a structured, professional Japanese response containing:\n"
        "1. **根本原因分析 (Root Cause Analysis)**: Summarize why the reported error occurs based on historical cases.\n"
        "2. **類似JIRAチケット参照 (Similar Historical Tickets)**: List matched tickets (Key, Summary, Status).\n"
        "3. **横展展開提案 (Horizontal Scope & Actionable Fix)**: Recommend specific code changes or configuration fixes to prevent this bug elsewhere in the project.\n"
        "Keep the advice practical, precise, and directly pasteable."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Reported Bug / Query: {query}")
    ]

    try:
        response = llm.invoke(messages)
        draft = response.content
        if isinstance(draft, list):
            draft = "".join(part if isinstance(part, str) else part.get("text", "") for part in draft)
    except Exception as e:
        logger.error(f"Error generating JIRA bug repair advice: {e}")
        draft = "Failed to generate horizontal repair advice. Please check API configuration."

    return {
        "jira_tickets": tickets,
        "horizontal_repair_advice": draft,
        "draft_response": draft
    }
