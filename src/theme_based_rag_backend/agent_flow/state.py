from typing import TypedDict, List, Literal, Optional

class AgentState(TypedDict):
    message: str
    history: List[dict]
    category: Literal["rag", "refuse", "jira_bug"]
    retrieved_documents: Optional[str]
    jira_tickets: List[dict]
    horizontal_repair_advice: Optional[str]
    draft_response: str
    critique_feedback: Optional[str]
    attempts: int

