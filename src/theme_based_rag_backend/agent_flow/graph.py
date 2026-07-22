from langgraph.graph import StateGraph, END
from src.theme_based_rag_backend.agent_flow.state import AgentState
from src.theme_based_rag_backend.agent_flow.nodes import (
    routing_node,
    rag_qa_node,
    refusal_node,
    critique_node,
    jira_bug_qa_node
)
from src.theme_based_rag_backend.agent_flow.edges import (
    route_by_category,
    route_after_critique
)

# Workflow Graph Setup
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("routing", routing_node)
workflow.add_node("rag_qa", rag_qa_node)
workflow.add_node("refusal", refusal_node)
workflow.add_node("critique", critique_node)
workflow.add_node("jira_bug_qa", jira_bug_qa_node)

# Set Entry Point and Edges
workflow.set_entry_point("routing")

workflow.add_conditional_edges(
    "routing",
    route_by_category,
    {
        "rag": "rag_qa",
        "refuse": "refusal",
        "jira_bug": "jira_bug_qa"
    }
)

workflow.add_edge("rag_qa", "critique")
workflow.add_edge("refusal", "critique")
workflow.add_edge("jira_bug_qa", "critique")

workflow.add_conditional_edges(
    "critique",
    route_after_critique,
    {
        "approved": END,
        "rejected": "routing"  # Loop back to the start (Routing Node)
    }
)

# Compile Workflow Graph
agent_graph = workflow.compile()
