import logging
from langchain_core.messages import SystemMessage, HumanMessage
from src.theme_based_rag_backend.config import CHATBOT_THEME, FORCE_RAG_KEYWORDS
from src.theme_based_rag_backend.agent_flow.state import AgentState

logger = logging.getLogger(__name__)

def routing_node(state: AgentState) -> dict:
    from src.theme_based_rag_backend.agent_flow import llm
    
    query = state["message"]
    
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Agent Flow] Classifying user query theme via LLM Agent\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    try:
        # Construct dynamic prompt referencing theme and any forced keywords
        keywords_str = ", ".join(f"'{kw}'" for kw in FORCE_RAG_KEYWORDS)
        
        system_prompt = (
            f"You are a routing agent for a customer service chatbot.\n"
            f"Your task is to classify the user query into one of three categories:\n"
            f"1. 'jira_bug': The query mentions an error log, bug, JIRA ticket, issue, fix, exception, traceback, or requests bug repair/横展分析.\n"
            f"2. 'rag': The query is related to '{CHATBOT_THEME}', financial technology, or mentions any of the theme-specific keywords ({keywords_str}).\n"
            f"3. 'refuse': The query is completely unrelated to the theme or bug analysis.\n\n"
            f"Output exactly 'jira_bug', 'rag', or 'refuse'. Do not include any other explanation, text, or punctuation."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
        
        category = content.strip().lower()
        if "jira" in category or "bug" in category:
            category = "jira_bug"
        elif "rag" in category:
            category = "rag"
        elif "refuse" in category:
            category = "refuse"
        else:
            # Fallback check for bug keywords in query
            if any(kw in query.lower() for kw in ["error", "bug", "exception", "jira", "fail", "log", "issue", "横展"]):
                category = "jira_bug"
            else:
                category = "refuse"
            
    except Exception as e:
        logger.error(f"Error during LLM classification: {e}. Falling back to 'refuse'.")
        category = "refuse"
        
    print(f"-> Categorized as: '{category}'")
    return {"category": category}
