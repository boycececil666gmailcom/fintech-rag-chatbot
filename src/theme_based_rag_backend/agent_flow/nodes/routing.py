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
            f"Your task is to classify whether a user query is related to the theme: '{CHATBOT_THEME}'.\n"
            f"Queries referencing the following proprietary or theme-specific keywords should also be routed as relevant: {keywords_str}.\n\n"
            f"Classification Criteria:\n"
            f"- 'rag': The query is related to '{CHATBOT_THEME}', financial technology, or mentions any of the theme-specific keywords ({keywords_str}).\n"
            f"- 'refuse': The query is completely unrelated to the theme.\n\n"
            f"Output exactly 'rag' or 'refuse'. Do not include any other explanation, text, or punctuation."
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
        if "rag" in category:
            category = "rag"
        elif "refuse" in category:
            category = "refuse"
        else:
            # Fallback parsing in case the LLM outputs something else
            category = "refuse"
            
    except Exception as e:
        logger.error(f"Error during LLM classification: {e}. Falling back to 'refuse'.")
        category = "refuse"
        
    print(f"-> Categorized as: '{category}'")
    return {"category": category}
