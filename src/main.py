import os
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Import configuration
from src.config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, HOST, PORT

# Import modular components
import src.vector_db as db
from src.tools import retrieve_local_documents, web_search, search_tool
from src.models import MessageSchema, QueryRequest, QueryResponse, IngestRequest, IngestResponse

app = FastAPI(title="Stateless Ollama Vector RAG Backend")

# Initialize LLM & Tool binding
llm = None
llm_with_tools = None

try:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=OLLAMA_TEMPERATURE)
    llm_with_tools = llm.bind_tools([retrieve_local_documents, web_search])
except Exception as e:
    print(f"Error initializing ChatOllama model: {e}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    try:
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [1/1] [{os.path.basename(__file__)}] Forwarding ingestion request to Vector Store\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        chunk_count = db.add_document_text(request.text, request.metadata)
        return IngestResponse(status="success", chunk_count=chunk_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    if llm is None or llm_with_tools is None:
        raise HTTPException(
            status_code=500, 
            detail="ChatOllama LLM component is not initialized."
        )
    
    tool_calls_executed = []
    fallback_triggered = False
    
    try:
        # Step 1/4: Payload mapping
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [1/4] [{os.path.basename(__file__)}] Parsing request and history payload\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        
        messages = [
            SystemMessage(content=(
                "You are an assistant with access to a local vector document database and a web search tool.\n"
                "You must route your query processing through one of these actions:\n"
                "- If the query refers to private documentation, workspace context, or internal topics (like 'Supernova'), you MUST call 'retrieve_local_documents'.\n"
                "- If the query refers to real-time, current events, weather, or active public news, you MUST call 'web_search'.\n"
                "- Otherwise (greetings, general math, standard facts, code snippets), answer directly without any tool calls."
            ))
        ]
        
        # Hydrate stateless message history
        for msg in request.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                messages.append(SystemMessage(content=msg.content))
                
        # Append current user query
        messages.append(HumanMessage(content=request.message))
        
        # Step 2/4: Agent Routing Decision
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [2/4] [{os.path.basename(__file__)}] Invoking ChatOllama to determine routing paths\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        response = llm_with_tools.invoke(messages)
        
        # Step 3/4: Tool Execution & Safeguards
        print(f"\n\033[1;96m========================================================\033[0m")
        print(f"\033[1;92m>>> [3/4] [{os.path.basename(__file__)}] Executing routing tools with safeguards\033[0m")
        print(f"\033[1;96m========================================================\033[0m\n")
        
        if response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Safeguard 1: Hallucinated / Invalid Tool Name check
                if tool_name not in ["retrieve_local_documents", "web_search"]:
                    print(f"Warning: Hallucinated tool call '{tool_name}' detected. Triggering safeguard fallback.")
                    fallback_triggered = True
                    continue
                    
                tool_calls_executed.append(tool_name)
                
                # Safeguard 2: Parameter parsing safety
                q_val = (
                    tool_args.get("query")
                    or tool_args.get("input")
                    or list(tool_args.values())[0]
                    if isinstance(tool_args, dict) and tool_args
                    else str(tool_args)
                )
                
                # Run actual tool
                try:
                    if tool_name == "retrieve_local_documents":
                        tool_output = retrieve_local_documents.invoke(q_val)
                        
                        # Safeguard 3: Similarity score empty gate check
                        if "No matching local documents found." in tool_output or "Error" in tool_output:
                            print("Warning: Local database did not return matches. Falling back to public Web Search.")
                            fallback_triggered = True
                            tool_output = web_search.invoke(q_val)
                            tool_calls_executed.append("web_search")
                    else:
                        tool_output = web_search.invoke(q_val)
                except Exception as tool_err:
                    print(f"Tool execution failed: {tool_err}. Triggering direct fallback.")
                    fallback_triggered = True
                    tool_output = f"Error: Failed execution context fallback: {str(tool_err)}"
                
                tool_message = ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                )
                messages.append(tool_message)
                
            # Step 4/4: Final Synthesis
            print(f"\n\033[1;96m========================================================\033[0m")
            print(f"\033[1;92m>>> [4/4] [{os.path.basename(__file__)}] Synthesizing final answer with tools context\033[0m")
            print(f"\033[1;96m========================================================\033[0m\n")
            final_response = llm_with_tools.invoke(messages)
            response_content = final_response.content
        else:
            # Step 4/4 (Direct Path)
            print(f"\n\033[1;96m========================================================\033[0m")
            print(f"\033[1;92m>>> [4/4] [{os.path.basename(__file__)}] Generating direct LLM response\033[0m")
            print(f"\033[1;96m========================================================\033[0m\n")
            response_content = response.content
            
        print("Query execution completed successfully.\n")
        return QueryResponse(
            response=response_content,
            tool_calls_executed=tool_calls_executed,
            fallback_triggered=fallback_triggered
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

@app.get("/health")
async def health_check():
    vector_ok = "ok" if db.vector_store is not None else "failed"
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "search": "DuckDuckGo",
        "vector_store": vector_ok
    }

if __name__ == "__main__":
    uvicorn.run("src/main:app", host=HOST, port=PORT, reload=True)
