# Ollama Web Search & Vector RAG Backend

A modular, stateless local Retrieval-Augmented Generation (RAG) backend utilizing a local Ollama instance, Chroma Vector Database for local document storage, and DuckDuckGo for public internet search context.

## Architecture & Logic Flow

The system operates as a FastAPI web application. When a query is received, the Ollama model is invoked with tool-calling capabilities. It dynamically decides whether it needs to query the local vector database, perform a public web search, or answer directly.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as FastAPI Server (main.py)
    participant Ollama as Local Ollama LLM
    participant VectorStore as Chroma Vector DB
    participant Search as DuckDuckGo Search

    User->>App: POST /query {"message": "...", "history": [...]}
    App->>Ollama: Check query context & tools
    
    alt Path A: Needs Local Document Context
        rect rgb(224, 242, 254)
            note right of App: Tool call: retrieve_local_documents
            Ollama-->>App: Tool call request (retrieve_local_documents)
            App->>VectorStore: Search documents (k=5)
            VectorStore-->>App: Raw document chunks
            note right of App: Apply heuristic re-ranking (token overlap)
            App->>Ollama: Prompt with top 2 re-ranked context chunks
            Ollama-->>App: Final answer text
        end
    else Path B: Needs Real-Time Public Context
        rect rgb(219, 234, 254)
            note right of App: Tool call: web_search
            Ollama-->>App: Tool call request (web_search)
            App->>Search: invoke(query)
            Search-->>App: Search results (Web context)
            App->>Ollama: Prompt with search results context
            Ollama-->>App: Final answer text
        end
    else Path C: Can Answer Directly
        rect rgb(220, 252, 231)
            note right of App: Direct generation
            Ollama-->>App: Direct answer text
        end
    end
    
    App-->>User: Response {"response": "...", "tool_calls_executed": [...]}
```