# Fintech RAG Chatbot

A modular, stateless local Retrieval-Augmented Generation (RAG) customer service chatbot for a Fintech SaaS platform utilizing a local Ollama instance and a Chroma Vector Database for local document storage.

## Business & Product Flow (Overview)

Below is a simplified view of how information flows through the Fintech RAG Chatbot system, designed for product managers and operations:

```mermaid
flowchart TD
    %% Styling Node classes
    classDef client fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef router fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#b45309;
    classDef kb fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8;
    classDef reply fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#166534;
    classDef block fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#991b1b;
    classDef process fill:#f9fafb,stroke:#d1d5db,stroke-width:1px,color:#374151;

    %% Main customer interaction entry point
    Client(["📱 Client App / Customer"])
    
    Client -->|"1. Submits chat query"| Router{"🤖 LLM Router (Ollama)"}
    
    %% Router checks relevance
    Router -->|"2. Checks query topic"| IsFintech{"Is it a Fintech SaaS query?"}
    
    %% Safeguard Refusal Path (Left branch)
    IsFintech -->|"No (General Query)"| Safeguard["🛡️ Refusal Safeguard"]
    Safeguard -->|"Blocks answering from LLM memory"| RefusalReply["Polite Decline Answer"]
    RefusalReply -->|"Returns response"| Client
    
    %% RAG Retrieval Path (Right branch)
    IsFintech -->|"Yes (Fintech Query)"| Search["🔍 Search internal knowledge base"]
    
    %% Ingestion background flow
    subgraph Ingestion ["Knowledge Ingestion (Offline Feed)"]
        Admin(["Product / Ops Admin"]) -->|"Uploads FAQs & Guides"| DB[("📚 Knowledge Base (Chroma DB)")]
    end
    
    Search -->|"Fetch context chunks"| DB
    DB -->|"Returns factual source text"| Synth["✏️ Synthesis Engine"]
    Synth -->|"Generates grounded response"| VerifiedReply["Verified Platform Answer"]
    
    VerifiedReply -->|"Returns response"| Client

    %% Class assignments
    class Client,Admin client;
    class Router,IsFintech router;
    class DB kb;
    class Search,Synth process;
    class VerifiedReply reply;
    class Safeguard,RefusalReply block;
```

## Features & API Endpoints

The backend exposes two main HTTP POST endpoints under FastAPI:

- **`POST /ingest`**: Accepts raw text documents, splits them into manageable chunks, generates vector embeddings, and stores them in the local Chroma database.
- **`POST /query`**: Accepts user queries and history. An LLM agent routes queries to retrieve platform documentation from the local database. If a query does not trigger retrieval, the direct pathway is refused to ensure responses are fully grounded in the local database.

---

## Architecture & Logic Flow

Below is a high-level flowchart showing how ingestion and querying are routed through the FastAPI backend:

```mermaid
graph TD
    Server[FastAPI Server]
    
    %% Ingest path
    Server -->|POST /ingest| Ingest[Document Ingestion Path]
    Ingest --> Split[RecursiveCharacterTextSplitter]
    Split --> Embed[Ollama Embeddings]
    Embed --> DB[(Chroma Vector DB)]

    %% Query path
    Server -->|POST /query| Query[Query Processing Path]
    Query --> Router{Ollama Agent Router}
    Router -->|Local Context| Local[retrieve_local_documents Tool]
    Router -->|Direct Generation Refused| Direct[Refusal Response]
    
    Local --> DenseSearch[1. Dense Search: Chroma]
    Local --> GetDocs[2. Get All Documents]
    GetDocs --> BM25Search[3. Sparse Search: BM25]
    DenseSearch --> RRF[4. Reciprocal Rank Fusion - RRF]
    BM25Search --> RRF
    RRF --> Rerank[5. Rerank: FlashRank Cross-Encoder]
    Rerank --> Synth[Final Synthesis]
```

### 1. Ingestion Path

The ingestion pipeline converts plain text into queryable semantic chunks inside the Chroma Vector Database.

```mermaid
sequenceDiagram
    autonumber
    actor Client as Client / Ingestion Script
    participant App as FastAPI Server (main.py)
    participant VectorStore as Chroma Vector DB

    Client->>App: POST /ingest {"text": "...", "metadata": {...}}
    Note over App: Chunks text using<br/>RecursiveCharacterTextSplitter
    App->>VectorStore: Add document chunks with embeddings
    VectorStore-->>App: Confirmation
    App-->>Client: Response {"status": "success", "chunk_count": X}
```

### 2. Query Path

When a query is received, the Ollama model is invoked with tool-calling capabilities. It dynamically decides whether it needs to query the local vector database for platform facts or answer directly.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as FastAPI Server (main.py)
    participant BM25 as BM25Retriever (LangChain)
    participant RRF as RRF Module (rrf.py)
    participant Reranker as FlashrankRerank (LangChain)
    participant Ollama as Local Ollama LLM
    participant VectorStore as Chroma Vector DB

    User->>App: POST /query {"message": "...", "history": [...]}
    App->>Ollama: Check query context & tools
    
    alt Path A: Needs Local Document Context
        rect rgb(224, 242, 254)
            note right of App: Tool call: retrieve_local_documents
            Ollama-->>App: Tool call request (retrieve_local_documents)
            
            App->>VectorStore: Get dense semantic results (k=10)
            VectorStore-->>App: Semantic documents + distance
            
            App->>BM25: BM25Retriever.invoke(query)
            BM25-->>App: Sorted sparse documents
            
            App->>RRF: reciprocal_rank_fusion(dense, sparse)
            RRF-->>App: Top 5 fused documents
            
            App->>Reranker: FlashrankRerank.compress_documents(fused_docs, query)
            Reranker-->>App: Top-2 compressed/reranked documents
            
            App->>Ollama: Prompt with top 2 reranked context chunks
            Ollama-->>App: Final answer text
        end
    else Path B: Direct Generation Refused
        rect rgb(254, 226, 226)
            note right of App: Refusal path
            App-->>User: Refusal response (Pre-trained answering disabled)
        end
    end
    
    App-->>User: Response {"response": "...", "tool_calls_executed": [...]}
```