# Theme-Based RAG Workflow

A modular, stateless Retrieval-Augmented Generation (RAG) customer service chatbot utilizing the Google Gemini API, LangGraph agent orchestration, and Qdrant for document vector storage.

---

## 1. Executive Summary & Technology Stack

### Business Overview
The **Theme-Based RAG Workflow** is an enterprise-grade customer service chatbot system designed to deliver strictly grounded, accurate responses while preventing off-topic queries and AI hallucinations. 

- **Topic Boundaries**: Automatically enforces business domain boundaries (e.g., Fintech SaaS platform documentation) by routing off-theme queries to a dedicated refusal engine.
- **Self-Correcting Groundedness Verification**: Incorporates a self-critique agent loop that evaluates answer candidates against retrieved context before presenting them to customers.
- **Enterprise Ingestion Pipeline**: Ingests company documentation into a high-performance vector store with hybrid dense (semantic) and sparse (keyword) indexing.

### Technical Overview
Built on a microservice architecture separating an API Gateway proxy (`theme_based_rag_gateway`) from the core RAG execution engine (`theme_based_rag_backend`). The system utilizes LangGraph for state management, combining hybrid Qdrant search with FlashRank neural reranking for passage retrieval.

### Technology Stack & Dependencies

* ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) **Python 3.10+**: Core programming environment and runtime.
* ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) **FastAPI**: Asynchronous web framework used for the API Gateway and backend services.
* ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white) **LangGraph**: Framework for orchestrating stateful, multi-node agent loops, conditional routing, and critique workflows.
* ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white) **LangChain**: Framework for text chunking (`RecursiveCharacterTextSplitter`), document abstractions, and model integrations.
* ![Google Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google-gemini&logoColor=white) **Google Gemini API**: Powers LLM decision-making (`gemini-3.1-flash-lite`) and dense text embeddings (`gemini-embedding-001`).
* ![Qdrant](https://img.shields.io/badge/Qdrant-DC2626?style=for-the-badge&logo=qdrant&logoColor=white) **Qdrant Vector DB**: Vector store supporting hybrid dense-sparse retrieval and payload filtering.
* ![FastEmbed](https://img.shields.io/badge/FastEmbed_BM25-FF6F00?style=for-the-badge&logo=python&logoColor=white) **FastEmbed BM25**: Fast lexical embedding engine for sparse keyword matching (`Qdrant/bm25`).
* ![FlashRank](https://img.shields.io/badge/FlashRank-000000?style=for-the-badge&logo=lightning&logoColor=white) **FlashRank**: Ultra-fast neural reranking model used to rerank retrieved document passages.
* ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) **Docker**: Containerization infrastructure for Qdrant and microservice deployment.
* ![Uvicorn](https://img.shields.io/badge/Uvicorn-4053D6?style=for-the-badge&logo=python&logoColor=white) **Uvicorn**: Production-ready ASGI server implementation powering FastAPI endpoints.
* ![HTTPX](https://img.shields.io/badge/HTTPX-5B60EA?style=for-the-badge&logo=python&logoColor=white) **HTTPX**: Asynchronous HTTP client powering the gateway proxy routing layer.

---

## 2. Business Flow Overview

Below is a simplified operational workflow designed for business managers and product stakeholders, illustrating how customer queries and knowledge base updates move through the system without technical jargon:

```mermaid
flowchart TD
    %% Styling Node classes
    classDef client fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef router fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#b45309;
    classDef kb fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px,color:#6b21a8;
    classDef reply fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#166534;
    classDef block fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#991b1b;
    classDef process fill:#f9fafb,stroke:#d1d5db,stroke-width:1px,color:#374151;

    %% Client App / Customer Entry
    Customer(["📱 Customer / App User"]):::client
    
    Customer -->|"1. Submits customer support question"| TopicCheck{"🤖 AI Intent Classifier<br/>(Check if query matches business domain)"}:::router
    
    %% Routing Decision Branches
    TopicCheck -->|"Topic matches business domain"| SearchKB["🔍 Search Company Knowledge Base"]:::process
    TopicCheck -->|"Topic outside business domain"| DraftRefusal["🛡️ Prepare Polite Refusal Message"]:::block
    
    %% Knowledge Base Search
    SearchKB -->|"Retrieve context documents"| KnowledgeBase[("📚 Knowledge Base Vector Store")]:::kb
    KnowledgeBase -->|"Return matching document passages"| GenerateAnswer["✍️ Draft Answer Using Document Context"]:::process
    
    %% Quality Control Check
    GenerateAnswer -->|"2. Draft answer ready"| QualityCheck{"🔎 Quality & Groundedness Checker<br/>(Verify zero hallucination)"}:::router
    DraftRefusal -->|"Refusal draft ready"| QualityCheck
    
    %% Quality Outcomes
    QualityCheck -->|"Passes verification check"| FinalAnswer["✅ Verified Helpful Answer"]:::reply
    QualityCheck -->|"Fails verification (Information unverified)"| TopicCheck
    
    FinalAnswer -->|"3. Send final answer back to customer"| Customer
    
    %% Ingestion background flow
    subgraph DocumentIngestion ["Document Ingestion Flow (Offline Updates)"]
        style DocumentIngestion fill:#f9fafb,stroke:#d1d5db,stroke-width:1px;
        OpsAdmin(["👤 Operations / Content Admin"]):::client -->|"Uploads new FAQs & User Guides"| TextSplitter["✂️ Break Documents into Small Passages"]:::process
        TextSplitter -->|"Generate Dense Semantic & Keyword Embeddings"| KnowledgeBase
    end
```

---

## 3. Technical System Architecture

Below is a detailed component architecture diagram illustrating the internal modules, state graphs, data structures, and interactions between `theme_based_rag_gateway` and `theme_based_rag_backend`:

```mermaid
flowchart TD
    %% Styling
    classDef gateway fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e40af;
    classDef backend fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#166534;
    classDef graphNode fill:#fdf4ff,stroke:#c026d3,stroke-width:2px,color:#86198f;
    classDef dbNode fill:#fff7ed,stroke:#ea580c,stroke-width:2px,color:#9a3412;
    classDef modelNode fill:#f8fafc,stroke:#64748b,stroke-width:1px,color:#334155;

    subgraph GatewayModule ["API Gateway Service (src.theme_based_rag_gateway)"]
        style GatewayModule fill:#f8fafc,stroke:#94a3b8,stroke-width:1px;
        GatewayMain["main.py (FastAPI App)"]:::gateway
        RouteQuery["route_query(QueryRequest)"]:::gateway
        RouteIngest["route_ingest(IngestRequest)"]:::gateway
        HTTPXClient["httpx.AsyncClient"]:::gateway
        
        GatewayMain --> RouteQuery
        GatewayMain --> RouteIngest
        RouteQuery --> HTTPXClient
        RouteIngest --> HTTPXClient
    end

    subgraph BackendModule ["Core RAG Backend Service (src.theme_based_rag_backend)"]
        style BackendModule fill:#f8fafc,stroke:#94a3b8,stroke-width:1px;
        BackendMain["main.py (FastAPI App)"]:::backend
        RunQuery["run_query(QueryRequest)"]:::backend
        IngestDoc["ingest_document(IngestRequest)"]:::backend
        
        BackendMain --> RunQuery
        BackendMain --> IngestDoc
        
        subgraph VectorStoreModule ["Vector DB Pipeline (vector_db.py)"]
            style VectorStoreModule fill:#fff7ed,stroke:#fdba74,stroke-width:1px;
            GetVS["get_vector_store()"]:::dbNode
            AddDocText["add_document_text(text, metadata)"]:::dbNode
            Splitter["RecursiveCharacterTextSplitter"]:::dbNode
            DenseEmbed["GoogleGenerativeAIEmbeddings<br/>(gemini-embedding-001)"]:::dbNode
            SparseEmbed["FastEmbedSparse<br/>(Qdrant/bm25)"]:::dbNode
            QdrantStore["QdrantVectorStore<br/>(collection: local_rag_documents)"]:::dbNode
            
            AddDocText --> Splitter
            Splitter --> QdrantStore
            GetVS --> DenseEmbed
            GetVS --> SparseEmbed
            GetVS --> QdrantStore
        end
        
        subgraph LangGraphAgent ["Agent Execution Graph (agent_flow/graph.py)"]
            style LangGraphAgent fill:#fdf4ff,stroke:#f0abfc,stroke-width:2px;
            State["AgentState (TypedDict)"]:::graphNode
            CompiledGraph["agent_graph (Compiled StateGraph)"]:::graphNode
            
            RoutingNode["routing_node(AgentState)"]:::graphNode
            RAGQANode["rag_qa_node(AgentState)"]:::graphNode
            RefusalNode["refusal_node(AgentState)"]:::graphNode
            CritiqueNode["critique_node(AgentState)"]:::graphNode
            
            RouteEdge["route_by_category(AgentState)"]:::graphNode
            CritiqueEdge["route_after_critique(AgentState)"]:::graphNode
            
            RetrieveTool["tools.retrieve_local_documents"]:::graphNode
            FlashRankRerank["flashrank.Ranker"]:::graphNode

            CompiledGraph --> RoutingNode
            RoutingNode --> RouteEdge
            RouteEdge -->|"rag"| RAGQANode
            RouteEdge -->|"refuse"| RefusalNode
            RAGQANode --> RetrieveTool
            RetrieveTool --> GetVS
            RetrieveTool --> FlashRankRerank
            RAGQANode --> CritiqueNode
            RefusalNode --> CritiqueNode
            CritiqueNode --> CritiqueEdge
            CritiqueEdge -->|"approved"| ENDNode([END]):::graphNode
            CritiqueEdge -->|"rejected"| RoutingNode
        end
    end

    HTTPXClient -->|"POST /query"| RunQuery
    HTTPXClient -->|"POST /ingest"| IngestDoc
    RunQuery --> CompiledGraph
    IngestDoc --> AddDocText
```

---

## 4. Technical Sequence & Business Logic Execution

Below is a sequence diagram detailing the end-to-end execution flow across exact class names, function calls, and data transitions, highlighting conditional logic branches with colorized alternative blocks:

```mermaid
sequenceDiagram
    autonumber
    actor Client as Client / Mobile App
    participant Gateway as src.theme_based_rag_gateway.main
    participant Backend as src.theme_based_rag_backend.main
    participant Graph as agent_flow.graph.agent_graph
    participant Routing as agent_flow.nodes.routing_node
    participant QA as agent_flow.nodes.rag_qa_node
    participant Tool as tools.retrieve_local_documents
    participant DB as vector_db.get_vector_store
    participant Refusal as agent_flow.nodes.refusal_node
    participant Critique as agent_flow.nodes.critique_node
    participant Edges as agent_flow.edges.routing

    Client->>Gateway: POST /query (QueryRequest)
    Gateway->>Backend: httpx.AsyncClient.post("/query", QueryRequest)
    Backend->>Graph: agent_graph.ainvoke(AgentState)
    
    loop Execution Loop (Max 3 attempts)
        Graph->>Routing: routing_node(AgentState)
        Note over Routing: LLM classifies query category ("rag" or "refuse")
        Routing-->>Graph: returns {"category": category}
        
        Graph->>Edges: route_by_category(AgentState)
        
        alt Path A: Category is 'rag' (Domain-related query)
            rect rgb(224, 242, 254)
                Edges-->>Graph: returns "rag"
                Graph->>QA: rag_qa_node(AgentState)
                QA->>Tool: retrieve_local_documents.invoke(query)
                Tool->>DB: QdrantVectorStore.similarity_search_with_score()
                DB-->>Tool: Raw doc passages
                Note over Tool: FlashRank.Ranker.rerank() top passages
                Tool-->>QA: Filtered & Reranked doc context string
                Note over QA: LLM synthesizes answer using retrieved docs
                QA-->>Graph: returns {"draft_response": content, "retrieved_documents": docs}
            end
        else Path B: Category is 'refuse' (Off-theme query)
            rect rgb(254, 226, 226)
                Edges-->>Graph: returns "refuse"
                Graph->>Refusal: refusal_node(AgentState)
                Note over Refusal: LLM generates polite refusal response
                Refusal-->>Graph: returns {"draft_response": content}
            end
        end

        Graph->>Critique: critique_node(AgentState)
        Note over Critique: LLM verifies groundedness & refusal compliance
        
        alt Validation Status: PASS
            rect rgb(220, 252, 231)
                Critique-->>Graph: returns {"critique_feedback": "PASS"}
                Graph->>Edges: route_after_critique(AgentState)
                Edges-->>Graph: returns "approved" -> Exit Loop to END
            end
        else Validation Status: FAIL (And attempts < 3)
            rect rgb(254, 243, 199)
                Critique-->>Graph: returns {"critique_feedback": reason, "attempts": attempts + 1}
                Graph->>Edges: route_after_critique(AgentState)
                Edges-->>Graph: returns "rejected" -> Loop back to routing_node
            end
        end
    end

    Graph-->>Backend: Final AgentState result
    Backend-->>Gateway: QueryResponse(response, tool_calls_executed, retrieved_documents)
    Gateway-->>Client: HTTP 200 OK (QueryResponse JSON)
```