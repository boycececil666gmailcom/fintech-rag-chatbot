from typing import List, Optional, Dict
from pydantic import BaseModel

class MessageSchema(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class QueryRequest(BaseModel):
    message: str
    history: Optional[List[MessageSchema]] = []

class QueryResponse(BaseModel):
    response: str
    tool_calls_executed: List[str] = []
    fallback_triggered: bool = False

class IngestRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, str]] = None

class IngestResponse(BaseModel):
    status: str
    chunk_count: int
