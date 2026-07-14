import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Ollama settings
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

# FastAPI server settings
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Chroma DB settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
if not os.path.isabs(CHROMA_PERSIST_DIR):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHROMA_PERSIST_DIR = os.path.abspath(os.path.join(project_root, CHROMA_PERSIST_DIR))
