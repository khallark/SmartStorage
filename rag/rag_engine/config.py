from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
PERSIST_DIR = BASE_DIR / "db" / "chroma_db"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
REWRITE_MODEL = "gpt-4o-mini"
MAX_HISTORY_MESSAGES = 6
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DOCS_GLOB = "*.txt"
TOP_K = 5