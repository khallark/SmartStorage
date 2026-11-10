from langchain_chroma import Chroma
from .config import PERSIST_DIR

def get_vector_store(embeddings) -> Chroma:
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
