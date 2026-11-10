from dotenv import load_dotenv
from .embeddings import get_embeddings
from .vectorstore import get_vector_store
from .llm import get_llm
from .pipeline import RAGPipeline

if __name__ == "__main__":
    load_dotenv()
    pipeline = RAGPipeline(get_vector_store(get_embeddings()), get_llm())
    print(f"Ingested {pipeline.ingest()} chunks")