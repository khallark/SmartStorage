from langchain_openai import OpenAIEmbeddings
from .config import EMBEDDING_MODEL

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)