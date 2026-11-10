from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from .config import LLM_MODEL

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL)

def create_input(query: str, relevant_docs: list[Document]):
    return f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""