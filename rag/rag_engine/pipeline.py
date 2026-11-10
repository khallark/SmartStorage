from langchain_core.messages import HumanMessage, SystemMessage
from .config import DOCS_DIR, TOP_K
from .loaders import load_documents
from .chunking import split_documents
from .llm import create_input

class RAGPipeline:
    def __init__(self, vector_store, llm, top_k: int = TOP_K):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        self.llm = llm
    
    def ingest(self, docs_path=DOCS_DIR):
        documents = load_documents(docs_path)
        chunks = split_documents(documents)
        self.vector_store.add_documents(chunks)
        return len(chunks)


    async def answer(self, query: str):
        relevant_docs = await self.retriever.ainvoke(query)
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=create_input(query, relevant_docs)),
        ]
        result = await self.llm.ainvoke(messages)
        return {
            "answer": result.content,
            "sources": [
                {"source": doc.metadata.get("source"), "excerpt": doc.page_content[:200]}
                for doc in relevant_docs
            ],
        }