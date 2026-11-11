from langchain_core.messages import HumanMessage, SystemMessage
from .config import DOCS_DIR, TOP_K, MAX_HISTORY_MESSAGES
from .loaders import load_documents
from .chunking import split_documents
from .llm import create_input, create_rewrite_input, to_messages, REWRITE_SYSTEM_PROMPT

class RAGPipeline:
    def __init__(self, vector_store, llm, rewriter_llm=None, top_k: int = TOP_K):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        self.llm = llm
        self.rewriter_llm = rewriter_llm or llm
    
    
    def ingest(self, docs_path=DOCS_DIR):
        documents = load_documents(docs_path)
        chunks = split_documents(documents)
        self.vector_store.add_documents(chunks)
        return len(chunks)


    async def rewrite(self, question: str, history: list[dict]) -> str:
        messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=create_rewrite_input(question, history))
        ]
        result = await self.rewriter_llm.ainvoke(messages)
        return result.content.strip()


    async def answer(self, question: str, history: list[dict] | None = None):
        history = (history or [])[-MAX_HISTORY_MESSAGES:]
        search_query = await self.rewrite(question, history) if history else question
        relevant_docs = await self.retriever.ainvoke(search_query)
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            # *to_messages(history),
            HumanMessage(content=create_input(search_query, relevant_docs)),
        ]
        result = await self.llm.ainvoke(messages)
        return {
            "answer": result.content,
            "question": search_query
        }