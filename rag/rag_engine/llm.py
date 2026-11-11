from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from .config import LLM_MODEL

def get_llm(model: str = LLM_MODEL) -> ChatOpenAI:
    return ChatOpenAI(model=model)

REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into standalone questions.
Given a conversation and a follow-up question, rewrite the follow-up so it can be understood without the conversation.
- Do NOT answer the question.
- If it is already standalone, return it unchanged.
- If it switches to a new topic, do not carry over the old topic.
- Output only the rewritten question, nothing else."""

def create_rewrite_input(question: str, history: list[dict]) -> str:
    conversation = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
    return f"""Conversation:
{conversation}

Follow-up question: {question}

Standalone question:"""


def to_messages(history: list[dict]):
    """Convert plain history dicts into LangChain message objects."""
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in history
    ]

def create_input(query: str, relevant_docs: list[Document]):
    return f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""