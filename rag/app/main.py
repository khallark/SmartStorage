from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import TypedDict, Literal

load_dotenv()   # before anything creates an OpenAI client

from rag_engine.embeddings import get_embeddings
from rag_engine.vectorstore import get_vector_store
from rag_engine.llm import get_llm
from rag_engine.pipeline import RAGPipeline
from rag_engine.config import REWRITE_MODEL

class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str

class QueryRequest(BaseModel):
    history: list[HistoryMessage] | None = Field(
        default=None,
        max_length=10
    )
    question: str = Field(min_length=1, max_length=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_vector_store(get_embeddings())
    
    if store._collection.count() == 0:
        raise RuntimeError("Vector store is empty. Run: python -m rag_engine.ingest")
    
    app.state.pipeline = RAGPipeline(store, get_llm(), rewriter_llm=get_llm(REWRITE_MODEL))   # created ONCE
    yield


app = FastAPI(title="My AI RAG API", lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "online", "message": "FastAPI server initialized successfully!"}

@app.get("/health")
async def health():
    return {"health": "OK"}

@app.post("/question")
async def answer_query(data: QueryRequest, request: Request):
    answer = await request.app.state.pipeline.answer(data.question, data.history)
    return answer
        
# "What was NVIDIA's first graphics accelerator called?"
# "Which company did NVIDIA acquire to enter the mobile processor market?"
# "What was Microsoft's first hardware product release?"
# "How much did Microsoft pay to acquire GitHub?"
# "In what year did Tesla begin production of the Roadster?"
# "Who succeeded Ze'ev Drori as CEO in October 2008?"
# "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# "What was the original name of Microsoft before it became Microsoft?"