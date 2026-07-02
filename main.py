"""
VaultQA API — a LangChain RAG service over your Obsidian vault, exposed
as a plain HTTP API so Make.com (or anything else) can call it.

Endpoints:
    POST /ask       { "question": "..." }  -> answer + cited sources
    POST /reindex   (re-runs ingestion, use after adding/editing notes)
    GET  /health    simple liveness check

Auth: every request must include header  X-API-Key: <VAULTQA_API_KEY>

Run locally:
    uvicorn main:app --reload

Deploy: push this repo to Railway / Render / Fly.io, set the env vars
from .env.example in their dashboard, and use the public URL it gives
you as the HTTP module target in Make.com.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

API_KEY = os.getenv("VAULTQA_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="VaultQA", description="RAG over an Obsidian vault")

# Loaded lazily on first request so the app boots fast (important for
# platforms like Railway that health-check on startup).
_retriever = None
_chain = None


def get_chain():
    global _retriever, _chain
    if _chain is not None:
        return _chain

    if not os.path.exists(CHROMA_DB_PATH):
        raise RuntimeError(
            f"No index found at '{CHROMA_DB_PATH}'. Run `python ingest.py` first, "
            "or call POST /reindex."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1000)

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant answering questions using ONLY the "
        "notes provided below. If the notes don't contain the answer, say so "
        "clearly rather than guessing.\n\n"
        "NOTES:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "Answer concisely, grounded in the notes above."
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[{d.metadata.get('note_title', 'unknown')}]\n{d.page_content}" for d in docs
        )

    _chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return _chain


class Question(BaseModel):
    question: str


def check_auth(x_api_key: str | None):
    if not API_KEY:
        return  # no key configured -> auth disabled (fine for local dev only)
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reindex")
def reindex(x_api_key: str | None = Header(default=None)):
    check_auth(x_api_key)
    from ingest import build_index

    global _retriever, _chain
    _retriever = None
    _chain = None
    build_index()
    return {"status": "reindexed"}


@app.post("/ask")
def ask(query: Question, x_api_key: str | None = Header(default=None)):
    check_auth(x_api_key)

    chain = get_chain()
    retriever = _retriever  # populated by get_chain()

    sources = retriever.invoke(query.question)
    answer = chain.invoke(query.question)

    return {
        "answer": answer,
        "sources": sorted(set(d.metadata.get("note_title", "unknown") for d in sources)),
    }
