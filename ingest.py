"""
Loads all .md notes from VAULT_PATH, splits them into chunks, embeds them
locally (free, no API cost) with sentence-transformers, and stores the
result in a local Chroma vector index at CHROMA_DB_PATH.

Run this once up front, and again any time your notes change:
    python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

VAULT_PATH = os.getenv("VAULT_PATH", "sample_vault")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_index():
    print(f"Loading notes from: {VAULT_PATH}")
    loader = DirectoryLoader(
        VAULT_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} notes.")

    if not docs:
        raise SystemExit(
            f"No .md files found in '{VAULT_PATH}'. Point VAULT_PATH at your vault."
        )

    # Tag each chunk with its note title so answers can cite sources
    for doc in docs:
        filename = os.path.basename(doc.metadata.get("source", "unknown.md"))
        doc.metadata["note_title"] = filename.replace(".md", "")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    print(f"Embedding with {EMBEDDING_MODEL} (runs locally, first call downloads the model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Writing vector index to: {CHROMA_DB_PATH}")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    print("Done. Index is ready for querying via main.py")


if __name__ == "__main__":
    build_index()
