import io
import os
from pathlib import Path
from typing import List

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# python-dotenv reads a local .env file and copies its values into the process's
# environment variables — same job as the `dotenv` npm package. We use it so the
# OpenAI API key lives in a gitignored file on disk, never hardcoded or committed.
load_dotenv()

app = FastAPI()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# Loaded once at startup, not inside the route — loading model weights or opening
# a database connection per-request would repeat that cost on every single upload.
# Same instinct as creating one DB connection pool at app startup in an Express app.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

CHROMA_DIR = Path(__file__).parent / "chroma_db"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection("documents")


class ChromaRetriever(BaseRetriever):
    """Bridges our raw chromadb collection into LangChain's retriever interface."""

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        query_embedding = embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)
        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(results["documents"][0], results["metadatas"][0])
        ]


retriever = ChromaRetriever()

# ChatPromptTemplate.from_template() is a plain string with {placeholders} --
# LangChain fills them in from a dict when the chain runs, the same idea as an
# f-string but deferred until invoke() time instead of built immediately.
# This part needs no API key, so it's safe to build at import time.
prompt = ChatPromptTemplate.from_template(
    "Use the following context to answer the question. "
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_answer_chain = None


def get_answer_chain():
    """Builds the LLM + chain on first use, not at import time.

    ChatOpenAI's constructor eagerly creates OpenAI's real client, which
    validates credentials immediately and raises if no API key is present.
    Building it at module level would crash the *entire app* on startup
    whenever no key is configured -- including routes that have nothing to
    do with OpenAI. Building it lazily, only after /ask has already
    confirmed a key exists, keeps every other route working regardless.
    Cached in _answer_chain after the first call, same "build once" instinct
    as embedding_model and chroma_client above.
    """
    global _answer_chain
    if _answer_chain is None:
        # temperature=0 makes answers deterministic/focused rather than
        # creative -- appropriate for "answer from this context".
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # This is the LCEL pipe replacing RetrievalQA: prompt's filled-in text
        # goes into llm, llm's raw response object goes into StrOutputParser,
        # which pulls out the plain answer string.
        _answer_chain = prompt | llm | StrOutputParser()
    return _answer_chain


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@app.get("/")
def health_check():
    return {"status": "ok"}


def extract_text(filename: str, contents: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(contents))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return contents.decode("utf-8")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    contents = await file.read()
    text = extract_text(file.filename, contents)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)

    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{file.filename}-{i}" for i in range(len(chunks))]
    metadatas = [{"filename": file.filename, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return {
        "filename": file.filename,
        "characters": len(text),
        "num_chunks": len(chunks),
        "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
        "total_vectors_in_collection": collection.count(),
    }


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(payload: AskRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Add it to a .env file in backend/.",
        )

    # .ainvoke(), not .invoke() -- same reasoning as await file.read() earlier:
    # both calls wait on I/O (a network request, here to Chroma then OpenAI), so
    # we free up the event loop to handle other requests while waiting.
    docs = await retriever.ainvoke(payload.question)
    answer = await get_answer_chain().ainvoke(
        {"context": format_docs(docs), "question": payload.question}
    )

    sources = [
        {"filename": doc.metadata.get("filename"), "chunk_index": doc.metadata.get("chunk_index")}
        for doc in docs
    ]

    return {
        "question": payload.question,
        "answer": answer,
        "sources": sources,
    }
