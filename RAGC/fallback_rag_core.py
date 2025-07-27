#!/usr/bin/env python3
"""
fallback_rag_core.py

A 5‑level Fallback Mechanism RAG system for:
 “What happens after physical death?”
(with explicit UTF-8 loading of data/afterlife.txt)
"""

import os
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# ─── 0. API KEY ───────────────────────────────────────────────────────────────
# (For quick testing—move to an env‑var in production)

# ─── 1. LOAD & INDEX YOUR DOCUMENTS WITH UTF‑8 ─────────────────────────────────
# Manually read the file as UTF‑8, ignoring any bad bytes
file_path = "data/afterlife.txt"
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()

# Wrap it into a single Document
docs = [Document(page_content=raw_text)]

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

# Embed & index with FAISS
embeddings   = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# ─── 2. PROMPT TEMPLATE FOR FINAL OUTPUT ────────────────────────────────────────
fallback_template = PromptTemplate(
    input_variables=[
        "level", "confidence", "status",
        "response", "sources", "limitations", "suggestions"
    ],
    template="""
📊 Retrieval Level Used: {level}
🎯 Confidence Level: {confidence}
📝 Information Status: {status}

Answer:
{response}

📚 Sources Used: {sources}
⚠ Limitations: {limitations}
💡 Suggestions: {suggestions}
""".strip()
)

# ─── 3. HELPERS ─────────────────────────────────────────────────────────────────
def expand_keywords(q: str) -> str:
    return q + " life after death afterlife beliefs near-death experiences"

def semantic_expand(q: str) -> str:
    return q + " consciousness survival post-mortem psyche continuation"

def cross_domain_expand(q: str) -> str:
    return "energy conservation analogy death " + q

def evaluate_sufficiency(results, threshold=0.8, min_docs=3) -> bool:
    # 'results' is list of (Document, score)
    good = [score for _, score in results if score >= threshold]
    return len(good) >= min_docs

# ─── 4. MAIN RAG PROCESSOR ─────────────────────────────────────────────────────
def process_query(user_query: str) -> str:
    llm = OpenAI(temperature=0)

    # Level 1: Primary (vector similarity)
    lvl1 = vector_store.similarity_search_with_score(user_query, k=10)
    if evaluate_sufficiency(lvl1):
        docs = [doc for doc, _ in lvl1]
        context = "\n\n".join(d.page_content for d in docs)
        resp = llm(f"Use the following context to answer:\n\n{context}\n\nQ: {user_query}")
        return fallback_template.format(
            level="PRIMARY",
            confidence="HIGH",
            status="COMPLETE",
            response=resp.strip(),
            sources=f"{len(docs)} chunks",
            limitations="",
            suggestions="None"
        )

    # Level 2: Secondary (keyword expansion)
    q2 = expand_keywords(user_query)
    lvl2 = vector_store.similarity_search_with_score(q2, k=10)
    if evaluate_sufficiency(lvl2):
        docs = [doc for doc, _ in lvl2]
        context = "\n\n".join(d.page_content for d in docs)
        resp = llm(f"Use the following context to answer:\n\n{context}\n\nQ: {user_query}")
        return fallback_template.format(
            level="SECONDARY",
            confidence="MEDIUM",
            status="COMPLETE",
            response=resp.strip(),
            sources=f"{len(docs)} chunks",
            limitations="",
            suggestions="None"
        )

    # Level 3: Tertiary (semantic expansion)
    q3 = semantic_expand(user_query)
    lvl3 = vector_store.similarity_search_with_score(q3, k=10)
    if evaluate_sufficiency(lvl3):
        docs = [doc for doc, _ in lvl3]
        context = "\n\n".join(d.page_content for d in docs)
        resp = llm(f"Use the following context to answer:\n\n{context}\n\nQ: {user_query}")
        return fallback_template.format(
            level="TERTIARY",
            confidence="MEDIUM",
            status="COMPLETE",
            response=resp.strip(),
            sources=f"{len(docs)} chunks",
            limitations="",
            suggestions="None"
        )

    # Level 4: Quaternary (cross-domain search)
    q4 = cross_domain_expand(user_query)
    lvl4 = vector_store.similarity_search_with_score(q4, k=10)
    if lvl4:
        docs = [doc for doc, _ in lvl4]
        context = "\n\n".join(d.page_content for d in docs)
        resp = llm(f"Use the following context to answer:\n\n{context}\n\nQ: {user_query}")
        return fallback_template.format(
            level="QUATERNARY",
            confidence="LOW",
            status="PARTIAL",
            response=resp.strip(),
            sources=f"{len(docs)} chunks",
            limitations="Information may be tangential",
            suggestions="Consult domain‑specific sources"
        )

    # Level 5: Final Fallback
    return fallback_template.format(
        level="FALLBACK",
        confidence="INSUFFICIENT",
        status="LIMITED",
        response="I’m sorry, I don’t have enough information to answer that fully.",
        sources="None",
        limitations="All retrieval methods returned insufficient results.",
        suggestions="Try rephrasing or consult specialized sources."
    )

# ─── 5. RUN & PRINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query  = "What happens after physical death?"
    output = process_query(query)
    print("\n=== Fallback‑RAG Response ===\n")
    print(output)
