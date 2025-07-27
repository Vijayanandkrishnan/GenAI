#!/usr/bin/env python3
"""
web_search_rag_core.py

Runs a hybrid Internal+Web Search RAG for three scenarios:
  1. CRISPR Gene Therapy Breakthroughs
  2. EV Adoption in India
  3. Remote Team Management Best Practices
"""

import os
import re
from datetime import datetime
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ─── 0. CONFIG & API KEY ───────────────────────────────────────────────────────
# Replace with your own key or set the OPENAI_API_KEY env‑var
# ─── 1. LOAD & INDEX INTERNAL KB ───────────────────────────────────────────────
# Place all your internal .txt files under data/internal/
loader = DirectoryLoader("data/internal", glob="**/*.txt")
internal_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(internal_docs)

vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
internal_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ─── 2. SET UP WEB SEARCH TOOL ─────────────────────────────────────────────────
web_search = DuckDuckGoSearchRun()

# ─── 3. PROMPT FOR SYNTHESIS ────────────────────────────────────────────────────
integration_prompt = PromptTemplate(
    input_variables=[
        "strategy", "currency", "confidence",
        "user_query", "internal_context", "web_context", "last_updated"
    ],
    template="""
🔍 Search Strategy: {strategy}
🕐 Information Currency: {currency}
🎯 Confidence Level: {confidence}

— INTERNAL KB —
{internal_context}

— WEB RESULTS —
{web_context}

Answer the question: "{user_query}"

📚 Sources: Internal + Web  
🔄 Last Updated: {last_updated}
""".strip()
)
integration_chain = LLMChain(llm=OpenAI(temperature=0), prompt=integration_prompt)

# ─── 4. EVALUATION HELPERS ─────────────────────────────────────────────────────
TEMPORAL_PATTERN = re.compile(r"\b(latest|recent|current|\d{4}|today)\b", re.IGNORECASE)

def evaluate_internal(docs):
    """Return (recency, coverage, authority, completeness)."""
    count = len(docs)
    recency = 0.5
    coverage = min(count / 5.0, 1.0)
    authority = 0.8
    completeness = coverage
    return recency, coverage, authority, completeness

def needs_web_query(recency, coverage, query):
    return (
        recency < 0.6
        or coverage < 0.7
        or bool(TEMPORAL_PATTERN.search(query))
    )

# ─── 5. CORE RAG FUNCTION ──────────────────────────────────────────────────────
def run_web_search_rag(user_query: str) -> str:
    llm = OpenAI(temperature=0)

    # 5.1 Internal retrieval + evaluation
    docs = internal_retriever.get_relevant_documents(user_query)
    internal_ctx = "\n\n".join(d.page_content for d in docs) or "(no internal matches)"
    recency, coverage, authority, completeness = evaluate_internal(docs)

    # 5.2 Decide on web search
    if needs_web_query(recency, coverage, user_query):
        strategy, currency, confidence = "WEB_SUPPLEMENTED", "CURRENT", "HIGH"
        web_ctx = web_search.run(user_query)
    else:
        strategy, currency, confidence = "INTERNAL_ONLY", "HISTORICAL", "MEDIUM"
        web_ctx = "(no web search performed)"

    # 5.3 Synthesize answer
    output = integration_chain.run({
        "strategy":         strategy,
        "currency":         currency,
        "confidence":       confidence,
        "user_query":       user_query,
        "internal_context": internal_ctx,
        "web_context":      web_ctx,
        "last_updated":     datetime.utcnow().isoformat()
    })
    return output.strip()

# ─── 6. SCENARIOS & MAIN ────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "name": "CRISPR Gene Therapy Breakthroughs",
        "query": "What are the emerging breakthroughs in CRISPR-based gene therapies as of 2024?"
    },
    {
        "name": "EV Adoption in India",
        "query": "How have electric vehicle adoption rates changed in India during 2024?"
    },
    {
        "name": "Remote Team Management Best Practices",
        "query": "What are the current best practices for managing a fully remote software engineering team?"
    },
]

def main():
    for scenario in SCENARIOS:
        print(f"\n=== Scenario: {scenario['name']} ===")
        print(run_web_search_rag(scenario["query"]))
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()
