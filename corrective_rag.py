# corrective_rag.py

import os
import re
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ─── 0. Set your API key in code (for testing only) ─────────────────────────────────

# ─── 1. LLM & Retriever Setup ───────────────────────────────────────────────────────
llm = OpenAI(temperature=0)

# 1.1 Load & split documents
loader = TextLoader("data/contracts.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 1.2 Embed & index
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ─── 2. Prompt Definitions ──────────────────────────────────────────────────────────
# 2.1 Evaluation prompt
eval_template = PromptTemplate(
    input_variables=["user_query", "retrieved_context"],
    template="""
EVALUATE_CONTEXT:
Rate the following retrieved context for the given query:

Query: {user_query}

Retrieved Context: {retrieved_context}

Evaluation Criteria (0–1):
1. Relevance:
2. Completeness:
3. Accuracy:
4. Specificity:

Overall Quality: [EXCELLENT/GOOD/FAIR/POOR]
""".strip()
)
eval_chain = LLMChain(llm=llm, prompt=eval_template)

# 2.2 Final answer prompt
answer_template = PromptTemplate(
    input_variables=["user_query", "final_context", "quality", "confidence", "sources"],
    template="""
🔍 Context Quality: {quality}
📊 Confidence Level: {confidence}

🎯 Answer the question:
"{user_query}"

Using this context:
{final_context}

📚 Sources: {sources}

⚠ Note: Initial retrieval was insufficient. Used corrected context.
""".strip()
)
answer_chain = LLMChain(llm=llm, prompt=answer_template)

# ─── 3. Utility to parse quality ────────────────────────────────────────────────────
def parse_quality(eval_text: str) -> str:
    m = re.search(r"Overall Quality:\s*\[?(EXCELLENT|GOOD|FAIR|POOR)\]?", eval_text, re.IGNORECASE)
    return m.group(1).upper() if m else "POOR"

# ─── 4. Corrective RAG Workflow ────────────────────────────────────────────────────
def corrective_rag(user_query: str) -> str:
    # 4.1 Initial retrieval
    docs = retriever.get_relevant_documents(user_query)
    context = "\n\n".join(d.page_content for d in docs)

    # 4.2 Evaluate that context
    eval_out = eval_chain.run({
        "user_query": user_query,
        "retrieved_context": context
    })
    quality = parse_quality(eval_out)

    # 4.3 Possibly re-retrieve
    if quality in ("POOR", "FAIR"):
        refined_query = user_query + " effects consequences problems"
        docs = retriever.get_relevant_documents(refined_query)
        context = "\n\n".join(d.page_content for d in docs)
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    # 4.4 Generate final answer
    final_out = answer_chain.run({
        "user_query": user_query,
        "final_context": context,
        "quality": quality,
        "confidence": confidence,
        "sources": "…your source list…"
    })
    return final_out

# ─── 5. Script Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = "What are the side effects of Machine Learning overfitting?"
    print(corrective_rag(q))
