#!/usr/bin/env python3
"""
corrective_rag_core.py

Demo of a Prompt‑for‑Corrective RAG system for:
 “Is enlightenment really the end of suffering?”
"""

import os
import re
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ─── 0. Set your OpenAI API Key (for testing only) ─────────────────────────────

# ─── 1. Initialize your LLM ─────────────────────────────────────────────────────
llm = OpenAI(temperature=0)

# ─── 2. Define the Evaluation Prompt ─────────────────────────────────────────────
eval_template = PromptTemplate(
    input_variables=["user_query", "retrieved_context"],
    template="""
EVALUATE_CONTEXT:
Rate the following retrieved context for the given query:

Query: {user_query}

Retrieved Context:
{retrieved_context}

Evaluation Criteria:
1. Relevance Score (0-1):
2. Completeness Score (0-1):
3. Accuracy Score (0-1):
4. Specificity Score (0-1):

Overall Quality: [EXCELLENT/GOOD/FAIR/POOR]
""".strip()
)
eval_chain = LLMChain(llm=llm, prompt=eval_template)

# ─── 3. Define the Answer Prompt ─────────────────────────────────────────────────
#    **Notice** the "Answer:" at the end so the LLM will actually generate text.
answer_template = PromptTemplate(
    input_variables=["user_query", "final_context", "quality", "confidence", "sources"],
    template="""
🔍 Context Quality: {quality}
📊 Confidence Level: {confidence}

Using the context below, answer the question as fully and clearly as possible.

Context:
{final_context}

Question:
{user_query}

Answer:
""".strip()
)
answer_chain = LLMChain(llm=llm, prompt=answer_template)

# ─── 4. Helper to Parse Overall Quality ──────────────────────────────────────────
def parse_quality(text: str) -> str:
    match = re.search(r"Overall Quality:\s*\[?(EXCELLENT|GOOD|FAIR|POOR)\]?", text, re.IGNORECASE)
    return match.group(1).upper() if match else "POOR"

# ─── 5. Core Workflow ────────────────────────────────────────────────────────────
def corrective_rag_enlightenment():
    # 5.1 Student Query & First Context
    user_query      = "Is enlightenment really the end of suffering?"
    initial_context = "The Buddha attained enlightenment (nirvana) under the Bodhi tree at Bodh Gaya."

    # 5.2 Step 1: Evaluate Context Quality
    eval_out = eval_chain.run({
        "user_query":       user_query,
        "retrieved_context": initial_context
    })
    print("\n--- EVALUATION ---\n", eval_out)
    quality = parse_quality(eval_out)

    # 5.3 Step 2: Correction Decision
    if quality in ("POOR", "FAIR"):
        print("\n⚠ ACTION: RETRIEVE_AGAIN")
        # Hard‑coded refinement from example
        refined_query = "Buddhism enlightenment end suffering nirvana cessation dukkha"
        reasoning     = (
            "Initial context only describes the event of enlightenment, "
            "but does not address whether it ends suffering (dukkha)."
        )

        # Simulate second retrieval
        corrected_context = (
            "In Buddhist doctrine, enlightenment (nirvana) is the extinguishing "
            "of craving, aversion, and ignorance—thereby ending all dukkha (suffering) "
            "and liberating one from the cycle of rebirth."
        )
        confidence = "HIGH"
        sources    = (
            "Dhammacakkappavattana Sutta; Four Noble Truths; Dhammapada Verses 277–279"
        )
    else:
        print("\n✅ ACTION: PROCEED_WITH_ANSWER")
        corrected_context = initial_context
        confidence        = "MEDIUM"
        sources           = "Initial context"

    # 5.4 Step 3: Generate Final Response
    final_answer = answer_chain.run({
        "user_query":    user_query,
        "final_context": corrected_context,
        "quality":       quality,
        "confidence":    confidence,
        "sources":       sources
    })

    # 5.5 Print the final structured answer
    print("\n=== FINAL RESPONSE ===\n", final_answer)

# ─── 6. Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    corrective_rag_enlightenment()
