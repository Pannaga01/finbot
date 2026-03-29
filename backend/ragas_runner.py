# ragas_runner.py

import os
from datasets import Dataset

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

from rag import answer
from router import route_query
from config import mock_ques, mock_ground_truth
from langsmith import traceable

# -------------------------------
# Evaluation Pipeline
# -------------------------------
def evaluation():
    rows = []

    for question, ground_truth in zip(mock_ques, mock_ground_truth):

        route = route_query(question, None)

        ans, context_chunks = answer(question, route, "c_level")

        rows.append({
            "question": question,
            "contexts": context_chunks,
            "answer": ans,
            "ground_truth": ground_truth,
        })

    dataset = Dataset.from_list(rows)

    # -------------------------------
    # LLM + Embeddings for RAGAS
    # -------------------------------
    ragas_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    )

    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    metrics = [
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    ]

    run_config = RunConfig(
        timeout=120,
        max_retries=3,
    )

    # -------------------------------
    # Run Evaluation
    # -------------------------------
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_config,
        raise_exceptions=False,
    )

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(results)


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    evaluation()