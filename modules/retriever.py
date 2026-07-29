"""
modules/retriever.py

Hybrid Retriever for MedAssist AI
---------------------------------
Combines:
1. FAISS Semantic Search
2. BM25 Keyword Search

Returns the highest ranked medical context for Gemini.
"""

from typing import List
from collections import defaultdict
import re
import numpy as np

from .vector_store import MedicalVectorStore
from .embeddings import MedicalEmbeddingModel


class HybridRetriever:

    def __init__(self, vector_store_path=None):

        self.vector_store = MedicalVectorStore(vector_store_path)
        self.vector_store.load()

        self.embedder = MedicalEmbeddingModel()

        self.keyword_index = defaultdict(set)

        for idx, text in enumerate(self.vector_store.texts):
            for token in set(re.findall(r"[a-z0-9]+", text.lower())):
                self.keyword_index[token].add(idx)
        self._generic_query_terms = {
            "what",
            "are",
            "the",
            "symptom",
            "symptoms",
            "sign",
            "signs",
            "cause",
            "causes",
            "treatment",
            "treatments",
            "medicine",
            "medicines",
            "disease",
            "condition",
            "patient",
            "health",
            "medical",
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):
        """
        Hybrid Retrieval

        Returns
        -------
        List[dict]
        """

        # -------- Semantic Search -------- #

        query_embedding = self.embedder.embed_query(query)

        semantic_results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        # -------- BM25 Search -------- #

        query_terms = re.findall(r"[a-z0-9]+", query.lower())
        if "symptom" in query_terms or "symptoms" in query_terms:
            query_terms.extend(["sign", "signs"])
        important_terms = [
            term
            for term in query_terms
            if term not in self._generic_query_terms and len(term) > 2
        ]

        keyword_scores = defaultdict(float)

        for term in query_terms:
            weight = 3.0 if term in important_terms else 1.0
            for idx in self.keyword_index.get(term, ()):
                keyword_scores[idx] += weight

        keyword_indices = [
            idx
            for idx, _ in sorted(
                keyword_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:top_k]
        ]

        combined = {}

        # Semantic results
        for rank, (score, text, metadata) in enumerate(semantic_results, start=1):

            key = metadata["source"] + "_" + str(metadata["page"])

            combined[key] = {
                "text": text,
                "metadata": metadata,
                "semantic_score": float(score),
                "semantic_rank_score": 1.0 / rank,
                "keyword_score": 0,
                "keyword_rank_score": 0,
            }

        # BM25 results
        for rank, idx in enumerate(keyword_indices, start=1):

            metadata = self.vector_store.metadata[idx]

            key = metadata["source"] + "_" + str(metadata["page"])

            if key in combined:

                combined[key]["keyword_score"] = float(
                    keyword_scores[idx]
                )
                combined[key]["keyword_rank_score"] = 1.0 / rank

            else:

                combined[key] = {

                    "text": self.vector_store.texts[idx],

                    "metadata": metadata,

                    "semantic_score": 0,
                    "semantic_rank_score": 0,

                    "keyword_score": float(
                        keyword_scores[idx]
                    ),
                    "keyword_rank_score": 1.0 / rank,
                }

        results = list(combined.values())

        results.sort(
            key=lambda x:
            (x["semantic_rank_score"] * 0.5) +
            (x["keyword_rank_score"] * 1.5),
            reverse=True
        )

        return results[:top_k]


if __name__ == "__main__":

    retriever = HybridRetriever()

    query = "What are symptoms of diabetes?"

    results = retriever.retrieve(query)

    print("=" * 80)

    for i, item in enumerate(results, start=1):

        print(f"\nResult {i}")

        print("Source :", item["metadata"]["source"])

        print("Page   :", item["metadata"]["page"])

        print("Semantic :", round(item["semantic_score"], 3))

        print("BM25      :", round(item["keyword_score"], 3))

        print(item["text"][:400])
