"""
modules/vector_store.py

FAISS vector store manager for MedAssist AI.
Builds, saves and loads a FAISS index together with metadata.
"""

from pathlib import Path
import pickle
from typing import List, Tuple

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "vector_store"


class MedicalVectorStore:
    def __init__(self, index_dir: str | Path | None = None):
        if index_dir is None:
            index_dir = DEFAULT_INDEX_DIR

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_dir / "medical.index"
        self.metadata_file = self.index_dir / "metadata.pkl"

        self.index = None
        self.texts = []
        self.metadata = []

    def build(self, texts: List[str], embeddings: np.ndarray, metadata: List[dict]):
        """Build a FAISS cosine-similarity index."""
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))

        self.index = index
        self.texts = texts
        self.metadata = metadata

    def save(self):
        """Save index and metadata to disk."""
        if self.index is None:
            raise ValueError("No FAISS index has been built.")

        faiss.write_index(self.index, str(self.index_file))

        with open(self.metadata_file, "wb") as f:
            pickle.dump(
                {
                    "texts": self.texts,
                    "metadata": self.metadata,
                },
                f,
            )

        print(f"Index saved to {self.index_dir}")

    def load(self):
        """Load an existing FAISS index."""
        if not self.index_file.exists():
            raise FileNotFoundError(self.index_file)

        self.index = faiss.read_index(str(self.index_file))

        with open(self.metadata_file, "rb") as f:
            data = pickle.load(f)

        self.texts = data["texts"]
        self.metadata = data["metadata"]

        print("Vector store loaded successfully.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple]:
        """
        Search FAISS index.

        Returns:
            [(score, text, metadata), ...]
        """
        if self.index is None:
            raise ValueError("Index not loaded.")

        query = np.asarray([query_embedding]).astype("float32")

        scores, indices = self.index.search(query, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                (
                    float(score),
                    self.texts[idx],
                    self.metadata[idx],
                )
            )

        return results


if __name__ == "__main__":
    print("MedicalVectorStore module ready.")
