"""
modules/embeddings.py

Generates sentence embeddings for MedAssist AI using
Sentence Transformers.
"""

import os
from typing import List

from langchain_core.documents import Document
import numpy as np


class MedicalEmbeddingModel:
    """
    Embedding wrapper.

    Uses SentenceTransformer when explicitly requested. Defaults to a
    deterministic local HashingVectorizer so the project can build offline.
    """

    def __init__(
        self,
        model_name: str | None = None
    ):
        self.model_name = (
            model_name
            or os.getenv("MEDASSIST_EMBEDDING_MODEL")
            or "local-hash"
        )
        self.model = None
        self.vectorizer = None

        if self.model_name.lower() == "local-hash":
            self._use_local_hash()
            return

        try:
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as exc:
            print(
                "Unable to load SentenceTransformer. "
                f"Using local hash embeddings instead. Details: {exc}"
            )
            self._use_local_hash()

    def _use_local_hash(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import normalize

        print("Loading embedding model: local-hash")
        self.model_name = "local-hash"
        self.normalize = normalize
        self.vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            stop_words="english",
        )

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.vectorizer is not None:
            vectors = self.vectorizer.transform(texts)
            vectors = self.normalize(vectors, norm="l2", axis=1)
            return vectors.astype("float32").toarray()

        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def embed_documents(self, documents: List[Document]):
        """
        Generate embeddings for LangChain Documents.

        Returns
        -------
        tuple
            (texts, embeddings, metadata)
        """
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        embeddings = self._embed_texts(texts)

        return texts, embeddings, metadata

    def embed_query(self, query: str):
        """
        Generate embedding for a user query.
        """
        return self._embed_texts([query])[0]


if __name__ == "__main__":

    from pdf_loader import PDFLoader
    from chunker import MedicalTextChunker

    loader = PDFLoader()
    docs = loader.load_directory("../documents")

    chunker = MedicalTextChunker()
    chunks = chunker.split_documents(docs)

    embedder = MedicalEmbeddingModel()

    texts, vectors, metadata = embedder.embed_documents(chunks)

    print("=" * 70)
    print(f"Chunks      : {len(texts)}")
    print(f"Embeddings  : {vectors.shape}")
    print(f"Metadata    : {len(metadata)}")
