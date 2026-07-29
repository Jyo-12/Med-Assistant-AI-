"""
build_index.py

Builds the complete RAG index for MedAssist AI.

Pipeline:
PDFs -> Chunking -> Embeddings -> FAISS Index
"""

from modules.pdf_loader import PDFLoader
from modules.chunker import MedicalTextChunker
from modules.embeddings import MedicalEmbeddingModel
from modules.vector_store import MedicalVectorStore


def main():

    print("=" * 70)
    print("MedAssist AI - Building Vector Index")
    print("=" * 70)

    # Folder containing WHO / PubMed PDFs
    DOCUMENT_FOLDER = "documents"

    # Output directory
    VECTOR_STORE = "vector_store"

    # Step 1 - Load PDFs
    loader = PDFLoader()
    documents = loader.load_directory(DOCUMENT_FOLDER)

    if len(documents) == 0:
        raise ValueError("No PDF documents found.")

    # Step 2 - Chunk
    chunker = MedicalTextChunker(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = chunker.split_documents(documents)

    # Step 3 - Embeddings
    embedder = MedicalEmbeddingModel()

    texts, embeddings, metadata = embedder.embed_documents(chunks)

    # Step 4 - Build FAISS
    store = MedicalVectorStore(VECTOR_STORE)

    store.build(
        texts=texts,
        embeddings=embeddings,
        metadata=metadata
    )

    # Step 5 - Save
    store.save()

    print("\nIndex creation completed successfully!")
    print(f"Documents Loaded : {len(documents)}")
    print(f"Chunks Created   : {len(chunks)}")
    print(f"Embedding Shape  : {embeddings.shape}")
    print(f"Index Directory  : {VECTOR_STORE}")


if __name__ == "__main__":
    main()
