"""
modules/chunker.py

Splits LangChain Documents into overlapping chunks for RAG.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class MedicalTextChunker:
    """
    Chunk medical documents while preserving context.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of LangChain Documents.

        Returns:
            List of chunked Documents.
        """
        chunks = self.splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks.")
        return chunks


if __name__ == "__main__":

    from pdf_loader import PDFLoader

    loader = PDFLoader()

    docs = loader.load_directory("../documents")

    chunker = MedicalTextChunker()

    chunks = chunker.split_documents(docs)

    print("=" * 70)
    print(f"Total Chunks : {len(chunks)}")
    print("=" * 70)

    if chunks:
        print(chunks[0].metadata)
        print(chunks[0].page_content[:600])
