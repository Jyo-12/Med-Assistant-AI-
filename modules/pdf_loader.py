"""
modules/pdf_loader.py

Utility to load medical PDF documents for the MedAssist AI project.
"""

from pathlib import Path
from typing import List

import fitz
from langchain_core.documents import Document


class PDFLoader:
    def load_pdf(self, pdf_path: str) -> List[Document]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} not found.")

        documents = []

        with fitz.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_number,
                                "path": str(pdf_path)
                            },
                        )
                    )
        return documents

    def load_directory(self, folder_path: str) -> List[Document]:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"{folder} not found.")

        all_documents = []
        for pdf_file in sorted(folder.glob("*.pdf")):
            print(f"Loading {pdf_file.name}...")
            all_documents.extend(self.load_pdf(str(pdf_file)))

        print(f"Loaded {len(all_documents)} pages.")
        return all_documents


if __name__ == "__main__":
    loader = PDFLoader()
    docs = loader.load_directory("../documents")
    print(f"Loaded {len(docs)} pages.")
