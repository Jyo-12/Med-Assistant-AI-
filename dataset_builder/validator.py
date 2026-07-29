"""
dataset_builder/validator.py
FINAL VERSION

Validates medical PDF documents.
"""

from pathlib import Path
import fitz


class DatasetValidator:

    def __init__(self):
        pass

    def validate(self, pdf_path: str) -> bool:
        """
        Validate a PDF file.

        Checks:
        - Exists
        - Is PDF
        - Can be opened
        - Contains pages
        - Contains extractable text
        """

        path = Path(pdf_path)

        if not path.exists():
            print(f"❌ File not found: {path}")
            return False

        if path.suffix.lower() != ".pdf":
            print(f"❌ Not a PDF: {path.name}")
            return False

        try:
            document = fitz.open(pdf_path)

            if len(document) == 0:
                print(f"❌ Empty PDF: {path.name}")
                document.close()
                return False

            total_chars = 0

            for page in document:
                total_chars += len(page.get_text("text").strip())

            document.close()

            if total_chars == 0:
                print(f"❌ No readable text: {path.name}")
                return False

            print(f"✅ Valid PDF: {path.name}")

            return True

        except Exception as e:
            print(f"❌ Validation Error ({path.name})")
            print(e)
            return False

    def validate_folder(self, folder):

        folder = Path(folder)

        results = {}

        for pdf in folder.glob("*.pdf"):
            results[pdf.name] = self.validate(str(pdf))

        return results


if __name__ == "__main__":

    validator = DatasetValidator()

    folder = input("Enter documents folder: ")

    results = validator.validate_folder(folder)

    print("\nSummary")
    print("-" * 40)

    for file, status in results.items():
        print(file, "✓" if status else "✗")