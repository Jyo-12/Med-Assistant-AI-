"""
dataset_builder/metadata.py
FINAL VERSION

Generates metadata for medical PDF datasets.
"""

from pathlib import Path
import fitz
from datetime import datetime


class MetadataGenerator:

    def __init__(self):
        pass

    def generate(self, pdf_files):

        metadata = []

        for pdf in pdf_files:

            try:
                path = Path(pdf)

                document = fitz.open(pdf)

                total_pages = len(document)

                total_words = 0

                title = path.stem

                for page in document:

                    text = page.get_text("text")

                    total_words += len(text.split())

                document.close()

                metadata.append(
                    {
                        "filename": path.name,
                        "title": title,
                        "filepath": str(path.resolve()),
                        "pages": total_pages,
                        "word_count": total_words,
                        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(
                            path.stat().st_ctime
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "modified": datetime.fromtimestamp(
                            path.stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            except Exception as e:

                print(f"Error reading {pdf}")
                print(e)

        return metadata

    def save_as_csv(self, metadata, output_file="dataset_metadata.csv"):

        import pandas as pd

        df = pd.DataFrame(metadata)

        df.to_csv(output_file, index=False)

        print(f"Metadata saved to {output_file}")

        return output_file


if __name__ == "__main__":

    folder = Path("../documents")

    pdfs = list(folder.glob("*.pdf"))

    generator = MetadataGenerator()

    metadata = generator.generate([str(pdf) for pdf in pdfs])

    for item in metadata:
        print(item)

    generator.save_as_csv(metadata)