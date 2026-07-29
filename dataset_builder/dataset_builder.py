"""
dataset_builder/dataset_builder.py
FINAL VERSION

Complete dataset building pipeline.
"""

from pathlib import Path

from .downloader import MedicalDatasetDownloader
from .validator import DatasetValidator
from .metadata import MetadataGenerator
from .merger import DatasetMerger


class MedicalDatasetBuilder:

    def __init__(self, documents_dir="../documents"):

        self.documents_dir = Path(documents_dir)

        self.downloader = MedicalDatasetDownloader(documents_dir)
        self.validator = DatasetValidator()
        self.metadata = MetadataGenerator()
        self.merger = DatasetMerger()

    def build(self, download_urls=None):

        print("=" * 60)
        print("MedAssist Dataset Builder")
        print("=" * 60)

        # -------------------------
        # Download PDFs
        # -------------------------

        if download_urls:

            print("\nDownloading documents...")

            self.downloader.download_from_list(download_urls)

        # -------------------------
        # Find PDFs
        # -------------------------

        pdfs = list(self.documents_dir.glob("*.pdf"))

        print(f"\nFound {len(pdfs)} PDF files.")

        if len(pdfs) == 0:
            print("No PDF files found.")
            return

        # -------------------------
        # Validate
        # -------------------------

        print("\nValidating PDFs...")

        valid_files = []

        for pdf in pdfs:

            if self.validator.validate(str(pdf)):
                valid_files.append(str(pdf))

        print(f"Valid PDFs : {len(valid_files)}")

        # -------------------------
        # Metadata
        # -------------------------

        print("\nGenerating metadata...")

        metadata = self.metadata.generate(valid_files)

        for item in metadata:
            print(
                f"{item['filename']} "
                f"({item['size_mb']} MB)"
            )

        # -------------------------
        # Merge
        # -------------------------

        print("\nPreparing merged dataset...")

        merged = self.merger.merge(metadata)

        print(f"Dataset contains {len(merged)} files.")

        print("\nDataset build completed successfully.")

        return merged


if __name__ == "__main__":

    builder = MedicalDatasetBuilder()

    builder.build()