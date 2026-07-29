"""
dataset_builder/merger.py
FINAL VERSION

Merges validated dataset metadata into a single dataset.
"""

from pathlib import Path
import pandas as pd


class DatasetMerger:

    def __init__(self):
        pass

    def merge(self, metadata):

        if not metadata:
            print("No metadata available.")
            return []

        return metadata

    def save_csv(
        self,
        metadata,
        output_file="merged_dataset.csv"
    ):

        if not metadata:
            print("Nothing to save.")
            return None

        df = pd.DataFrame(metadata)

        df.to_csv(
            output_file,
            index=False
        )

        print(f"Merged dataset saved to {output_file}")

        return output_file

    def save_excel(
        self,
        metadata,
        output_file="merged_dataset.xlsx"
    ):

        if not metadata:
            print("Nothing to save.")
            return None

        df = pd.DataFrame(metadata)

        df.to_excel(
            output_file,
            index=False
        )

        print(f"Merged dataset saved to {output_file}")

        return output_file

    def summary(self, metadata):

        if not metadata:
            print("No dataset available.")
            return

        total_files = len(metadata)

        total_pages = sum(
            item.get("pages", 0)
            for item in metadata
        )

        total_words = sum(
            item.get("word_count", 0)
            for item in metadata
        )

        total_size = round(
            sum(
                item.get("size_mb", 0)
                for item in metadata
            ),
            2
        )

        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print(f"Total PDFs      : {total_files}")
        print(f"Total Pages     : {total_pages}")
        print(f"Total Words     : {total_words}")
        print(f"Total Size (MB) : {total_size}")

        print("=" * 60)


if __name__ == "__main__":

    sample = [
        {
            "filename": "WHO.pdf",
            "pages": 120,
            "word_count": 43000,
            "size_mb": 5.8
        },
        {
            "filename": "PubMed.pdf",
            "pages": 82,
            "word_count": 27000,
            "size_mb": 3.4
        }
    ]

    merger = DatasetMerger()

    merger.summary(sample)

    merger.save_csv(sample)

    merger.save_excel(sample)