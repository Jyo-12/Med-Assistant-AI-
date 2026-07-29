"""
dataset_builder/downloader.py
FINAL VERSION

Downloads medical PDFs from a list of URLs.
"""

import os
import requests
from pathlib import Path


class MedicalDatasetDownloader:

    def __init__(self, output_dir="../documents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str):

        filename = url.split("/")[-1]

        if not filename.endswith(".pdf"):
            filename += ".pdf"

        output_path = self.output_dir / filename

        if output_path.exists():
            print(f"✓ Already exists: {filename}")
            return str(output_path)

        print(f"Downloading {filename}")

        response = requests.get(
            url,
            stream=True,
            timeout=60
        )

        response.raise_for_status()

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"Saved -> {output_path}")

        return str(output_path)

    def download_from_list(self, urls):

        downloaded = []

        for url in urls:
            try:
                path = self.download_file(url)
                downloaded.append(path)

            except Exception as e:
                print(f"Failed: {url}")
                print(e)

        return downloaded


if __name__ == "__main__":

    urls = [

        "https://apps.who.int/iris/bitstream/handle/10665/44584/9789241547691_eng.pdf",

        "https://www.ncbi.nlm.nih.gov/books/NBK279396/pdf/Bookshelf_NBK279396.pdf"

    ]

    downloader = MedicalDatasetDownloader()

    downloader.download_from_list(urls)
