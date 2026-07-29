"""
image_loader.py
------------------------------------
Image loading and preprocessing utilities
for MedAssist AI Vision Module.
"""

import os
import numpy as np
from PIL import Image

from .preprocess import ImagePreprocessor


SUPPORTED_FORMATS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


class ImageLoader:
    """
    Loads and preprocesses medical images for prediction.
    """

    def __init__(self, image_size=28, color_mode="L"):
        self.image_size = image_size
        self.color_mode = color_mode
        self.processor = ImagePreprocessor()

    def load_image(self, image_path):
        """
        Load an image from disk.

        Parameters
        ----------
        image_path : str

        Returns
        -------
        torch.Tensor
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        extension = os.path.splitext(image_path)[1].lower()

        if extension not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {extension}"
            )

        image = Image.open(image_path)

        image = image.convert(self.color_mode)

        # Resize for MedMNIST
        image = image.resize(
            (self.image_size, self.image_size)
        )

        image = np.array(image)

        image = self.processor.preprocess(image)

        return image

    def load_pil_image(self, pil_image):
        """
        Load image directly from a PIL Image
        (useful for Streamlit uploads).
        """

        image = pil_image.convert(self.color_mode)

        image = image.resize(
            (self.image_size, self.image_size)
        )

        image = np.array(image)

        image = self.processor.preprocess(image)

        return image

    def load_numpy_image(self, image_array):
        """
        Load image from numpy array.
        """

        if len(image_array.shape) == 3:

            image = Image.fromarray(image_array)

            image = image.convert("L")

            image = image.resize(
                (self.image_size, self.image_size)
            )

            image = np.array(image)

        else:

            image = Image.fromarray(image_array)

            image = image.resize(
                (self.image_size, self.image_size)
            )

            image = np.array(image)

        image = self.processor.preprocess(image)

        return image


if __name__ == "__main__":

    loader = ImageLoader()

    image = loader.load_image(
        "../sample_images/chest_xray.png"
    )

    print("Image Shape :", image.shape)
