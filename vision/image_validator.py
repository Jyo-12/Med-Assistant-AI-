"""
image_validator.py
------------------------------------
Image validation utilities for
MedAssist AI Vision Module.
"""

import os
from PIL import Image

SUPPORTED_FORMATS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

MAX_FILE_SIZE_MB = 10


class ImageValidator:
    """
    Validates uploaded medical images.
    """

    def __init__(self):
        pass

    def validate_path(self, image_path):
        """
        Validate image stored on disk.
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

        file_size = os.path.getsize(image_path)

        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(
                f"Image exceeds {MAX_FILE_SIZE_MB} MB."
            )

        try:

            image = Image.open(image_path)

            image.verify()

        except Exception:

            raise ValueError(
                "Corrupted or unreadable image."
            )

        return True

    def validate_pil(self, image):
        """
        Validate PIL image.
        """

        if image is None:
            raise ValueError("Invalid image.")

        width, height = image.size

        if width < 20 or height < 20:
            raise ValueError(
                "Image resolution too small."
            )

        return True

    def validate_numpy(self, image_array):
        """
        Validate numpy image.
        """

        if image_array is None:
            raise ValueError(
                "Image array is empty."
            )

        if image_array.size == 0:
            raise ValueError(
                "Image contains no data."
            )

        return True

    def validate_dimensions(
        self,
        image,
        min_width=20,
        min_height=20
    ):
        """
        Validate image dimensions.
        """

        width, height = image.size

        if width < min_width:
            raise ValueError(
                f"Width must be >= {min_width}px"
            )

        if height < min_height:
            raise ValueError(
                f"Height must be >= {min_height}px"
            )

        return True


if __name__ == "__main__":

    validator = ImageValidator()

    try:

        validator.validate_path(
            "../sample_images/chest_xray.png"
        )

        print("✅ Image is valid.")

    except Exception as e:

        print(e)