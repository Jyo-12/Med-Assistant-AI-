"""
preprocess.py
----------------
Image preprocessing utilities for MedAssist AI Vision Module
Supports all MedMNIST datasets.
"""

import torch


class ImagePreprocessor:
    """
    Handles preprocessing of MedMNIST images before training or prediction.
    """

    def __init__(self):
        pass

    def preprocess(self, image):
        """
        Preprocess a single image.

        Parameters
        ----------
        image : numpy.ndarray

        Returns
        -------
        torch.Tensor
        """

        image = torch.tensor(image, dtype=torch.float32)

        # Normalize pixel values
        image = image / 255.0

        # Add channel dimension if grayscale
        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        # Convert HWC -> CHW if RGB
        elif len(image.shape) == 3:
            image = image.permute(2, 0, 1)

        return image


def normalize_batch(images):
    """
    Normalize an entire batch of images.

    Parameters
    ----------
    images : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    return images.float() / 255.0


def prepare_for_prediction(image):
    """
    Prepare a single image before inference.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    torch.Tensor
    """

    processor = ImagePreprocessor()
    image = processor.preprocess(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    return image