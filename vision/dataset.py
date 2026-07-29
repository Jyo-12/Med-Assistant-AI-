"""
dataset.py
-----------------------
Universal MedMNIST Dataset Loader
Supports all MedMNIST datasets.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .preprocess import ImagePreprocessor
except ImportError:
    from preprocess import ImagePreprocessor


class MedMNISTDataset(Dataset):
    """
    Universal dataset loader for MedMNIST.
    """

    def __init__(self, npz_path, split="train"):
        """
        Parameters
        ----------
        npz_path : str
            Path to MedMNIST .npz file

        split : str
            train / val / test
        """

        self.data = np.load(npz_path)

        self.images = self.data[f"{split}_images"]
        self.labels = self.data[f"{split}_labels"]

        self.processor = ImagePreprocessor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]

        label = self.labels[index]

        image = self.processor.preprocess(image)

        label = torch.tensor(
            label,
            dtype=torch.long
        ).squeeze()

        return image, label


def create_datasets(npz_path):
    """
    Creates Train, Validation and Test datasets.
    """

    train_dataset = MedMNISTDataset(
        npz_path,
        split="train"
    )

    val_dataset = MedMNISTDataset(
        npz_path,
        split="val"
    )

    test_dataset = MedMNISTDataset(
        npz_path,
        split="test"
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":

    DATASET_PATH = "../images/chestmnist.npz"

    train_ds, val_ds, test_ds = create_datasets(DATASET_PATH)

    print("=" * 50)

    print("Train Samples :", len(train_ds))

    print("Validation Samples :", len(val_ds))

    print("Test Samples :", len(test_ds))

    print("=" * 50)

    image, label = train_ds[0]

    print("Image Shape :", image.shape)

    print("Label :", label)
