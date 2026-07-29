"""
predict.py
------------------------------------
Inference Script for MedAssist AI Vision Module
Supports all trained MedMNIST models.
"""

import argparse
import numpy as np
import torch

from .cnn import create_model
from .model_utils import get_device, load_model
from .preprocess import prepare_for_prediction


def predict(model, image, device):
    """
    Perform prediction on a single image.

    Parameters
    ----------
    model : torch.nn.Module
    image : numpy.ndarray
    device : torch.device

    Returns
    -------
    predicted_class : int
    confidence : float
    probabilities : numpy.ndarray
    """

    model.eval()

    image = prepare_for_prediction(image)
    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    return (
        prediction.item(),
        confidence.item(),
        probabilities.squeeze().cpu().numpy()
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.pth)"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to .npy image"
    )

    parser.add_argument(
        "--classes",
        required=True,
        type=int,
        help="Number of output classes"
    )

    args = parser.parse_args()

    device = get_device()

    model = create_model(args.classes)

    model = load_model(
        model,
        args.model,
        device
    )

    image = np.load(args.image)

    predicted_class, confidence, probabilities = predict(
        model,
        image,
        device
    )

    print("=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence:.4f}")
    print("\nClass Probabilities:")
    print(probabilities)
    print("=" * 60)


if __name__ == "__main__":
    main()
