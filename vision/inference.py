"""
inference.py
------------------------------------
Central inference engine for
MedAssist AI Vision Module.
"""

import time
import torch

from cnn import create_model
from image_loader import ImageLoader
from image_validator import ImageValidator
from model_utils import get_device, load_model
from class_labels import get_label


class InferenceEngine:
    """
    Main inference engine.
    """

    def __init__(
        self,
        model_path,
        dataset_name,
        num_classes
    ):

        self.device = get_device()

        self.dataset_name = dataset_name

        self.loader = ImageLoader()

        self.validator = ImageValidator()

        self.model = create_model(num_classes)

        self.model = load_model(
            self.model,
            model_path,
            self.device
        )

    def predict(self, image_path):
        """
        Predict disease from image.

        Parameters
        ----------
        image_path : str

        Returns
        -------
        dict
        """

        start_time = time.time()

        # Validate image
        self.validator.validate_path(image_path)

        # Load image
        image = self.loader.load_image(image_path)

        # Add batch dimension
        image = image.unsqueeze(0)

        image = image.to(self.device)

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        prediction = prediction.item()

        confidence = confidence.item()

        probabilities = (
            probabilities.squeeze()
            .cpu()
            .numpy()
            .tolist()
        )

        label = get_label(
            self.dataset_name,
            prediction
        )

        elapsed = time.time() - start_time

        return {

            "prediction": prediction,

            "label": label,

            "confidence": round(
                confidence * 100,
                2
            ),

            "probabilities": probabilities,

            "processing_time": round(
                elapsed,
                3
            )
        }


if __name__ == "__main__":

    MODEL_PATH = "../models/pneumoniamnist_model.pth"

    IMAGE_PATH = "../sample_images/chest_xray.png"

    engine = InferenceEngine(

        model_path=MODEL_PATH,

        dataset_name="pneumoniamnist",

        num_classes=2

    )

    result = engine.predict(IMAGE_PATH)

    print("=" * 60)

    print("Prediction Result")

    print("=" * 60)

    print(f"Prediction      : {result['label']}")

    print(f"Confidence      : {result['confidence']}%")

    print(f"Processing Time : {result['processing_time']} sec")

    print("=" * 60)