import time
import logging
import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
from PIL import Image

# Vision Modules
from .gradcam import GradCAM, generate_visualization
from .cnn import MedicalCNN
from .model_utils import load_model
from .image_loader import ImageLoader
from .image_validator import ImageValidator
from .preprocess import ImagePreprocessor
from .predict import predict
from .report_generator import MedicalReportGenerator as ReportGenerator
from .class_labels import (
    CLASS_NAMES,
    NUM_CLASSES
)

RGB_DATASETS = {
    "dermamnist",
    "retinamnist",
}

logger = logging.getLogger(__name__)
 
logger = logging.getLogger(__name__)
 
 
class VisionEngine:
    """
    Main AI Vision Engine.
 
    Pipeline
    --------
        Image
          │
          ▼
     Validation
          │
          ▼
    Preprocessing
          │
          ▼
      MedicalCNN
          │
          ▼
      Prediction
          │
          ▼
       Grad-CAM
          │
          ▼
      AI Report
    """
 
    ##################################################
    # Constructor
    ##################################################
 
    def __init__(
        self,
        model_path,
        dataset_name="pneumoniamnist",
        device=None
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to trained MedicalCNN weights.
 
        device : str or torch.device
            cpu or cuda
        """
 
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)
 
        self.model_path = model_path
        self.dataset_name = dataset_name.lower()
 
        self.model = None
 
        self.gradcam = None
        self.validator = ImageValidator()
        self.color_mode = "RGB" if self.dataset_name in RGB_DATASETS else "L"
        self.image_loader = ImageLoader(color_mode=self.color_mode)
        self.preprocessor = ImagePreprocessor()
 
        self.report_generator = ReportGenerator()
 
        self._load_model()
 
    ##################################################
    # Load Trained Model
    ##################################################
 
    def _load_model(self):
        """
        Load trained MedicalCNN.
        """
 
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found:\n{self.model_path}"
            )
 
        num_classes = NUM_CLASSES.get(self.dataset_name, 2)
        in_channels = 3 if self.dataset_name in RGB_DATASETS else 1

        model = MedicalCNN(
            num_classes=num_classes,
            in_channels=in_channels
        )
 
        model = load_model(
            model=model,
            model_path=self.model_path,
            device=self.device
        )
 
        model.eval()
 
        self.model = model
 
        ##################################################
        # Initialize GradCAM
        ##################################################
 
        self.gradcam = GradCAM(
            model=self.model,
            target_layer=self.model.layer4
        )
 
        print("=" * 60)
        print("MedicalCNN Loaded Successfully")
        print("=" * 60)
        print(f"Device      : {self.device}")
        print(f"Model Path  : {self.model_path}")
        print(f"GradCAM     : layer4")
        print("=" * 60)
 
    ##################################################
    # Utility Properties
    ##################################################
 
    @property
    def is_gpu(self):
        """
        Returns True if GPU is being used.
        """
        return self.device.type == "cuda"
 
    @property
    def is_cpu(self):
        """
        Returns True if CPU is being used.
        """
        return self.device.type == "cpu"
 
    ##################################################
    # Engine Summary
    ##################################################
 
    def summary(self):
        """
        Print Vision Engine information.
        """
 
        print("\n")
        print("=" * 60)
        print("MedAssist AI Vision Engine")
        print("=" * 60)
 
        print(f"Device          : {self.device}")
        print(f"Model Path      : {self.model_path}")
        print(f"Number Classes  : {NUM_CLASSES}")
        print(f"GradCAM Layer   : layer4")
 
        print("=" * 60)
 
    ##################################################
    # Full Analysis Pipeline
    ##################################################
 
    def analyze(
        self,
        image_path: str,
        top_k: int = 3,
        confidence_threshold: float = 0.50,
        save_gradcam_path: str = None,
    ) -> Dict[str, Any]:
        """
        Complete inference pipeline.
 
        Pipeline:
        1. Validate image
        2. Load image
        3. Preprocess
        4. Predict
        5. Generate Grad-CAM
        6. Generate report
 
        Parameters
        ----------
        image_path : str
            Path to medical image.
 
        top_k : int
            Number of predictions.
 
        confidence_threshold : float
            Minimum confidence required.
 
        save_gradcam_path : str, optional
            If provided, path to save the Grad-CAM overlay image.
 
        Returns
        -------
        Dict[str, Any]
            Complete analysis dictionary.
        """
 
        start_time = time.time()
 
        logger.info("=" * 70)
        logger.info("Starting medical image analysis")
        logger.info("Image: %s", image_path)
 
        result: Dict[str, Any] = {
            "success": False,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
        }
 
        try:
 
            # ---------------------------------------------------------
            # Step 1: Validate image
            # ---------------------------------------------------------
 
            logger.info("Validating image...")
 
            self.validator.validate_path(image_path)
            result["validation"] = {"valid": True}
 
            logger.info("Validation successful")
 
            # ---------------------------------------------------------
            # Step 2: Load image
            # ---------------------------------------------------------
 
            logger.info("Loading image...")
 
            pil_image = Image.open(image_path).convert(self.color_mode)
            original_image = np.array(pil_image.resize((28, 28)))
            input_tensor = self.image_loader.load_image(image_path)
 
            result["image_size"] = (
                pil_image.size
            )
 
            # ---------------------------------------------------------
            # Step 3: Preprocess
            # ---------------------------------------------------------
 
            logger.info("Preprocessing image...")
 
            if input_tensor is None:
                result["error"] = "Image preprocessing failed."
                logger.error(result["error"])
                return result
 
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            result["input_shape"] = tuple(input_tensor.shape)
 
            # ---------------------------------------------------------
            # Step 4: Prediction
            # ---------------------------------------------------------
 
            logger.info("Running model inference...")
 
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities_tensor = torch.softmax(outputs, dim=1).squeeze()

            top_count = min(top_k, probabilities_tensor.numel())
            top_values, top_indices = torch.topk(probabilities_tensor, top_count)

            predicted_index = int(top_indices[0].item())
            confidence = float(top_values[0].item())
            labels = CLASS_NAMES.get(self.dataset_name, {})
            predicted_class = labels.get(predicted_index, f"Class {predicted_index}")
            probabilities = probabilities_tensor.detach().cpu().numpy().tolist()
            top_predictions = [
                {
                    "class_id": int(index.item()),
                    "label": labels.get(int(index.item()), f"Class {int(index.item())}"),
                    "confidence": float(value.item()),
                }
                for value, index in zip(top_values, top_indices)
            ]

            result["prediction"] = {
                "predicted_class": predicted_class,
                "predicted_index": predicted_index,
                "confidence": confidence,
                "top_predictions": top_predictions,
            }
 
            result["predicted_class"] = predicted_class
            result["predicted_index"] = predicted_index
            result["confidence"] = confidence
            result["probabilities"] = probabilities
            result["top_predictions"] = top_predictions
 
            logger.info(
                "Prediction: %s (%.2f%%)",
                predicted_class,
                confidence * 100,
            )
 
            result["confidence_threshold"] = confidence_threshold
            result["high_confidence"] = (
                confidence >= confidence_threshold
            )
 
            # ---------------------------------------------------------
            # Step 5: Generate Grad-CAM
            # ---------------------------------------------------------
 
            gradcam_result = generate_visualization(
                model=self.model,
                input_tensor=input_tensor,
                original_image=original_image,
                class_idx=predicted_index
            )
 
            heatmap = gradcam_result["heatmap"]
            gradcam_image = gradcam_result["overlay"]
 
            # ---------------------------------------------------------
            # Save Grad-CAM (optional)
            # ---------------------------------------------------------
 
            if save_gradcam_path is not None:
 
                import cv2
 
                cv2.imwrite(
                    save_gradcam_path,
                    gradcam_image
                )
 
            # ---------------------------------------------------------
            # Step 6: Generate Medical Report
            # ---------------------------------------------------------
 
            report = self.report_generator.generate(
                dataset_name=self.dataset_name,
                prediction=predicted_class,
                confidence=confidence * 100,
                probabilities=probabilities,
                processing_time=time.time() - start_time
            )
 
            # ---------------------------------------------------------
            # Step 7: Final Results
            # ---------------------------------------------------------
 
            result["heatmap"] = heatmap
            result["gradcam_image"] = gradcam_image
            result["report"] = report
            result["success"] = True
            result["elapsed_seconds"] = time.time() - start_time
 
            logger.info("Vision analysis completed successfully.")
 
            return result
 
        except Exception as exc:
            result["error"] = f"Unexpected error during analysis: {exc}"
            logger.exception(result["error"])
            return result
