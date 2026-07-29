"""
gradcam.py
Grad-CAM implementation for MedAssist AI

Compatible with:
MedicalCNN
Last convolution layer:
model.layer4
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer=None):
        """
        Args:
            model : trained MedicalCNN model
            target_layer : last convolution block
                           default -> model.layer4
        """

        self.model = model
        self.model.eval()

        if target_layer is None:
            target_layer = model.layer4

        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    ######################################################
    # Hook Functions
    ######################################################

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    ######################################################
    # Generate GradCAM
    ######################################################

    def generate(self, input_tensor, class_idx=None):
        """
        Generates GradCAM heatmap

        Args:
            input_tensor : Tensor [1,C,H,W]
            class_idx : target class

        Returns:
            heatmap (numpy)
        """

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()

        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(weights * activations, dim=1).squeeze()

        cam = F.relu(cam)

        cam -= cam.min()

        if cam.max() != 0:
            cam /= cam.max()

        cam = cam.cpu().numpy()

        return cam
######################################################
# Heatmap Visualization
######################################################

def overlay_heatmap(
    image,
    heatmap,
    alpha=0.4,
    colormap=cv2.COLORMAP_JET
):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image : Original image (NumPy array)
        heatmap : Heatmap from GradCAM.generate()
        alpha : Transparency factor
        colormap : OpenCV colormap

    Returns:
        overlay image
    """

    # Resize heatmap to match image size
    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    # Convert to uint8
    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap = cv2.applyColorMap(
        heatmap,
        colormap
    )

    # If original image is grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    # Normalize image if needed
    if image.dtype != np.uint8:
        image = np.uint8(image * 255)

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay


######################################################
# Save GradCAM Result
######################################################

def save_gradcam(
    image,
    heatmap,
    save_path,
    alpha=0.4
):
    """
    Save GradCAM overlay image.

    Args:
        image : Original image
        heatmap : GradCAM heatmap
        save_path : Output filename
        alpha : Transparency
    """

    overlay = overlay_heatmap(
        image,
        heatmap,
        alpha
    )

    cv2.imwrite(save_path, overlay)

    print(f"Grad-CAM saved to {save_path}")


######################################################
# Utility Function
######################################################

def visualize_gradcam(
    model,
    input_tensor,
    original_image,
    class_idx=None,
    save_path=None
):
    """
    Complete GradCAM pipeline.

    Args:
        model : MedicalCNN
        input_tensor : Tensor [1,C,H,W]
        original_image : Original image (NumPy)
        class_idx : Target class
        save_path : Optional file path

    Returns:
        overlay image
    """

    gradcam = GradCAM(model)

    heatmap = gradcam.generate(
        input_tensor,
        class_idx
    )

    overlay = overlay_heatmap(
        original_image,
        heatmap
    )

    if save_path is not None:
        cv2.imwrite(save_path, overlay)

    return overlay


######################################################
# Remove Hooks
######################################################

######################################################
# Remove Hooks
######################################################

def remove_hooks(gradcam):
    """
    Remove registered hooks to free resources.
    """

    gradcam.forward_handle.remove()
    gradcam.backward_handle.remove()


######################################################
# Streamlit Helper
######################################################

def generate_visualization(
    model,
    input_tensor,
    original_image,
    class_idx=None
):
    """
    Generate both heatmap and overlay image.

    Returns
    -------
    dict
    """

    gradcam = GradCAM(model)

    heatmap = gradcam.generate(
        input_tensor,
        class_idx
    )

    overlay = overlay_heatmap(
        original_image,
        heatmap
    )

    remove_hooks(gradcam)

    return {
        "heatmap": heatmap,
        "overlay": overlay
    }