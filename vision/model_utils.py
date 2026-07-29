"""
model_utils.py
-----------------
Utility functions for the MedAssist AI Vision Module.
"""

import os
import torch


def get_device():
    """
    Returns the available device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    return device


def save_model(model, save_path):
    """
    Save model weights.

    Parameters
    ----------
    model : torch.nn.Module
    save_path : str
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully at:\n{save_path}")


def load_model(model, model_path, device):
    """
    Load saved model weights.

    Parameters
    ----------
    model : torch.nn.Module
    model_path : str
    device : torch.device
    """
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()

    print(f"Model loaded successfully from:\n{model_path}")

    return model


def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy.

    Parameters
    ----------
    outputs : torch.Tensor
    labels : torch.Tensor

    Returns
    -------
    float
    """

    _, predicted = torch.max(outputs, 1)

    correct = (predicted == labels).sum().item()

    accuracy = correct / labels.size(0)

    return accuracy


def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def print_model_summary(model):
    """
    Print model information.
    """

    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    print(model)

    print("\nTrainable Parameters:", count_parameters(model))

    print("=" * 60)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save training checkpoint.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved at {path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Resume training from checkpoint.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Checkpoint loaded from epoch {epoch}")

    return model, optimizer, epoch, loss
