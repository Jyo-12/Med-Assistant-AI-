"""
train.py
------------------------------------
Universal Training Script for MedAssist AI
Supports all MedMNIST datasets
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

try:
    from .cnn import create_model
    from .dataset import create_datasets
    from .model_utils import (
        get_device,
        save_model,
        calculate_accuracy,
    )
except ImportError:
    from cnn import create_model
    from dataset import create_datasets
    from model_utils import (
        get_device,
        save_model,
        calculate_accuracy,
    )


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0.0
    running_acc = 0.0

    progress = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        acc = calculate_accuracy(outputs, labels)

        running_loss += loss.item()
        running_acc += acc

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{acc:.4f}"
        )

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            acc = calculate_accuracy(outputs, labels)

            running_loss += loss.item()
            running_acc += acc

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)

    return epoch_loss, epoch_acc


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to MedMNIST npz file"
    )

    parser.add_argument(
        "--classes",
        required=True,
        type=int,
        help="Number of output classes"
    )

    parser.add_argument(
        "--epochs",
        default=15,
        type=int
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the best model"
    )

    parser.add_argument(
        "--in_channels",
        default=None,
        type=int,
        help="Input channels. Defaults to 1 for grayscale or 3 for RGB."
    )

    args = parser.parse_args()

    device = get_device()

    train_ds, val_ds, _ = create_datasets(args.dataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False
    )

    sample_image, _ = train_ds[0]
    in_channels = args.in_channels or int(sample_image.shape[0])

    model = create_model(
        args.classes,
        in_channels=in_channels
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.001
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_accuracy = 0.0

    print("=" * 60)
    print("Starting Training...")
    print("=" * 60)

    for epoch in range(args.epochs):

        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        print(
            f"Train Loss : {train_loss:.4f} | "
            f"Train Acc : {train_acc:.4f}"
        )

        print(
            f"Val Loss   : {val_loss:.4f} | "
            f"Val Acc   : {val_acc:.4f}"
        )

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            dataset_name = os.path.splitext(
                os.path.basename(args.dataset)
            )[0]

            save_path = args.output or os.path.join(
                "models",
                f"{dataset_name}_model.pth"
            )

            save_model(
                model,
                save_path
            )

            print("✅ Best model updated!")

    print("\nTraining Finished.")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
