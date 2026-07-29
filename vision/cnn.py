"""
cnn.py
-------------------------------------------------------
MedicalCNN Backbone for MedAssist AI

Features
--------
✓ Grad-CAM Compatible
✓ Batch Normalization
✓ Xavier Initialization
✓ Adaptive Average Pooling
✓ Dropout
✓ Modular Architecture
✓ Production Ready
-------------------------------------------------------
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolution Block

    Conv
    BatchNorm
    ReLU
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)


class MedicalCNN(nn.Module):

    """
    CNN Backbone for Medical Images
    """

    def __init__(self,
                 num_classes=2,
                 in_channels=1):

        super().__init__()

        ##############################################
        # Feature Extractor
        ##############################################

        self.layer1 = ConvBlock(
            in_channels,
            32
        )

        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = ConvBlock(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.layer3 = ConvBlock(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(2)

        ###################################################
        # LAST CONVOLUTION LAYER
        # GradCAM will use this layer
        ###################################################

        self.layer4 = ConvBlock(
            128,
            256
        )

        ###################################################

        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        ##############################################
        # Classifier
        ##############################################

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(
                p=0.40
            ),

            nn.Linear(
                128,
                num_classes
            )

        )

        self.initialize_weights()

    ####################################################
    # Forward
    ####################################################

    def forward(self, x):

        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)
        x = self.pool3(x)

        ##############################################
        # GradCAM target layer
        ##############################################

        x = self.layer4(x)

        ##############################################

        x = self.global_pool(x)

        x = self.classifier(x)

        return x

    ####################################################
    # Feature Extraction
    ####################################################

    def extract_features(self, x):

        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)
        x = self.pool3(x)

        x = self.layer4(x)

        return x

    ####################################################
    # GradCAM Support
    ####################################################

    def get_last_conv_layer(self):

        """
        Returns the last convolution layer
        for GradCAM.
        """

        return self.layer4
    ####################################################
    # Weight Initialization
    ####################################################

    def initialize_weights(self):
        """
        Xavier Initialization
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):

                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    ####################################################
    # Freeze Backbone
    ####################################################

    def freeze_features(self):

        """
        Freeze convolution layers.
        Useful for transfer learning.
        """

        for parameter in [
            *self.layer1.parameters(),
            *self.layer2.parameters(),
            *self.layer3.parameters(),
            *self.layer4.parameters()
        ]:

            parameter.requires_grad = False

    ####################################################
    # Unfreeze Backbone
    ####################################################

    def unfreeze_features(self):

        """
        Enable training of convolution layers.
        """

        for parameter in self.parameters():

            parameter.requires_grad = True

    ####################################################
    # Count Parameters
    ####################################################

    def count_parameters(self):

        return sum(

            parameter.numel()

            for parameter in self.parameters()

            if parameter.requires_grad

        )


##########################################################
# Factory Function
##########################################################

def create_model(
        num_classes,
        in_channels=1
):

    """
    Factory function for creating the model.
    """

    model = MedicalCNN(

        num_classes=num_classes,

        in_channels=in_channels

    )

    return model


##########################################################
# Model Summary
##########################################################

def print_model_summary(model):

    print("=" * 60)

    print("MedicalCNN")

    print("=" * 60)

    print(model)

    print("=" * 60)

    print(
        f"Trainable Parameters : {model.count_parameters():,}"
    )

    print("=" * 60)


##########################################################
# Self Test
##########################################################

if __name__ == "__main__":

    NUM_CLASSES = 2

    model = create_model(
        num_classes=NUM_CLASSES
    )

    print_model_summary(model)

    x = torch.randn(
        8,
        1,
        28,
        28
    )

    y = model(x)

    print("\nInput Shape :", x.shape)

    print("Output Shape:", y.shape)

    print(
        "\nLast Conv Layer:"
    )

    print(
        model.get_last_conv_layer()
    )

    features = model.extract_features(x)

    print(
        "\nFeature Map Shape:",
        features.shape
    )

    print(
        "\nMedicalCNN Ready for Training."
    )

    print(
        "Grad-CAM Compatible."
    )