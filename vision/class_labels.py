"""
class_labels.py
------------------------------------
Class label mappings for MedMNIST datasets.
"""

CLASS_LABELS = {

    "breastmnist": {
        0: "Normal",
        1: "Breast Cancer"
    },

    "pneumoniamnist": {
        0: "Normal",
        1: "Pneumonia"
    },

    "dermamnist": {
        0: "Actinic Keratoses",
        1: "Basal Cell Carcinoma",
        2: "Benign Keratosis",
        3: "Dermatofibroma",
        4: "Melanoma",
        5: "Nevus",
        6: "Vascular Lesion"
    },

    "retinamnist": {
        0: "Normal",
        1: "Diabetic Retinopathy",
        2: "Glaucoma",
        3: "Cataract",
        4: "Age-related Macular Degeneration"
    },

    "octmnist": {
        0: "Choroidal Neovascularization",
        1: "Diabetic Macular Edema",
        2: "Drusen",
        3: "Normal"
    },

    "chestmnist": {
        0: "Atelectasis",
        1: "Cardiomegaly",
        2: "Effusion",
        3: "Infiltration",
        4: "Mass",
        5: "Nodule",
        6: "Pneumonia",
        7: "Pneumothorax",
        8: "Consolidation",
        9: "Edema",
        10: "Emphysema",
        11: "Fibrosis",
        12: "Pleural Thickening",
        13: "Hernia"
    }
}


def get_label(dataset_name, class_id):
    """
    Returns the class name for a prediction.
    """

    dataset_name = dataset_name.lower()

    if dataset_name not in CLASS_LABELS:
        return f"Class {class_id}"

    return CLASS_LABELS[dataset_name].get(class_id, f"Class {class_id}")
# ------------------------------------------------------------------
# Compatibility aliases
# ------------------------------------------------------------------

CLASS_NAMES = CLASS_LABELS

NUM_CLASSES = {
    dataset: len(labels)
    for dataset, labels in CLASS_LABELS.items()
}