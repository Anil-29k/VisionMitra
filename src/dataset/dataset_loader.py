"""
VisionMitra - APTOS Dataset Loader V3

Uses OFFLINE PREPROCESSED images.

Dataset split:
    70% Train
    15% Validation
    15% Test

Preprocessing:
    Quality Assessment + Enhancement
    has already been performed by:

        src/dataset/preprocess_aptos.py

This loader DOES NOT perform QA or enhancement.

It only:
    1. Loads the preprocessed image
    2. Applies training/validation/test transforms
    3. Returns image + label

Official APTOS test set remains untouched.
"""

from pathlib import Path

import pandas as pd
import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

APTOS_DIR = (
    PROJECT_ROOT
    / "data"
    / "APTOS-19"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "APTOS-19"
)

TRAIN_CSV = (
    APTOS_DIR
    / "train.csv"
)

OFFICIAL_TEST_CSV = (
    APTOS_DIR
    / "test.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_SIZE = 224


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):
    """
    Set random seed for reproducible dataset splitting.
    """

    torch.manual_seed(seed)


# ============================================================
# LOAD APTOS DATAFRAME
# ============================================================

def load_aptos_dataframe():
    """
    Load APTOS training CSV.
    """

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"APTOS training CSV not found:\n"
            f"{TRAIN_CSV}"
        )

    df = pd.read_csv(
        TRAIN_CSV
    )

    required_columns = {
        "id_code",
        "diagnosis",
    }

    if not required_columns.issubset(
        df.columns
    ):

        raise ValueError(
            f"CSV must contain: "
            f"{required_columns}\n"
            f"Found: {list(df.columns)}"
        )

    df = df[
        [
            "id_code",
            "diagnosis",
        ]
    ].copy()

    df["diagnosis"] = (
        df["diagnosis"].astype(int)
    )

    return df


# ============================================================
# STRATIFIED 70 / 15 / 15 SPLIT
# ============================================================

def create_splits(
    df,
    random_seed=RANDOM_SEED
):
    """
    Create the fixed stratified:

        70% Train
        15% Validation
        15% Test

    This MUST remain identical to the split used
    during offline preprocessing.
    """

    if abs(
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
        - 1.0
    ) > 1e-6:

        raise ValueError(
            "Train/Validation/Test ratios "
            "must sum to 1."
        )

    # --------------------------------------------------------
    # 70% Train
    # 30% temporary
    # --------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=(
            VAL_RATIO + TEST_RATIO
        ),
        stratify=df["diagnosis"],
        random_state=random_seed,
    )

    # --------------------------------------------------------
    # Remaining 30%:
    #
    # 15% Validation
    # 15% Test
    # --------------------------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(
            TEST_RATIO
            / (VAL_RATIO + TEST_RATIO)
        ),
        stratify=temp_df["diagnosis"],
        random_state=random_seed,
    )

    train_df = train_df.reset_index(
        drop=True
    )

    val_df = val_df.reset_index(
        drop=True
    )

    test_df = test_df.reset_index(
        drop=True
    )

    return (
        train_df,
        val_df,
        test_df,
    )


# ============================================================
# PREPROCESSED IMAGE PATH
# ============================================================

def get_processed_image_path(
    image_id,
    split_name
):
    """
    Return path to an already preprocessed image.

    Example:

        data/processed/APTOS-19/train/
            abc123_enhanced.png
    """

    split_dir = (
        PROCESSED_DIR
        / split_name
    )

    image_path = (
        split_dir
        / f"{image_id}_enhanced.png"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Preprocessed image not found:\n"
            f"{image_path}\n\n"
            f"Run:\n"
            f"python -m src.dataset.preprocess_aptos"
        )

    return image_path


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(
    df,
    name
):
    """
    Print class distribution.
    """

    print()
    print(
        f"{name}:"
    )

    print(
        "-" * 45
    )

    total = len(df)

    for grade in range(
        NUM_CLASSES
    ):

        count = int(
            (
                df["diagnosis"]
                == grade
            ).sum()
        )

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"Grade {grade} "
            f"({CLASS_NAMES[grade]:<20}) : "
            f"{count:4d} "
            f"({percentage:5.2f}%)"
        )

    print(
        f"Total: {total}"
    )


# ============================================================
# TRANSFORMS
# ============================================================

def get_transforms(
    train=True
):
    """
    PyTorch transforms.

    IMPORTANT:

    Offline QA + enhancement has already happened.

    These transforms are only responsible for:
        - resizing
        - training augmentation
        - tensor conversion
        - normalization
    """

    if train:

        return transforms.Compose([

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=10
            ),

            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.05,
                hue=0.02
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            ),
        ])

    else:

        return transforms.Compose([

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            ),
        ])


# ============================================================
# APTOS DATASET
# ============================================================

class APTOSDataset(Dataset):

    def __init__(
        self,
        dataframe,
        split_name,
        transform=None
    ):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
        )

        self.split_name = (
            split_name
        )

        self.transform = (
            transform
        )

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index
    ):

        row = (
            self.dataframe.iloc[index]
        )

        image_id = row[
            "id_code"
        ]

        label = int(
            row["diagnosis"]
        )

        # ----------------------------------------------------
        # Load PREPROCESSED image
        # ----------------------------------------------------

        image_path = (
            get_processed_image_path(
                image_id,
                self.split_name
            )
        )

        image = (
            Image.open(
                image_path
            ).convert("RGB")
        )

        # ----------------------------------------------------
        # Apply transforms
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets(
    random_seed=RANDOM_SEED
):

    set_seed(
        random_seed
    )

    # --------------------------------------------------------
    # Check processed directory
    # --------------------------------------------------------

    if not PROCESSED_DIR.exists():

        raise FileNotFoundError(
            f"Processed dataset not found:\n"
            f"{PROCESSED_DIR}\n\n"
            f"Run preprocessing first:\n"
            f"python -m src.dataset.preprocess_aptos"
        )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    df = load_aptos_dataframe()

    # --------------------------------------------------------
    # Same fixed split
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df,
    ) = create_splits(
        df,
        random_seed=random_seed
    )

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = APTOSDataset(
        dataframe=train_df,
        split_name="train",
        transform=get_transforms(
            train=True
        )
    )

    val_dataset = APTOSDataset(
        dataframe=val_df,
        split_name="validation",
        transform=get_transforms(
            train=False
        )
    )

    test_dataset = APTOSDataset(
        dataframe=test_df,
        split_name="test",
        transform=get_transforms(
            train=False
        )
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_df,
        val_df,
        test_df,
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size=16,
    random_seed=RANDOM_SEED
):

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_df,
        val_df,
        test_df,
    ) = create_datasets(
        random_seed=random_seed
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_df,
        val_df,
        test_df,
    )


# ============================================================
# OFFICIAL APTOS TEST
# ============================================================

def get_official_test_dataframe():

    if not OFFICIAL_TEST_CSV.exists():

        raise FileNotFoundError(
            f"Official APTOS test CSV not found:\n"
            f"{OFFICIAL_TEST_CSV}"
        )

    df = pd.read_csv(
        OFFICIAL_TEST_CSV
    )

    if "id_code" not in df.columns:

        raise ValueError(
            "Official test.csv must contain "
            "'id_code'."
        )

    return df


# ============================================================
# MAIN SANITY CHECK
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "VisionMitra - APTOS Dataset Loader V3"
    )
    print("=" * 70)

    print()
    print(
        "Mode: PREPROCESSED DATA"
    )

    print(
        "Quality assessment: ALREADY DONE"
    )

    print(
        "Enhancement: ALREADY DONE"
    )

    print()

    # --------------------------------------------------------
    # Load original labels
    # --------------------------------------------------------

    df = load_aptos_dataframe()

    print(
        f"Total labeled APTOS images: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Create fixed split
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df,
    ) = create_splits(
        df,
        random_seed=RANDOM_SEED
    )

    print()
    print(
        "Dataset split:"
    )

    print(
        f"Train      : {len(train_df)} "
        f"({len(train_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Validation : {len(val_df)} "
        f"({len(val_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Test       : {len(test_df)} "
        f"({len(test_df) / len(df) * 100:.2f}%)"
    )

    print(
        f"Total      : "
        f"{len(train_df) + len(val_df) + len(test_df)}"
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    print_class_distribution(
        train_df,
        "TRAIN"
    )

    print_class_distribution(
        val_df,
        "VALIDATION"
    )

    print_class_distribution(
        test_df,
        "TEST"
    )

    # --------------------------------------------------------
    # Check split overlap
    # --------------------------------------------------------

    train_ids = set(
        train_df["id_code"]
    )

    val_ids = set(
        val_df["id_code"]
    )

    test_ids = set(
        test_df["id_code"]
    )

    train_val_overlap = (
        train_ids & val_ids
    )

    train_test_overlap = (
        train_ids & test_ids
    )

    val_test_overlap = (
        val_ids & test_ids
    )

    print()
    print(
        "Split overlap check:"
    )

    print(
        f"Train ∩ Validation : "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(val_test_overlap)}"
    )

    assert len(
        train_val_overlap
    ) == 0

    assert len(
        train_test_overlap
    ) == 0

    assert len(
        val_test_overlap
    ) == 0

    print(
        "No image overlap: PASSED"
    )

    # --------------------------------------------------------
    # Check processed files
    # --------------------------------------------------------

    print()
    print(
        "Checking processed images..."
    )

    missing = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:

        for image_id in split_df[
            "id_code"
        ]:

            path = (
                PROCESSED_DIR
                / split_name
                / f"{image_id}_enhanced.png"
            )

            if not path.exists():

                missing.append(
                    str(path)
                )

    if missing:

        print(
            f"Missing processed images: "
            f"{len(missing)}"
        )

        for path in missing[:10]:

            print(
                f"  {path}"
            )

        raise RuntimeError(
            "Processed dataset verification failed."
        )

    print(
        f"Processed images verified: "
        f"{len(df)}"
    )

    print(
        "Processed dataset: PASSED ✅"
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
        _,
        _,
        _,
    ) = create_datasets(
        random_seed=RANDOM_SEED
    )

    print()
    print(
        "Dataset objects:"
    )

    print(
        f"Train      : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation : "
        f"{len(val_dataset)}"
    )

    print(
        f"Test       : "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Test individual samples
    # --------------------------------------------------------

    print()
    print(
        "Testing individual samples..."
    )

    train_image, train_label = (
        train_dataset[0]
    )

    val_image, val_label = (
        val_dataset[0]
    )

    test_image, test_label = (
        test_dataset[0]
    )

    print(
        f"Train image shape : "
        f"{train_image.shape}"
    )

    print(
        f"Train label       : "
        f"{train_label.item()}"
    )

    print(
        f"Validation image shape : "
        f"{val_image.shape}"
    )

    print(
        f"Validation label       : "
        f"{val_label.item()}"
    )

    print(
        f"Test image shape : "
        f"{test_image.shape}"
    )

    print(
        f"Test label       : "
        f"{test_label.item()}"
    )

    print(
        "Individual samples: PASSED ✅"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print()
    print(
        "Testing DataLoaders..."
    )

    (
        train_loader,
        val_loader,
        test_loader,
        _,
        _,
        _,
    ) = create_dataloaders(
        batch_size=16,
        random_seed=RANDOM_SEED
    )

    train_images, train_labels = next(
        iter(train_loader)
    )

    val_images, val_labels = next(
        iter(val_loader)
    )

    test_images, test_labels = next(
        iter(test_loader)
    )

    print(
        f"Train batch images : "
        f"{train_images.shape}"
    )

    print(
        f"Train batch labels : "
        f"{train_labels.shape}"
    )

    print(
        f"Validation batch images : "
        f"{val_images.shape}"
    )

    print(
        f"Validation batch labels : "
        f"{val_labels.shape}"
    )

    print(
        f"Test batch images : "
        f"{test_images.shape}"
    )

    print(
        f"Test batch labels : "
        f"{test_labels.shape}"
    )

    print(
        "DataLoader test: PASSED ✅"
    )

    # --------------------------------------------------------
    # Official test
    # --------------------------------------------------------

    official_test = (
        get_official_test_dataframe()
    )

    print()
    print(
        "Official APTOS test set:"
    )

    print(
        f"Images: "
        f"{len(official_test)}"
    )

    print(
        "Ground-truth labels: "
        "NOT PROVIDED"
    )

    print(
        "Official test set remains untouched 🔒"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "DATASET LOADER V3 CHECK COMPLETE ✅"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()