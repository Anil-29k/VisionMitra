"""
VisionMitra - APTOS Dataset Loader V3

Uses OFFLINE PREPROCESSED images and manifest.csv.

Dataset split:
    70% Train
    15% Validation
    15% Test

Offline preprocessing:
    Quality Assessment + Enhancement
    has already been performed by:

        src/dataset/preprocess_aptos.py

This loader DOES NOT perform QA or enhancement.

It only:
    1. Loads the preprocessing manifest
    2. Loads the preprocessed image
    3. Applies training/validation/test transforms
    4. Returns image + label

Official APTOS test set remains untouched.
"""

from pathlib import Path

import pandas as pd
import torch

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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

MANIFEST_CSV = (
    PROCESSED_DIR
    / "manifest.csv"
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

IMAGE_SIZE = 224


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):
    """
    Set random seed for reproducibility.
    """

    torch.manual_seed(seed)


# ============================================================
# LOAD PREPROCESSING MANIFEST
# ============================================================

def load_manifest():
    """
    Load the offline preprocessing manifest.

    The manifest is the source of truth for:
        - image ID
        - diagnosis
        - dataset split
        - quality status
        - processed image path
        - quality metrics
    """

    if not MANIFEST_CSV.exists():

        raise FileNotFoundError(
            f"Preprocessing manifest not found:\n"
            f"{MANIFEST_CSV}\n\n"
            f"Run:\n"
            f"python -m src.dataset.preprocess_aptos"
        )

    manifest = pd.read_csv(MANIFEST_CSV)

    required_columns = {
        "id_code",
        "diagnosis",
        "split",
        "quality_status",
        "processed_path",
        "laplacian_score",
        "tenengrad_score",
        "mean_brightness",
        "brightness_std",
        "dark_ratio",
        "fov_ratio",
    }

    missing_columns = (
        required_columns - set(manifest.columns)
    )

    if missing_columns:

        raise ValueError(
            "Manifest is missing required columns:\n"
            f"{sorted(missing_columns)}\n\n"
            f"Found columns:\n"
            f"{list(manifest.columns)}"
        )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    manifest["diagnosis"] = (
        manifest["diagnosis"]
        .astype(int)
    )

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    invalid_splits = set(
        manifest["split"].unique()
    ) - valid_splits

    if invalid_splits:

        raise ValueError(
            "Manifest contains invalid split names:\n"
            f"{sorted(invalid_splits)}"
        )

    if manifest["id_code"].duplicated().any():

        duplicates = (
            manifest.loc[
                manifest["id_code"].duplicated(),
                "id_code"
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate image IDs found in manifest.\n"
            f"Examples: {duplicates[:10]}"
        )

    # --------------------------------------------------------
    # Convert processed paths to absolute paths
    # --------------------------------------------------------

    def resolve_processed_path(path_value):

        path = Path(str(path_value))

        if path.is_absolute():

            return str(path)

        return str(
            PROJECT_ROOT / path
        )

    manifest["processed_path"] = (
        manifest["processed_path"]
        .apply(resolve_processed_path)
    )

    return manifest


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
        "-" * 55
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
        transform=None
    ):
        """
        Create an APTOS dataset from manifest rows.
        """

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform

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

        # ----------------------------------------------------
        # Image information
        # ----------------------------------------------------

        image_id = row[
            "id_code"
        ]

        label = int(
            row["diagnosis"]
        )

        image_path = Path(
            row["processed_path"]
        )

        # ----------------------------------------------------
        # Check processed image
        # ----------------------------------------------------

        if not image_path.exists():

            raise FileNotFoundError(
                f"Processed image not found:\n"
                f"{image_path}\n\n"
                f"Image ID: {image_id}"
            )

        # ----------------------------------------------------
        # Load PREPROCESSED image
        # ----------------------------------------------------

        try:

            image = (
                Image.open(
                    image_path
                )
                .convert("RGB")
            )

        except Exception as e:

            raise RuntimeError(
                f"Failed to load processed image:\n"
                f"{image_path}\n"
                f"Image ID: {image_id}\n"
                f"Error: {e}"
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

def create_datasets():

    set_seed(
        RANDOM_SEED
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
    # Load manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    print()
    print(
        f"Manifest images: {len(manifest)}"
    )

    # --------------------------------------------------------
    # Create splits from manifest
    # --------------------------------------------------------

    train_df = (
        manifest[
            manifest["split"] == "train"
        ]
        .reset_index(drop=True)
    )

    val_df = (
        manifest[
            manifest["split"] == "validation"
        ]
        .reset_index(drop=True)
    )

    test_df = (
        manifest[
            manifest["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate total
    # --------------------------------------------------------

    total_split_images = (
        len(train_df)
        + len(val_df)
        + len(test_df)
    )

    if total_split_images != len(manifest):

        raise RuntimeError(
            "Split counts do not match manifest total.\n"
            f"Manifest: {len(manifest)}\n"
            f"Splits:   {total_split_images}"
        )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    for name, split_df in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:

        invalid_labels = (
            ~split_df["diagnosis"]
            .isin(range(NUM_CLASSES))
        )

        if invalid_labels.any():

            raise ValueError(
                f"{name} contains invalid diagnosis labels."
            )

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = APTOSDataset(
        dataframe=train_df,
        transform=get_transforms(
            train=True
        )
    )

    val_dataset = APTOSDataset(
        dataframe=val_df,
        transform=get_transforms(
            train=False
        )
    )

    test_dataset = APTOSDataset(
        dataframe=test_df,
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
    batch_size=16
):

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_df,
        val_df,
        test_df,
    ) = create_datasets()

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
# VERIFY PROCESSED IMAGES
# ============================================================

def verify_processed_images(
    manifest
):
    """
    Verify every processed image referenced
    by manifest.csv exists.
    """

    print()
    print(
        "Checking processed images..."
    )

    missing = []

    for _, row in manifest.iterrows():

        path = Path(
            row["processed_path"]
        )

        if not path.exists():

            missing.append(
                str(path)
            )

    if missing:

        print()
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
        f"{len(manifest)}"
    )

    print(
        "Processed dataset: PASSED ✅"
    )


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
        "Mode: OFFLINE PREPROCESSED DATA"
    )

    print(
        "Quality assessment: ALREADY DONE"
    )

    print(
        "Enhancement: ALREADY DONE"
    )

    print(
        f"Manifest: {MANIFEST_CSV}"
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    print()
    print(
        f"Total manifest images: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------------
    # Verify processed files
    # --------------------------------------------------------

    verify_processed_images(
        manifest
    )

    # --------------------------------------------------------
    # Create split dataframes
    # --------------------------------------------------------

    train_df = (
        manifest[
            manifest["split"] == "train"
        ]
        .reset_index(drop=True)
    )

    val_df = (
        manifest[
            manifest["split"] == "validation"
        ]
        .reset_index(drop=True)
    )

    test_df = (
        manifest[
            manifest["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    print()
    print(
        "Dataset split from manifest:"
    )

    print(
        f"Train      : {len(train_df)} "
        f"({len(train_df) / len(manifest) * 100:.2f}%)"
    )

    print(
        f"Validation : {len(val_df)} "
        f"({len(val_df) / len(manifest) * 100:.2f}%)"
    )

    print(
        f"Test       : {len(test_df)} "
        f"({len(test_df) / len(manifest) * 100:.2f}%)"
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
        "No image overlap: PASSED ✅"
    )

    # --------------------------------------------------------
    # Quality status summary
    # --------------------------------------------------------

    print()
    print(
        "Quality status:"
    )

    quality_summary = pd.crosstab(
        manifest["split"],
        manifest["quality_status"]
    )

    print(
        quality_summary
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
    ) = create_datasets()

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
        batch_size=16
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