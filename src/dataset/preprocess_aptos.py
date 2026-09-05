"""
VisionMitra - APTOS Offline Preprocessing

Purpose:
    Run quality assessment + enhancement ONCE and save the
    processed images to disk.

Dataset split:
    70% Train
    15% Validation
    15% Test

IMPORTANT:
    The split is fixed with RANDOM_SEED = 42 so it matches
    the dataset split used by dataset_loader.py.

Output:
    data/processed/APTOS-19/
        train/
        validation/
        test/
        manifest.csv
"""

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

from src.quality.quality_assessment import analyze_image_data
from src.preprocessing.enhancement import enhance_fundus_image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

APTOS_DIR = PROJECT_ROOT / "data" / "APTOS-19"

TRAIN_CSV = APTOS_DIR / "train.csv"
TRAIN_IMAGES = APTOS_DIR / "train_images"

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "APTOS-19"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SUPPORTED_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_aptos_dataframe():
    """
    Load the labeled APTOS training dataframe.
    """

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"APTOS CSV not found:\n{TRAIN_CSV}"
        )

    if not TRAIN_IMAGES.exists():
        raise FileNotFoundError(
            f"APTOS image directory not found:\n"
            f"{TRAIN_IMAGES}"
        )

    df = pd.read_csv(TRAIN_CSV)

    required_columns = {
        "id_code",
        "diagnosis"
    }

    if not required_columns.issubset(
        df.columns
    ):
        raise ValueError(
            "APTOS CSV must contain "
            "'id_code' and 'diagnosis'."
        )

    df = df[
        ["id_code", "diagnosis"]
    ].copy()

    df["diagnosis"] = (
        df["diagnosis"].astype(int)
    )

    return df


# ============================================================
# IMAGE PATH
# ============================================================

def get_image_path(image_id):
    """
    Find the original APTOS image.
    """

    for extension in SUPPORTED_EXTENSIONS:

        path = (
            TRAIN_IMAGES
            / f"{image_id}{extension}"
        )

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Image not found: {image_id}"
    )


# ============================================================
# FIXED STRATIFIED SPLIT
# ============================================================

def create_splits(
    df,
    random_seed=RANDOM_SEED
):
    """
    Create the same fixed 70/15/15 stratified split
    used by dataset_loader.py.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=(
            VAL_RATIO + TEST_RATIO
        ),
        stratify=df["diagnosis"],
        random_state=random_seed,
    )

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
        test_df
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def preprocess_image(image_path):
    """
    Run VisionMitra quality assessment and enhancement.

    Returns:
        enhanced_image
        quality_status
        quality_metrics
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    rgb_array = np.array(image)

    # --------------------------------------------------------
    # RGB -> BGR
    # --------------------------------------------------------

    bgr_image = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Quality assessment
    # --------------------------------------------------------

    quality_result = analyze_image_data(
        bgr_image
    )

    quality_status = (
        quality_result["status"]
    )

    # --------------------------------------------------------
    # Enhancement
    # --------------------------------------------------------

    if quality_status in [
        "GOOD",
        "BORDERLINE"
    ]:

        enhancement_result = (
            enhance_fundus_image(
                bgr_image
            )
        )

        enhanced_bgr = (
            enhancement_result["enhanced"]
        )

    else:

        # ----------------------------------------------------
        # POOR images are NOT enhanced and silently accepted.
        #
        # We save them separately so we can decide later
        # whether to exclude them from training.
        # ----------------------------------------------------

        enhanced_bgr = bgr_image.copy()

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    enhanced_rgb = cv2.cvtColor(
        enhanced_bgr,
        cv2.COLOR_BGR2RGB
    )

    enhanced_image = Image.fromarray(
        enhanced_rgb
    )

    # --------------------------------------------------------
    # Keep useful QA information
    # --------------------------------------------------------

    metrics = {
        "laplacian_score": (
            quality_result["laplacian_score"]
        ),
        "tenengrad_score": (
            quality_result["tenengrad_score"]
        ),
        "mean_brightness": (
            quality_result["mean_brightness"]
        ),
        "brightness_std": (
            quality_result["brightness_std"]
        ),
        "dark_ratio": (
            quality_result["dark_ratio"]
        ),
        "fov_ratio": (
            quality_result["fov_ratio"]
        ),
    }

    return (
        enhanced_image,
        quality_status,
        metrics
    )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    df,
    split_name,
    output_dir,
    manifest_rows
):
    """
    Process all images belonging to one split.
    """

    split_dir = (
        output_dir / split_name
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total = len(df)

    print()
    print("=" * 70)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 70)

    print(
        f"Images: {total}"
    )

    quality_counts = {
        "GOOD": 0,
        "BORDERLINE": 0,
        "POOR": 0,
    }

    successful = 0
    failed = 0

    start_time = time.time()

    for index, row in df.iterrows():

        image_id = row["id_code"]
        diagnosis = int(
            row["diagnosis"]
        )

        try:

            original_path = get_image_path(
                image_id
            )

            (
                processed_image,
                quality_status,
                metrics
            ) = preprocess_image(
                original_path
            )

            # ------------------------------------------------
            # Save processed image
            # ------------------------------------------------

            output_filename = (
                f"{image_id}_enhanced.png"
            )

            output_path = (
                split_dir
                / output_filename
            )

            processed_image.save(
                output_path,
                format="PNG"
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            quality_counts[
                quality_status
            ] += 1

            successful += 1

            # ------------------------------------------------
            # Manifest entry
            # ------------------------------------------------

            manifest_rows.append({
                "id_code": image_id,
                "diagnosis": diagnosis,
                "split": split_name,
                "quality_status": quality_status,
                "original_path": str(
                    original_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "processed_path": str(
                    output_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "laplacian_score": metrics[
                    "laplacian_score"
                ],
                "tenengrad_score": metrics[
                    "tenengrad_score"
                ],
                "mean_brightness": metrics[
                    "mean_brightness"
                ],
                "brightness_std": metrics[
                    "brightness_std"
                ],
                "dark_ratio": metrics[
                    "dark_ratio"
                ],
                "fov_ratio": metrics[
                    "fov_ratio"
                ],
            })

        except Exception as error:

            failed += 1

            print()
            print(
                f"ERROR processing "
                f"{image_id}:"
            )
            print(error)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        completed = index + 1

        if (
            completed == 1
            or completed % 50 == 0
            or completed == total
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"[{completed:4d}/{total}] "
                f"{completed / total * 100:6.2f}% | "
                f"GOOD={quality_counts['GOOD']} "
                f"BORDERLINE={quality_counts['BORDERLINE']} "
                f"POOR={quality_counts['POOR']} | "
                f"Failed={failed} | "
                f"Time={elapsed / 60:.1f} min"
            )

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        f"{split_name.upper()} COMPLETE"
    )

    print(
        f"Successful : {successful}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"GOOD       : "
        f"{quality_counts['GOOD']}"
    )

    print(
        f"BORDERLINE : "
        f"{quality_counts['BORDERLINE']}"
    )

    print(
        f"POOR       : "
        f"{quality_counts['POOR']}"
    )

    print(
        f"Time       : "
        f"{elapsed / 60:.1f} minutes"
    )

    return (
        successful,
        failed,
        quality_counts
    )


# ============================================================
# VERIFY OUTPUT
# ============================================================

def verify_outputs(
    manifest_df
):
    """
    Verify that every manifest entry points to an
    existing processed image.
    """

    print()
    print("=" * 70)
    print("VERIFYING PREPROCESSED DATA")
    print("=" * 70)

    missing = []

    for path in manifest_df[
        "processed_path"
    ]:

        full_path = (
            PROJECT_ROOT / path
        )

        if not full_path.exists():
            missing.append(path)

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
            "Preprocessing verification failed."
        )

    print(
        f"Verified images: "
        f"{len(manifest_df)}"
    )

    print(
        "All processed files exist. ✅"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print()
    print("=" * 70)
    print(
        "VISIONMITRA - APTOS OFFLINE PREPROCESSING"
    )
    print("=" * 70)

    print()
    print(
        "Pipeline:"
    )

    print(
        "Original Image"
    )
    print(
        "      ↓"
    )
    print(
        "Quality Assessment"
    )
    print(
        "      ↓"
    )
    print(
        "Adaptive Enhancement"
    )
    print(
        "      ↓"
    )
    print(
        "Saved Processed Image"
    )

    print()
    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Output directory:\n"
        f"{PROCESSED_DIR}"
    )

    # --------------------------------------------------------
    # Load dataframe
    # --------------------------------------------------------

    df = load_aptos_dataframe()

    print()
    print(
        f"Total labeled images: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Create exact same split
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df
    ) = create_splits(
        df,
        random_seed=RANDOM_SEED
    )

    print()
    print(
        "Fixed split:"
    )

    print(
        f"Train      : "
        f"{len(train_df)}"
    )

    print(
        f"Validation : "
        f"{len(val_df)}"
    )

    print(
        f"Test       : "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        (
            PROCESSED_DIR / split_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_rows = []

    # --------------------------------------------------------
    # Process train
    # --------------------------------------------------------

    process_split(
        train_df,
        "train",
        PROCESSED_DIR,
        manifest_rows
    )

    # --------------------------------------------------------
    # Process validation
    # --------------------------------------------------------

    process_split(
        val_df,
        "validation",
        PROCESSED_DIR,
        manifest_rows
    )

    # --------------------------------------------------------
    # Process test
    # --------------------------------------------------------

    process_split(
        test_df,
        "test",
        PROCESSED_DIR,
        manifest_rows
    )

    # --------------------------------------------------------
    # Create manifest
    # --------------------------------------------------------

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        PROCESSED_DIR
        / "manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MANIFEST SAVED"
    )

    print(
        manifest_path
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_outputs(
        manifest_df
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL QUALITY SUMMARY")
    print("=" * 70)

    print()

    quality_summary = pd.crosstab(
        manifest_df["split"],
        manifest_df["quality_status"]
    )

    print(
        quality_summary
    )

    print()
    print(
        "Total processed:"
    )

    print(
        len(manifest_df)
    )

    # --------------------------------------------------------
    # Final timing
    # --------------------------------------------------------

    total_elapsed = (
        time.time()
        - total_start
    )

    print()
    print("=" * 70)
    print(
        "PREPROCESSING COMPLETE ✅"
    )
    print("=" * 70)

    print()
    print(
        f"Total time: "
        f"{total_elapsed / 60:.1f} minutes"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Modify dataset_loader.py to read "
        "these saved processed images."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()