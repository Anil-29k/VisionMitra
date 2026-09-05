from pathlib import Path

import pandas as pd
from PIL import Image


# ==========================================
# VisionMitra - APTOS 2019 Dataset Inspection
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "APTOS-19"
)

TRAIN_CSV = DATASET_DIR / "train.csv"
TRAIN_IMAGES = DATASET_DIR / "train_images"


# ==========================================
# 1. CHECK DATASET STRUCTURE
# ==========================================

def check_dataset():

    print()
    print("=" * 60)
    print("       VISIONMITRA - APTOS DATASET CHECK")
    print("=" * 60)

    print()
    print("Dataset location:")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_DIR}"
        )

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"train.csv not found:\n{TRAIN_CSV}"
        )

    if not TRAIN_IMAGES.exists():
        raise FileNotFoundError(
            f"train_images folder not found:\n{TRAIN_IMAGES}"
        )

    print("\nDataset structure: OK ✅")


# ==========================================
# 2. LOAD TRAINING CSV
# ==========================================

def load_training_data():

    df = pd.read_csv(
        TRAIN_CSV
    )

    print()
    print("------------------------------------------")
    print("Training CSV")
    print("------------------------------------------")

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {list(df.columns)}"
    )

    print()
    print("First 5 records:")

    print(
        df.head()
    )

    return df


# ==========================================
# 3. CHECK DR CLASS DISTRIBUTION
# ==========================================

def show_class_distribution(df):

    print()
    print("------------------------------------------")
    print("DR Grade Distribution")
    print("------------------------------------------")

    distribution = (
        df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    grade_names = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }

    total = len(df)

    for grade, count in distribution.items():

        percentage = (
            count / total * 100
        )

        name = grade_names.get(
            grade,
            "Unknown"
        )

        print(
            f"Grade {grade} "
            f"({name:18}) : "
            f"{count:5} "
            f"({percentage:6.2f}%)"
        )


# ==========================================
# 4. CHECK IMAGE FILES
# ==========================================

def check_image_files(df):

    print()
    print("------------------------------------------")
    print("Image File Validation")
    print("------------------------------------------")

    missing = []
    found = 0

    for image_id in df["id_code"]:

        image_path = (
            TRAIN_IMAGES
            / f"{image_id}.png"
        )

        if image_path.exists():
            found += 1
        else:
            missing.append(
                image_id
            )

    print(
        f"Images found   : {found}"
    )

    print(
        f"Images missing : {len(missing)}"
    )

    if missing:

        print()
        print("First missing images:")

        for image_id in missing[:10]:

            print(
                f"  - {image_id}"
            )

    else:

        print(
            "CSV ↔ image mapping: OK ✅"
        )


# ==========================================
# 5. INSPECT SAMPLE IMAGES
# ==========================================

def inspect_sample_images(df):

    print()
    print("------------------------------------------")
    print("Sample Image Inspection")
    print("------------------------------------------")

    sample_ids = df["id_code"].head(5)

    for image_id in sample_ids:

        image_path = (
            TRAIN_IMAGES
            / f"{image_id}.png"
        )

        try:

            with Image.open(
                image_path
            ) as image:

                print(
                    f"{image_id}.png"
                    f" → "
                    f"{image.size[0]} x "
                    f"{image.size[1]} "
                    f"| Mode: {image.mode}"
                )

        except Exception as error:

            print(
                f"❌ Could not read "
                f"{image_id}.png"
            )

            print(
                f"   Reason: {error}"
            )


# ==========================================
# 6. MAIN
# ==========================================

def main():

    try:

        check_dataset()

        df = load_training_data()

        show_class_distribution(
            df
        )

        check_image_files(
            df
        )

        inspect_sample_images(
            df
        )

        print()
        print("=" * 60)
        print(
            "       APTOS INSPECTION COMPLETE"
        )
        print("=" * 60)

    except Exception as error:

        print()
        print(
            f"❌ Inspection failed:"
        )

        print(error)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()