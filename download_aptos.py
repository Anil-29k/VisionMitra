from pathlib import Path

import kagglehub
import shutil


# ==========================================
# VisionMitra - APTOS 2019 Dataset Download
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "APTOS-19"
)

DATASET_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Download competition files
download_path = kagglehub.competition_download(
    "aptos2019-blindness-detection"
)

download_path = Path(download_path)

print("Kaggle download location:")
print(download_path)


# Copy downloaded files into VisionMitra/data/APTOS-19
for item in download_path.iterdir():

    destination = DATASET_DIR / item.name

    if item.is_dir():

        if destination.exists():
            shutil.rmtree(destination)

        shutil.copytree(
            item,
            destination
        )

    else:

        shutil.copy2(
            item,
            destination
        )


print()
print("==========================================")
print("      APTOS-19 DOWNLOAD COMPLETE")
print("==========================================")
print()
print("Dataset stored at:")
print(DATASET_DIR)