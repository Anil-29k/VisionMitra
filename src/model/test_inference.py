import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model.dr_model import create_model, CLASS_NAMES


# ==========================================
# VisionMitra - Official APTOS Test Inference
# ==========================================

NUM_CLASSES = 5

MODEL_PATH = (
    "data/models/visionmitra_resnet18_v2_best.pth"
)

DATASET_DIR = Path("data/APTOS-19")

TEST_CSV = DATASET_DIR / "test.csv"
TEST_IMAGES = DATASET_DIR / "test_images"

OUTPUT_DIR = Path("data/predictions")
OUTPUT_FILE = OUTPUT_DIR / "visionmitra_aptos_test_predictions.csv"

IMAGE_SIZE = 224


def get_test_transform():

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_test_ids():

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"test.csv not found:\n{TEST_CSV}"
        )

    test_ids = []

    with open(
        TEST_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        if "id_code" not in reader.fieldnames:
            raise ValueError(
                "test.csv must contain an 'id_code' column."
            )

        for row in reader:
            test_ids.append(
                row["id_code"]
            )

    return test_ids


def main():

    print()
    print("=" * 60)
    print("   VISIONMITRA - OFFICIAL APTOS TEST INFERENCE")
    print("=" * 60)

    # --------------------------------------
    # Device
    # --------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    # --------------------------------------
    # Check files
    # --------------------------------------

    print()
    print("Checking test dataset...")

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n{TEST_IMAGES}"
        )

    test_ids = load_test_ids()

    print(
        f"Test images listed in CSV: {len(test_ids)}"
    )

    # --------------------------------------
    # Verify images
    # --------------------------------------

    missing_images = []

    for image_id in test_ids:

        image_path = (
            TEST_IMAGES / f"{image_id}.png"
        )

        if not image_path.exists():
            missing_images.append(
                image_id
            )

    if missing_images:

        raise FileNotFoundError(
            f"{len(missing_images)} test image(s) are missing."
        )

    print(
        "All test images found. ✅"
    )

    # --------------------------------------
    # Load model
    # --------------------------------------

    print()
    print("Loading V2 model...")

    model = create_model(
        pretrained=False
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    print(
        "V2 model loaded successfully. ✅"
    )

    print(
        "Model checkpoint:"
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------
    # Transform
    # --------------------------------------

    transform = get_test_transform()

    # --------------------------------------
    # Output directory
    # --------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------
    # Inference
    # --------------------------------------

    print()
    print("=" * 60)
    print("STARTING OFFICIAL TEST INFERENCE")
    print("=" * 60)

    predictions = []

    with torch.no_grad():

        for index, image_id in enumerate(test_ids):

            image_path = (
                TEST_IMAGES / f"{image_id}.png"
            )

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

            except Exception as error:

                raise RuntimeError(
                    f"Unable to read test image:\n"
                    f"{image_path}\n"
                    f"Reason: {error}"
                )

            image_tensor = transform(
                image
            )

            image_tensor = (
                image_tensor
                .unsqueeze(0)
                .to(device)
            )

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

            predicted_grade = (
                prediction.item()
            )

            predicted_confidence = (
                confidence.item()
            )

            predictions.append({
                "id_code": image_id,
                "diagnosis": predicted_grade,
                "predicted_class": CLASS_NAMES[
                    predicted_grade
                ],
                "confidence": predicted_confidence
            })

            if (
                (index + 1) % 100 == 0
                or index == 0
                or index + 1 == len(test_ids)
            ):

                print(
                    f"Processed "
                    f"{index + 1}/{len(test_ids)}"
                )

    # --------------------------------------
    # Save predictions
    # --------------------------------------

    print()
    print("Saving predictions...")

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "id_code",
            "diagnosis",
            "predicted_class",
            "confidence"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            predictions
        )

    # --------------------------------------
    # Summary
    # --------------------------------------

    print()
    print("=" * 60)
    print("OFFICIAL TEST INFERENCE COMPLETE ✅")
    print("=" * 60)

    print()
    print(
        f"Images processed: {len(predictions)}"
    )

    print()
    print("Prediction distribution:")

    distribution = {
        grade: 0
        for grade in range(NUM_CLASSES)
    }

    for result in predictions:

        distribution[
            result["diagnosis"]
        ] += 1

    for grade in range(NUM_CLASSES):

        print(
            f"  Grade {grade} "
            f"({CLASS_NAMES[grade]}): "
            f"{distribution[grade]}"
        )

    print()
    print("Predictions saved to:")

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "OFFICIAL APTOS TEST LABELS WERE NOT USED 🔒"
    )


if __name__ == "__main__":
    main()