```python
import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from src.dataset.dataset_loader import create_dataloaders
from src.model.dr_model import (
    create_model,
    CLASS_NAMES
)


# ============================================================
# VisionMitra - V3 Locked Test Evaluation
# ============================================================

MODEL_PATH = (
    "data/models/visionmitra_resnet18_v3_best.pth"
)

NUM_CLASSES = 5


def main():

    print()
    print("=" * 70)
    print("       VISIONMITRA - V3 TEST EVALUATION")
    print("=" * 70)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print()
    print("Loading APTOS dataset...")

    (
        train_loader,
        val_loader,
        test_loader,
        train_df,
        val_df,
        test_df
    ) = create_dataloaders(
        batch_size=16,
        random_seed=42
    )

    print()
    print("Dataset loaded successfully. ✅")

    print(
        f"Training images:   {len(train_df)}"
    )

    print(
        f"Validation images: {len(val_df)}"
    )

    print(
        f"Locked test images: {len(test_df)}"
    )

    print()
    print(
        "Validation set was used for model selection."
    )

    print(
        "Locked test set is being used ONLY for final evaluation. 🔒"
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print()
    print("Loading V3 trained model...")

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
        "Best V3 model loaded successfully. ✅"
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print()
    print("Running predictions on locked test set...")

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    print(
        "Test prediction complete. ✅"
    )

    print(
        f"Evaluated images: {len(all_labels)}"
    )

    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_predictions
        )
    )

    print()
    print("=" * 70)
    print("OVERALL TEST PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy:          {accuracy:.2%}"
    )

    print(
        f"Balanced Accuracy: {balanced_accuracy:.2%}"
    )

    # ========================================================
    # PER-CLASS PERFORMANCE
    # ========================================================

    print()
    print("=" * 70)
    print("PER-CLASS TEST PERFORMANCE")
    print("=" * 70)

    target_names = [
        CLASS_NAMES[index]
        for index in range(NUM_CLASSES)
    ]

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print()
    print("Rows = Actual")
    print("Columns = Predicted")
    print()

    print(
        "Actual       "
        "  0    1    2    3    4"
    )

    print("-" * 45)

    for index, row in enumerate(matrix):

        print(
            f"Grade {index}      "
            + " ".join(
                f"{value:4d}"
                for value in row
            )
        )

    # ========================================================
    # REFERABLE DR SCREENING
    # ========================================================

    # VisionMitra screening grouping:
    #
    # Grade 0-1 → Non-referable
    # Grade 2-4 → Referable

    true_referable = (
        all_labels >= 2
    )

    predicted_referable = (
        all_predictions >= 2
    )

    referable_accuracy = (
        accuracy_score(
            true_referable,
            predicted_referable
        )
    )

    referable_matrix = confusion_matrix(
        true_referable,
        predicted_referable,
        labels=[False, True]
    )

    tn = referable_matrix[0, 0]
    fp = referable_matrix[0, 1]
    fn = referable_matrix[1, 0]
    tp = referable_matrix[1, 1]

    if (tp + fn) > 0:

        referable_sensitivity = (
            tp / (tp + fn)
        )

    else:

        referable_sensitivity = 0.0

    if (tn + fp) > 0:

        referable_specificity = (
            tn / (tn + fp)
        )

    else:

        referable_specificity = 0.0

    print()
    print("=" * 70)
    print("REFERABLE DR SCREENING - TEST SET")
    print("=" * 70)

    print()
    print(
        "Grade 0-1 → Non-referable"
    )

    print(
        "Grade 2-4 → Referable"
    )

    print()

    print(
        f"Accuracy:    "
        f"{referable_accuracy:.2%}"
    )

    print(
        f"Sensitivity: "
        f"{referable_sensitivity:.2%}"
    )

    print(
        f"Specificity: "
        f"{referable_specificity:.2%}"
    )

    print()

    print(
        f"TP: {tp}"
    )

    print(
        f"TN: {tn}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("SIH SCREENING TARGET CHECK")
    print("=" * 70)

    print()
    print(
        "Target sensitivity: >90%"
    )

    print(
        "Target specificity: >85%"
    )

    print()

    if referable_sensitivity > 0.90:

        print(
            "Sensitivity target: PASSED ✅"
        )

    else:

        print(
            "Sensitivity target: NOT YET ACHIEVED ⚠️"
        )

    if referable_specificity > 0.85:

        print(
            "Specificity target: PASSED ✅"
        )

    else:

        print(
            "Specificity target: NOT YET ACHIEVED ⚠️"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("V3 TEST EVALUATION COMPLETE ✅")
    print("=" * 70)

    print()

    print(
        "The locked 550-image test set was used only "
        "for final evaluation."
    )

    print(
        "These results should be treated as the V3 held-out "
        "test performance."
    )

    print()


if __name__ == "__main__":

    main()

