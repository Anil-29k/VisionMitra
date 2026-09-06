"""
VisionMitra - DR Model Training V4

V4:
    Enhanced APTOS images
    ResNet18 fine-tuning
    Separate learning rates for backbone and FC layer

Dataset:
    APTOS 2019

Split:
    70% Train
    15% Validation
    15% Test

IMPORTANT:
    The test set is NEVER used during training or model selection.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset.dataset_loader import create_dataloaders
from src.model.dr_model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5

BATCH_SIZE = 16

NUM_EPOCHS = 10

BACKBONE_LR = 0.0001
FC_LR = 0.001

RANDOM_SEED = 42


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_df):
    """
    Calculate inverse-frequency class weights
    from the training set only.
    """

    class_counts = (
        train_df["diagnosis"]
        .value_counts()
        .sort_index()
    )

    total = len(train_df)

    weights = []

    for class_index in range(NUM_CLASSES):

        count = class_counts.get(
            class_index,
            0
        )

        if count == 0:

            weights.append(0.0)

        else:

            weight = (
                total
                / (NUM_CLASSES * count)
            )

            weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    val_loss = running_loss / total
    val_accuracy = correct / total

    return (
        val_loss,
        val_accuracy
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("       VISIONMITRA - DR MODEL TRAINING V4")
    print("=" * 70)

    start_time = time.time()

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
    # RANDOM SEED
    # ========================================================

    torch.manual_seed(
        RANDOM_SEED
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print("Loading APTOS dataset...")
    print()

    print("Dataset split:")
    print("  70% → Training")
    print("  15% → Validation")
    print("  15% → Test")
    print()

    print(
        "Test set will remain LOCKED during training. 🔒"
    )

    (
        train_loader,
        val_loader,
        test_loader,
        train_df,
        val_df,
        test_df
    ) = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    print()
    print("Dataset loaded successfully. ✅")

    print()
    print(
        f"Training images:   "
        f"{len(train_df)}"
    )

    print(
        f"Validation images: "
        f"{len(val_df)}"
    )

    print(
        f"Test images:       "
        f"{len(test_df)}"
    )

    print()
    print(
        "Test loader created but will NOT "
        "be used during training. 🔒"
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = calculate_class_weights(
        train_df
    )

    class_weights = class_weights.to(
        device
    )

    print()
    print("Class weights:")

    for index, weight in enumerate(
        class_weights
    ):

        print(
            f"  Grade {index}: "
            f"{weight.item():.4f}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("Creating pretrained ResNet18...")

    model = create_model(
        pretrained=True
    )

    # ========================================================
    # V4: FULL FINE-TUNING
    # ========================================================

    print()
    print(
        "V4 fine-tuning enabled. 🔓"
    )

    print(
        "Backbone learning rate:",
        BACKBONE_LR
    )

    print(
        "FC learning rate:",
        FC_LR
    )

    # Make every parameter trainable.
    for parameter in model.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    print()
    print("Model ready. ✅")

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print()
    print(
        "Loss: Weighted CrossEntropyLoss ✅"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.Adam(
        [
            {
                "params": model.conv1.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.bn1.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.layer1.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.layer2.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.layer3.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.layer4.parameters(),
                "lr": BACKBONE_LR
            },
            {
                "params": model.fc.parameters(),
                "lr": FC_LR
            }
        ]
    )

    print(
        "Optimizer: Adam ✅"
    )

    # ========================================================
    # MODEL DIRECTORY
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    model_dir = (
        project_root
        / "data"
        / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # V4 CHECKPOINT
    # ========================================================

    best_model_path = (
        model_dir
        / "visionmitra_resnet18_v4_best.pth"
    )

    print()
    print(
        "Best V4 model will be saved to:"
    )

    print(
        best_model_path
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING TRAINING V4")
    print("=" * 70)

    print()
    print(
        "Full backbone fine-tuning: ENABLED 🔓"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        f"Backbone LR: {BACKBONE_LR}"
    )

    print(
        f"FC LR:       {FC_LR}"
    )

    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(
        NUM_EPOCHS
    ):

        epoch_start = time.time()

        print()
        print(
            f"Epoch "
            f"{epoch + 1}/{NUM_EPOCHS}"
        )

        print(
            "-" * 70
        )

        # ====================================================
        # TRAIN
        # ====================================================

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ====================================================
        # METRICS
        # ====================================================

        print(
            f"Train Loss:      "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy:  "
            f"{train_accuracy:.2%}"
        )

        print(
            f"Val Loss:        "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy:    "
            f"{val_accuracy:.2%}"
        )

        print(
            f"Epoch Time:      "
            f"{epoch_time:.1f} seconds"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print()
            print(
                "⭐ New best V4 model saved!"
            )

            print(
                f"   Validation accuracy: "
                f"{val_accuracy:.2%}"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    total_time = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print("TRAINING V4 COMPLETE ✅")
    print("=" * 70)

    print()

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2%}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print()

    print(
        "Model saved to:"
    )

    print(
        best_model_path
    )

    print()

    print(
        f"Total training time: "
        f"{total_time / 60:.1f} minutes"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The test set was NOT used "
        "during training or model selection. 🔒"
    )

    print(
        "Run the separate V4 evaluation "
        "script after training."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":

    main()