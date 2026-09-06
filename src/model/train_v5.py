"""
VisionMitra - DR Model Training V5

V5:
    Enhanced APTOS images
    ResNet18 full fine-tuning
    Lower learning rates
    Cosine learning-rate scheduler
    Weighted CrossEntropyLoss
    Resume training from checkpoint
    Checkpoint saved after every epoch
    Best model saved separately

IMPORTANT:
    The 15% test set is NEVER used during training or model selection.
"""

import time
import argparse
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

NUM_EPOCHS = 15

# Lower than V4
BACKBONE_LR = 0.00003
FC_LR = 0.0003

RANDOM_SEED = 42

# Resume checkpoint
RESUME_CHECKPOINT_NAME = (
    "visionmitra_resnet18_v5_checkpoint.pth"
)

BEST_MODEL_NAME = (
    "visionmitra_resnet18_v5_best.pth"
)


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
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_accuracy,
    best_epoch
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_accuracy":
            best_val_accuracy,

        "best_epoch":
            best_epoch,

        "random_seed":
            RANDOM_SEED
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device
):

    print()
    print("=" * 70)
    print("RESUME CHECKPOINT FOUND 🔄")
    print("=" * 70)

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = (
        checkpoint["epoch"] + 1
    )

    best_val_accuracy = checkpoint.get(
        "best_val_accuracy",
        0.0
    )

    best_epoch = checkpoint.get(
        "best_epoch",
        0
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch'] + 1}"
    )

    print(
        f"Resuming from epoch: "
        f"{start_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.2%}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print("=" * 70)

    return (
        start_epoch,
        best_val_accuracy,
        best_epoch
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing V5 checkpoint and start fresh."
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("       VISIONMITRA - DR MODEL TRAINING V5")
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
    print("  15% → LOCKED TEST")
    print()

    print(
        "Test set will remain LOCKED "
        "during training. 🔒"
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
    # FULL FINE-TUNING
    # ========================================================

    print()
    print(
        "V5 full fine-tuning enabled. 🔓"
    )

    print(
        "Backbone learning rate:",
        BACKBONE_LR
    )

    print(
        "FC learning rate:",
        FC_LR
    )

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
    # SCHEDULER
    # ========================================================

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )

    print(
        "Scheduler: CosineAnnealingLR ✅"
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

    checkpoint_path = (
        model_dir
        / RESUME_CHECKPOINT_NAME
    )

    best_model_path = (
        model_dir
        / BEST_MODEL_NAME
    )

    # ========================================================
    # RESUME
    # ========================================================

    start_epoch = 0
    best_val_accuracy = 0.0
    best_epoch = 0

    if (
        checkpoint_path.exists()
        and not args.fresh
    ):

        (
            start_epoch,
            best_val_accuracy,
            best_epoch
        ) = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            device
        )

    else:

        if args.fresh:

            print()
            print(
                "Fresh training requested. "
                "Existing checkpoint ignored."
            )

        else:

            print()
            print(
                "No previous checkpoint found."
            )

        print(
            "Starting V5 from epoch 1."
        )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING TRAINING V5")
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

    print()
    print(
        "Checkpoint will be saved after EVERY epoch. 💾"
    )

    print(
        "Best model will be saved separately. ⭐"
    )

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
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

        # ====================================================
        # CURRENT LR
        # ====================================================

        current_backbone_lr = (
            optimizer.param_groups[0]["lr"]
        )

        current_fc_lr = (
            optimizer.param_groups[-1]["lr"]
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
            f"Backbone LR:     "
            f"{current_backbone_lr:.8f}"
        )

        print(
            f"FC LR:           "
            f"{current_fc_lr:.8f}"
        )

        print(
            f"Epoch Time:      "
            f"{epoch_time:.1f} seconds"
        )

        # ====================================================
        # BEST MODEL
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
                "⭐ New best V5 model saved!"
            )

            print(
                f"   Validation accuracy: "
                f"{val_accuracy:.2%}"
            )

            print(
                f"   Epoch: "
                f"{best_epoch}"
            )

        # ====================================================
        # SCHEDULER STEP
        # ====================================================

        scheduler.step()

        # ====================================================
        # SAVE RECOVERY CHECKPOINT
        # ====================================================

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_accuracy,
            best_epoch
        )

        print()
        print(
            "💾 Recovery checkpoint saved."
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
    print("TRAINING V5 COMPLETE ✅")
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
        "Best model saved to:"
    )

    print(
        best_model_path
    )

    print()

    print(
        "Resume checkpoint saved to:"
    )

    print(
        checkpoint_path
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

    print()

    print(
        "Run the separate V5 evaluation "
        "script after training."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()