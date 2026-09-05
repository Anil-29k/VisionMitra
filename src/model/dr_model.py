import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


# ==========================================
# VisionMitra - Diabetic Retinopathy Model
# ==========================================

NUM_CLASSES = 5

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}


# ==========================================
# 1. CREATE MODEL
# ==========================================

def create_model(
    num_classes=NUM_CLASSES,
    pretrained=True
):
    """
    Create a ResNet18-based DR classifier.

    The ImageNet-pretrained backbone is used as
    the starting point and the final classifier
    is replaced with a 5-class DR classifier.
    """

    if pretrained:

        weights = ResNet18_Weights.DEFAULT

    else:

        weights = None

    model = resnet18(
        weights=weights
    )

    # Number of features entering the
    # original classification layer.
    num_features = (
        model.fc.in_features
    )

    # Replace ImageNet classifier with
    # our 5-class DR classifier.
    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    return model


# ==========================================
# 2. FREEZE BACKBONE
# ==========================================

def freeze_backbone(model):
    """
    Freeze the ResNet backbone.

    Only the final classifier remains trainable.
    """

    for parameter in model.parameters():

        parameter.requires_grad = False

    # Make the final classifier trainable.
    for parameter in model.fc.parameters():

        parameter.requires_grad = True

    return model


# ==========================================
# 3. UNFREEZE BACKBONE
# ==========================================

def unfreeze_backbone(model):
    """
    Make the complete model trainable.

    This will be useful later for fine-tuning.
    """

    for parameter in model.parameters():

        parameter.requires_grad = True

    return model


# ==========================================
# 4. COUNT TRAINABLE PARAMETERS
# ==========================================

def count_parameters(model):
    """
    Count total and trainable parameters.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ==========================================
# 5. MODEL SUMMARY
# ==========================================

def print_model_summary(model):

    total, trainable = (
        count_parameters(model)
    )

    print()
    print("------------------------------------------")
    print("DR MODEL SUMMARY")
    print("------------------------------------------")

    print(
        f"Architecture       : ResNet18"
    )

    print(
        f"Output classes     : {NUM_CLASSES}"
    )

    print(
        f"Total parameters   : {total:,}"
    )

    print(
        f"Trainable params   : {trainable:,}"
    )

    print()

    print("Classes:")

    for index, name in CLASS_NAMES.items():

        print(
            f"  {index} → {name}"
        )


# ==========================================
# 6. MODEL TEST
# ==========================================

def main():

    print()
    print("=" * 60)
    print("       VISIONMITRA - DR MODEL TEST")
    print("=" * 60)

    print()
    print("Creating pretrained ResNet18...")

    model = create_model(
        pretrained=True
    )

    print(
        "Model created successfully. ✅"
    )

    # Freeze backbone for the first
    # training stage.
    model = freeze_backbone(
        model
    )

    print(
        "Backbone frozen. ✅"
    )

    print_model_summary(
        model
    )

    # --------------------------------------
    # Test forward pass
    # --------------------------------------

    print()
    print("------------------------------------------")
    print("Forward Pass Test")
    print("------------------------------------------")

    dummy_input = torch.randn(
        2,
        3,
        224,
        224
    )

    model.eval()

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print(
        f"Input shape  : "
        f"{dummy_input.shape}"
    )

    print(
        f"Output shape : "
        f"{output.shape}"
    )

    expected_shape = (
        2,
        NUM_CLASSES
    )

    if tuple(output.shape) == expected_shape:

        print(
            "Output shape test: PASSED ✅"
        )

    else:

        print(
            "Output shape test: FAILED ❌"
        )

    # --------------------------------------
    # Prediction test
    # --------------------------------------

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

    print()
    print(
        "Sample predictions:"
    )

    for index in range(
        len(predictions)
    ):

        predicted_class = (
            predictions[index].item()
        )

        confidence = (
            probabilities[
                index,
                predicted_class
            ].item()
        )

        print(
            f"  Sample {index + 1}: "
            f"Grade {predicted_class} "
            f"({CLASS_NAMES[predicted_class]}) "
            f"- {confidence:.2%}"
        )

    print()
    print(
        "Model test complete. ✅"
    )


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()