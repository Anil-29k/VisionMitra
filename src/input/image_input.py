from pathlib import Path
from PIL import Image


# ==========================================
# VisionMitra - Image Input Module
# ==========================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def load_fundus_image(image_path):
    """
    Load and validate a fundus image.

    Returns:
        PIL.Image.Image: RGB image
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: "
            f"{image_path.suffix}"
        )

    try:
        image = Image.open(image_path)

        image.load()

        image = image.convert("RGB")

    except Exception as e:
        raise ValueError(
            f"Unable to read image: {e}"
        )

    return image


def get_image_info(image):
    """
    Get basic image information.
    """

    width, height = image.size

    return {
        "width": width,
        "height": height,
        "channels": 3,
        "mode": image.mode
    }