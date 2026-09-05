from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# VisionMitra - Image Quality Assessment
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = PROJECT_ROOT / "data" / "test"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==========================================
# 1. LOAD IMAGE
# ==========================================

def load_image(image_path):
    """
    Load a fundus image using OpenCV.

    Parameters
    ----------
    image_path : str or Path
        Path to the input fundus image.

    Returns
    -------
    numpy.ndarray
        Image in OpenCV BGR format.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    return image


# ==========================================
# 2. FIND RETINAL FIELD
# ==========================================

def create_retinal_mask(image):
    """
    Detect the visible retinal field and create
    a binary mask.

    The black background surrounding the retina
    is excluded from quality calculations.

    Returns
    -------
    numpy.ndarray
        Binary retinal mask.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove very dark background pixels.
    initial_mask = (
        gray > 10
    ).astype(np.uint8) * 255

    # Remove small gaps/noise.
    kernel = np.ones(
        (15, 15),
        np.uint8
    )

    initial_mask = cv2.morphologyEx(
        initial_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    initial_mask = cv2.morphologyEx(
        initial_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find connected regions.
    contours, _ = cv2.findContours(
        initial_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "Unable to detect retinal field."
        )

    # Largest region is assumed to be the retinal field.
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    mask = np.zeros_like(gray)

    cv2.drawContours(
        mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Slightly smooth the boundary.
    mask = cv2.GaussianBlur(
        mask,
        (9, 9),
        0
    )

    mask = (
        mask > 127
    ).astype(np.uint8) * 255

    return mask


# ==========================================
# 3. SHARPNESS / FOCUS
# ==========================================

def calculate_sharpness(image, mask):
    """
    Calculate sharpness inside the retinal field.

    Two complementary measurements are used:

    1. Laplacian variance
    2. Tenengrad gradient strength

    Returns
    -------
    tuple
        (laplacian_score, tenengrad_score)
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    retinal_pixels = gray[mask > 0]

    if retinal_pixels.size == 0:
        return 0.0, 0.0

    # Normalize contrast before measuring focus.
    normalized = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # ------------------------------------------
    # Laplacian variance
    # ------------------------------------------

    laplacian = cv2.Laplacian(
        normalized,
        cv2.CV_64F
    )

    laplacian_values = (
        laplacian[mask > 0]
    )

    laplacian_score = float(
        np.var(laplacian_values)
    )

    # ------------------------------------------
    # Tenengrad
    # ------------------------------------------

    sobel_x = cv2.Sobel(
        normalized,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        normalized,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = (
        sobel_x ** 2 +
        sobel_y ** 2
    )

    tenengrad_values = (
        gradient_magnitude[mask > 0]
    )

    tenengrad_score = float(
        np.mean(tenengrad_values)
    )

    return (
        laplacian_score,
        tenengrad_score
    )


# ==========================================
# 4. ILLUMINATION
# ==========================================

def calculate_illumination(image, mask):
    """
    Analyze brightness and illumination uniformity
    inside the retinal field.

    Returns
    -------
    tuple
        (mean_brightness, brightness_std, dark_ratio)
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    retinal_pixels = gray[mask > 0]

    if retinal_pixels.size == 0:
        return 0.0, 0.0, 0.0

    mean_brightness = float(
        np.mean(retinal_pixels)
    )

    brightness_std = float(
        np.std(retinal_pixels)
    )

    # Extremely dark retinal pixels.
    dark_pixels = np.sum(
        retinal_pixels < 20
    )

    dark_ratio = float(
        dark_pixels / retinal_pixels.size
    )

    return (
        mean_brightness,
        brightness_std,
        dark_ratio
    )


# ==========================================
# 5. FIELD OF VIEW
# ==========================================

def calculate_fov(mask):
    """
    Calculate the percentage of the image occupied
    by the detected retinal field.
    """

    retinal_area = np.count_nonzero(mask)

    total_area = mask.size

    if total_area == 0:
        return 0.0

    fov_ratio = float(
        retinal_area / total_area
    )

    return fov_ratio


# ==========================================
# 6. PROVISIONAL QUALITY DECISION
# ==========================================

def classify_quality(
    laplacian_score,
    tenengrad_score,
    mean_brightness,
    dark_ratio,
    fov_ratio
):
    """
    Classify fundus-image quality.

    Returns
    -------
    str
        GOOD, BORDERLINE, or POOR

    IMPORTANT
    ---------
    These thresholds are provisional and must be
    calibrated using a representative retinal-image
    dataset before any clinical performance claim.
    """

    # ------------------------------------------
    # Illumination checks
    # ------------------------------------------

    if mean_brightness < 25:
        return "POOR"

    if mean_brightness > 235:
        return "POOR"

    if dark_ratio > 0.35:
        return "POOR"

    # ------------------------------------------
    # Field-of-view check
    # ------------------------------------------

    if fov_ratio < 0.25:
        return "POOR"

    # ------------------------------------------
    # Focus checks
    # ------------------------------------------

    # Strong gradient detail + sufficient
    # Laplacian response.
    focus_good = (
        tenengrad_score >= 200
        and laplacian_score >= 8
    )

    # Moderate detail.
    focus_borderline = (
        tenengrad_score >= 60
        or laplacian_score >= 5
    )

    # ------------------------------------------
    # Final decision
    # ------------------------------------------

    if focus_good:
        return "GOOD"

    if focus_borderline:
        return "BORDERLINE"

    return "POOR"


# ==========================================
# 7. COMPLETE QUALITY ANALYSIS
# ==========================================

def analyze_image_data(image):
    """
    Analyze an already-loaded OpenCV image.

    This is the main function that the VisionMitra
    processing pipeline should use.
    """

    mask = create_retinal_mask(image)

    (
        laplacian_score,
        tenengrad_score
    ) = calculate_sharpness(
        image,
        mask
    )

    (
        mean_brightness,
        brightness_std,
        dark_ratio
    ) = calculate_illumination(
        image,
        mask
    )

    fov_ratio = calculate_fov(mask)

    status = classify_quality(
        laplacian_score,
        tenengrad_score,
        mean_brightness,
        dark_ratio,
        fov_ratio
    )

    return {
        "image": image,
        "mask": mask,
        "laplacian_score": laplacian_score,
        "tenengrad_score": tenengrad_score,
        "mean_brightness": mean_brightness,
        "brightness_std": brightness_std,
        "dark_ratio": dark_ratio,
        "fov_ratio": fov_ratio,
        "status": status
    }


# ==========================================
# 8. FILE-BASED QUALITY ANALYSIS
# ==========================================

def analyze_image(image_path):
    """
    Load an image from disk and perform complete
    quality analysis.

    Useful for standalone QA testing.
    """

    image = load_image(image_path)

    return analyze_image_data(image)


# ==========================================
# 9. DIAGNOSTIC VISUALIZATION
# ==========================================

def show_quality_diagnostics(result):
    """
    Display:

    1. Original fundus
    2. Detected retinal field
    3. Region used for quality analysis
    """

    image = result["image"]
    mask = result["mask"]

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    retinal_only = cv2.bitwise_and(
        rgb_image,
        rgb_image,
        mask=mask
    )

    plt.figure(
        figsize=(15, 5)
    )

    # Original
    plt.subplot(1, 3, 1)

    plt.imshow(rgb_image)

    plt.title(
        "Original Fundus"
    )

    plt.axis("off")

    # Retinal mask
    plt.subplot(1, 3, 2)

    plt.imshow(
        mask,
        cmap="gray"
    )

    plt.title(
        "Detected Retinal Field"
    )

    plt.axis("off")

    # Analysis region
    plt.subplot(1, 3, 3)

    plt.imshow(
        retinal_only
    )

    plt.title(
        "Quality Analysis Region"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ==========================================
# 10. STANDALONE TEST
# ==========================================

def main():

    print()
    print("==========================================")
    print("   VISIONMITRA - IMAGE QUALITY CHECK")
    print("==========================================")

    if not TEST_DIR.exists():

        print(
            "\nTest directory not found:"
        )

        print(TEST_DIR)

        return

    images = [
        path
        for path in TEST_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    if not images:

        print(
            "\nNo fundus images found."
        )

        print(
            "\nPlace a JPG, JPEG, or PNG image inside:"
        )

        print(TEST_DIR)

        return

    for image_path in images:

        try:

            result = analyze_image(
                image_path
            )

            print()
            print("------------------------------------------")
            print(
                f"Image: {image_path.name}"
            )
            print("------------------------------------------")

            print(
                f"Laplacian Score : "
                f"{result['laplacian_score']:.2f}"
            )

            print(
                f"Tenengrad Score : "
                f"{result['tenengrad_score']:.2f}"
            )

            print(
                f"Mean Brightness : "
                f"{result['mean_brightness']:.2f}"
            )

            print(
                f"Brightness Std  : "
                f"{result['brightness_std']:.2f}"
            )

            print(
                f"Dark Pixel Ratio: "
                f"{result['dark_ratio']:.2%}"
            )

            print(
                f"Retinal FOV     : "
                f"{result['fov_ratio']:.2%}"
            )

            print(
                f"Quality Status  : "
                f"{result['status']}"
            )

            show_quality_diagnostics(
                result
            )

        except Exception as error:

            print()
            print(
                f"❌ Failed to analyze "
                f"{image_path.name}"
            )

            print(
                f"Reason: {error}"
            )


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()