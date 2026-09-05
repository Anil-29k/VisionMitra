from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# VisionMitra - Fundus Image Enhancement
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_ROOT / "data" / "test"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


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
            f"Unsupported image format: "
            f"{image_path.suffix}"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Unable to read image: "
            f"{image_path}"
        )

    return image


# ==========================================
# 2. CREATE RETINAL MASK
# ==========================================

def create_retinal_mask(image):
    """
    Detect the visible retinal field.

    The black area outside the fundus is excluded
    from enhancement operations.
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Invalid or empty image."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove very dark background.
    mask = (
        gray > 10
    ).astype(np.uint8) * 255

    # Clean small holes and noise.
    kernel = np.ones(
        (15, 15),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Detect connected regions.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return mask

    # Largest region = retinal field.
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    retinal_mask = np.zeros_like(
        gray
    )

    cv2.drawContours(
        retinal_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Smooth mask boundary.
    retinal_mask = cv2.GaussianBlur(
        retinal_mask,
        (9, 9),
        0
    )

    retinal_mask = (
        retinal_mask > 127
    ).astype(np.uint8) * 255

    return retinal_mask


# ==========================================
# 3. ILLUMINATION NORMALIZATION
# ==========================================

def normalize_illumination(image, mask):
    """
    Correct uneven illumination while preserving
    the natural appearance of the fundus.

    A slowly varying illumination field is estimated
    using Gaussian smoothing.

    Subtractive correction is used instead of direct
    division to avoid unnatural color/brightness shifts.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    # Estimate slowly varying illumination.
    background = cv2.GaussianBlur(
        l_channel,
        (0, 0),
        sigmaX=35,
        sigmaY=35
    )

    # Calculate average retinal brightness.
    retinal_pixels = l_channel[
        mask > 0
    ]

    if retinal_pixels.size == 0:
        return image.copy()

    target_brightness = float(
        np.mean(retinal_pixels)
    )

    # Subtractive illumination correction.
    corrected_l = (
        l_channel.astype(np.float32)
        + target_brightness
        - background.astype(np.float32)
    )

    # Keep values inside valid image range.
    corrected_l = np.clip(
        corrected_l,
        0,
        255
    ).astype(np.uint8)

    # Preserve the black background.
    corrected_l[
        mask == 0
    ] = 0

    corrected_lab = cv2.merge(
        [
            corrected_l,
            a_channel,
            b_channel
        ]
    )

    normalized = cv2.cvtColor(
        corrected_lab,
        cv2.COLOR_LAB2BGR
    )

    normalized[
        mask == 0
    ] = 0

    return normalized


# ==========================================
# 4. CLAHE LOCAL CONTRAST ENHANCEMENT
# ==========================================

def apply_clahe(image, mask):
    """
    Improve local retinal contrast using CLAHE.

    CLAHE is applied only to the luminance channel
    to avoid independently altering RGB colors.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    # Gentle CLAHE settings.
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(
        l_channel
    )

    # Preserve background.
    enhanced_l[
        mask == 0
    ] = 0

    enhanced_lab = cv2.merge(
        [
            enhanced_l,
            a_channel,
            b_channel
        ]
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    enhanced[
        mask == 0
    ] = 0

    return enhanced


# ==========================================
# 5. DENOISING
# ==========================================

def denoise_image(image, mask):
    """
    Reduce image noise while preserving retinal
    structures such as vessels and lesions.
    """

    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Preserve black background.
    denoised[
        mask == 0
    ] = 0

    return denoised


# ==========================================
# 6. COMPLETE ENHANCEMENT PIPELINE
# ==========================================

def enhance_fundus_image(image):
    """
    Run the complete fundus enhancement pipeline.

    Pipeline:

        Input Image
             ↓
        Retinal Mask
             ↓
        Illumination Normalization
             ↓
        CLAHE
             ↓
        Denoising
             ↓
        Enhanced Image

    Returns
    -------
    dict
        All intermediate processing stages.
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Invalid or empty image."
        )

    mask = create_retinal_mask(
        image
    )

    normalized = normalize_illumination(
        image,
        mask
    )

    contrast_enhanced = apply_clahe(
        normalized,
        mask
    )

    denoised = denoise_image(
        contrast_enhanced,
        mask
    )

    return {
        "original": image,
        "mask": mask,
        "normalized": normalized,
        "clahe": contrast_enhanced,
        "enhanced": denoised
    }


# ==========================================
# 7. VISUALIZATION
# ==========================================

def show_enhancement_results(results):
    """
    Display all enhancement stages.
    """

    original = cv2.cvtColor(
        results["original"],
        cv2.COLOR_BGR2RGB
    )

    normalized = cv2.cvtColor(
        results["normalized"],
        cv2.COLOR_BGR2RGB
    )

    clahe = cv2.cvtColor(
        results["clahe"],
        cv2.COLOR_BGR2RGB
    )

    enhanced = cv2.cvtColor(
        results["enhanced"],
        cv2.COLOR_BGR2RGB
    )

    plt.figure(
        figsize=(16, 10)
    )

    # Original
    plt.subplot(2, 2, 1)

    plt.imshow(original)

    plt.title(
        "Original Fundus"
    )

    plt.axis("off")

    # Illumination normalization
    plt.subplot(2, 2, 2)

    plt.imshow(normalized)

    plt.title(
        "Illumination Normalized"
    )

    plt.axis("off")

    # CLAHE
    plt.subplot(2, 2, 3)

    plt.imshow(clahe)

    plt.title(
        "CLAHE Enhanced"
    )

    plt.axis("off")

    # Final
    plt.subplot(2, 2, 4)

    plt.imshow(enhanced)

    plt.title(
        "Final Enhanced Image"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ==========================================
# 8. SAVE IMAGE
# ==========================================

def save_enhanced_image(
    image,
    original_path
):
    """
    Save the final enhanced image.

    Returns
    -------
    Path
        Path of the saved image.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    original_path = Path(
        original_path
    )

    output_path = (
        OUTPUT_DIR
        / f"{original_path.stem}_enhanced.png"
    )

    success = cv2.imwrite(
        str(output_path),
        image
    )

    if not success:
        raise IOError(
            f"Unable to save image: "
            f"{output_path}"
        )

    return output_path


# ==========================================
# 9. STANDALONE TEST
# ==========================================

def main():

    print()
    print("==========================================")
    print("   VISIONMITRA - IMAGE ENHANCEMENT")
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

            print(
                f"\nProcessing: "
                f"{image_path.name}"
            )

            image = load_image(
                image_path
            )

            results = enhance_fundus_image(
                image
            )

            output_path = save_enhanced_image(
                results["enhanced"],
                image_path
            )

            print(
                "\nEnhanced image saved to:"
            )

            print(output_path)

            show_enhancement_results(
                results
            )

        except Exception as error:

            print(
                f"\n❌ Enhancement failed "
                f"for {image_path.name}"
            )

            print(
                f"Reason: {error}"
            )


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()