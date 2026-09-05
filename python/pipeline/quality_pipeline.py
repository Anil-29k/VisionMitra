from pathlib import Path
import json
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# VisionMitra - Quality + Enhancement Pipeline
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_ROOT / "data" / "test"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


# ==========================================
# 1. PROJECT IMPORT PATH
# ==========================================

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ==========================================
# 2. IMPORT EXISTING MODULES
# ==========================================

from python.input.image_input import (
    load_fundus_image,
    get_image_info
)

from python.quality.quality_assessment import (
    analyze_image_data
)

from python.preprocessing.enhancement import (
    enhance_fundus_image,
    save_enhanced_image
)


# ==========================================
# 3. PIL → OPENCV
# ==========================================

def pil_to_opencv(image):
    """
    Convert a PIL RGB image into OpenCV BGR format.
    """

    rgb = np.array(image)

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return bgr


# ==========================================
# 4. FIND TEST IMAGES
# ==========================================

def find_test_images():
    """
    Find supported fundus images inside
    data/test.
    """

    if not TEST_DIR.exists():
        return []

    images = [
        path
        for path in TEST_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    return sorted(images)


# ==========================================
# 5. PRINT QUALITY RESULT
# ==========================================

def print_quality_result(
    result,
    stage
):
    """
    Print quality metrics in a readable format.
    """

    print()
    print("------------------------------------------")
    print(stage)
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


# ==========================================
# 6. BUILD QUALITY REPORT
# ==========================================

def quality_metrics(result):
    """
    Extract only serializable quality metrics.
    """

    return {
        "laplacian_score": result[
            "laplacian_score"
        ],
        "tenengrad_score": result[
            "tenengrad_score"
        ],
        "mean_brightness": result[
            "mean_brightness"
        ],
        "brightness_std": result[
            "brightness_std"
        ],
        "dark_ratio": result[
            "dark_ratio"
        ],
        "fov_ratio": result[
            "fov_ratio"
        ],
        "status": result[
            "status"
        ]
    }


# ==========================================
# 7. SAVE PIPELINE REPORT
# ==========================================

def save_pipeline_report(
    image_path,
    initial_result,
    final_result,
    action,
    enhanced_path=None
):
    """
    Save pipeline results as JSON.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        PROCESSED_DIR
        / f"{image_path.stem}_pipeline.json"
    )

    report = {
        "project": "VisionMitra",

        "input_image": image_path.name,

        "initial_quality": quality_metrics(
            initial_result
        ),

        "final_quality": (
            quality_metrics(final_result)
            if final_result is not None
            else None
        ),

        "action": action,

        "enhanced_image": (
            str(enhanced_path)
            if enhanced_path is not None
            else None
        )
    }

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    return report_path


# ==========================================
# 8. SHOW ORIGINAL VS ENHANCED
# ==========================================

def show_comparison(
    original,
    enhanced
):
    """
    Display original and final enhanced images.
    """

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    enhanced_rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(
        figsize=(14, 6)
    )

    # Original
    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        original_rgb
    )

    plt.title(
        "Original Fundus"
    )

    plt.axis(
        "off"
    )

    # Enhanced
    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        enhanced_rgb
    )

    plt.title(
        "Final Enhanced Fundus"
    )

    plt.axis(
        "off"
    )

    plt.suptitle(
        "VisionMitra - Quality + Enhancement"
    )

    plt.tight_layout()

    plt.show()


# ==========================================
# 9. PROCESS ONE IMAGE
# ==========================================

def process_image(image_path):

    print()
    print("=" * 60)
    print("             VISIONMITRA PIPELINE")
    print("=" * 60)

    print(
        f"\nInput Image: "
        f"{image_path.name}"
    )

    # --------------------------------------
    # STEP 1 - INPUT
    # --------------------------------------

    print()
    print("[1] Loading fundus image...")

    pil_image = load_fundus_image(
        image_path
    )

    info = get_image_info(
        pil_image
    )

    print(
        f"Resolution : "
        f"{info['width']} x {info['height']}"
    )

    print(
        f"Channels   : "
        f"{info['channels']}"
    )

    print(
        f"Mode       : "
        f"{info['mode']}"
    )

    original = pil_to_opencv(
        pil_image
    )

    print(
        "Image loaded successfully. ✅"
    )


    # --------------------------------------
    # STEP 2 - INITIAL QUALITY CHECK
    # --------------------------------------

    print()
    print(
        "[2] Running initial quality assessment..."
    )

    initial_result = analyze_image_data(
        original
    )

    print_quality_result(
        initial_result,
        "INITIAL QUALITY"
    )


    # --------------------------------------
    # STEP 3 - USABILITY DECISION
    # --------------------------------------

    if initial_result["status"] == "POOR":

        print()
        print(
            "🔴 Image is not suitable "
            "for further processing."
        )

        print(
            "Action: RECAPTURE IMAGE."
        )

        report_path = save_pipeline_report(
            image_path,
            initial_result,
            None,
            "RECAPTURE"
        )

        print()
        print(
            f"Pipeline report saved:\n"
            f"{report_path}"
        )

        return


    # --------------------------------------
    # STEP 4 - ENHANCEMENT
    # --------------------------------------

    print()

    if initial_result["status"] == "GOOD":

        print(
            "🟢 Initial image quality: GOOD"
        )

    else:

        print(
            "🟡 Initial image quality: "
            "BORDERLINE"
        )

    print(
        "Action: Applying enhancement..."
    )

    enhancement_result = enhance_fundus_image(
        original
    )

    enhanced = enhancement_result[
        "enhanced"
    ]

    print(
        "Enhancement completed. ✅"
    )


    # --------------------------------------
    # STEP 5 - SAVE ENHANCED IMAGE
    # --------------------------------------

    enhanced_path = save_enhanced_image(
        enhanced,
        image_path
    )

    print()
    print(
        "Enhanced image saved:"
    )

    print(
        enhanced_path
    )


    # --------------------------------------
    # STEP 6 - QUALITY RECHECK
    # --------------------------------------

    print()
    print(
        "[4] Re-checking enhanced image..."
    )

    final_result = analyze_image_data(
        enhanced
    )

    print_quality_result(
        final_result,
        "POST-ENHANCEMENT QUALITY"
    )


    # --------------------------------------
    # STEP 7 - FINAL DECISION
    # --------------------------------------

    if final_result["status"] == "GOOD":

        print()
        print(
            "🟢 Enhanced image passed "
            "quality verification."
        )

        print(
            "Action: PROCEED TO DR ANALYSIS."
        )

        action = (
            "ENHANCE_AND_PROCEED"
        )

    else:

        print()
        print(
            "🔴 Enhanced image did not "
            "meet the quality requirement."
        )

        print(
            "Action: RECAPTURE IMAGE."
        )

        action = "RECAPTURE"


    # --------------------------------------
    # STEP 8 - SAVE REPORT
    # --------------------------------------

    report_path = save_pipeline_report(
        image_path,
        initial_result,
        final_result,
        action,
        enhanced_path
    )

    print()
    print(
        f"Pipeline report saved:\n"
        f"{report_path}"
    )


    # --------------------------------------
    # STEP 9 - VISUAL COMPARISON
    # --------------------------------------

    show_comparison(
        original,
        enhanced
    )


# ==========================================
# 10. MAIN PIPELINE
# ==========================================

def main():

    print()
    print("╔══════════════════════════════════════════╗")
    print("║      VISIONMITRA - QUALITY PIPELINE     ║")
    print("╚══════════════════════════════════════════╝")

    images = find_test_images()

    if not images:

        print()
        print(
            "❌ No test images found."
        )

        print()
        print(
            "Place fundus images inside:"
        )

        print(
            TEST_DIR
        )

        return

    print()
    print(
        f"Found {len(images)} "
        f"test image(s)."
    )

    for image_path in images:

        try:

            process_image(
                image_path
            )

        except Exception as error:

            print()
            print(
                f"❌ Pipeline failed for:"
            )

            print(
                image_path.name
            )

            print(
                f"Reason: {error}"
            )

    print()
    print("=" * 60)
    print(
        "          QUALITY PIPELINE FINISHED"
    )
    print("=" * 60)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()