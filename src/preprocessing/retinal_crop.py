from pathlib import Path

import cv2
import numpy as np


def detect_retinal_fov(image):
    """
    Detect the fundus field-of-view (FOV) and return its bounding box.

    Returns:
        (x1, y1, x2, y2)
    """

    if image is None:
        raise ValueError("Image is None.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize brightness so the detector works across
    # different APTOS cameras/exposures.
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Smooth small retinal details while preserving
    # the large FOV boundary.
    blurred = cv2.GaussianBlur(
        gray,
        (21, 21),
        0
    )

    height, width = gray.shape

    # Estimate background level from the image borders.
    border_size = max(
        5,
        int(min(height, width) * 0.02)
    )

    borders = np.concatenate([
        blurred[:border_size, :].ravel(),
        blurred[-border_size:, :].ravel(),
        blurred[:, :border_size].ravel(),
        blurred[:, -border_size:].ravel()
    ])

    background_level = float(
        np.percentile(borders, 75)
    )

    # Adaptive threshold relative to the actual image.
    threshold = max(
        10,
        background_level + 8
    )

    mask = (
        blurred > threshold
    ).astype(np.uint8) * 255

    # Clean the mask.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (31, 31)
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

    # Find contours.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0, 0, width, height

    image_area = width * height

    # Select the largest contour that is reasonably large.
    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.05:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        bbox_area = w * h

        # Prefer contours that occupy a substantial portion
        # of the image and resemble the fundus FOV.
        fill_ratio = area / max(bbox_area, 1)

        candidates.append(
            (
                area,
                fill_ratio,
                x,
                y,
                w,
                h
            )
        )

    if not candidates:
        return 0, 0, width, height

    # Largest useful FOV contour.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    _, _, x, y, w, h = candidates[0]

    # --------------------------------------------------------
    # Safety margin
    # --------------------------------------------------------

    margin_x = int(w * 0.02)
    margin_y = int(h * 0.02)

    x1 = max(
        0,
        x - margin_x
    )

    y1 = max(
        0,
        y - margin_y
    )

    x2 = min(
        width,
        x + w + margin_x
    )

    y2 = min(
        height,
        y + h + margin_y
    )

    return x1, y1, x2, y2


def crop_retinal_fov(image):
    """
    Crop the detected retinal FOV.
    """

    if image is None:
        raise ValueError("Image is None.")

    x1, y1, x2, y2 = detect_retinal_fov(
        image
    )

    cropped = image[
        y1:y2,
        x1:x2
    ]

    return cropped


def crop_retinal_fov_from_path(
    image_path,
    output_path
):
    """
    Load, crop and save one fundus image.
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    cropped = crop_retinal_fov(
        image
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path),
        cropped
    )

    if not success:
        raise IOError(
            f"Could not save image: {output_path}"
        )

    return cropped


if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    input_dir = (
        project_root
        / "data"
        / "APTOS-19"
        / "train_images"
    )

    output_dir = (
        project_root
        / "data"
        / "crop_test_v2"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    images = sorted(
        input_dir.glob("*.png")
    )[:10]

    print("=" * 70)
    print("VISIONMITRA - RETINAL FOV CROP TEST V2")
    print("=" * 70)

    for image_path in images:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"Could not read: "
                f"{image_path.name}"
            )
            continue

        original_h, original_w = (
            image.shape[:2]
        )

        x1, y1, x2, y2 = (
            detect_retinal_fov(image)
        )

        cropped = image[
            y1:y2,
            x1:x2
        ]

        output_path = (
            output_dir
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            cropped
        )

        print()
        print(image_path.name)
        print(
            f"Original : "
            f"{original_w}x{original_h}"
        )
        print(
            f"Detected : "
            f"{x2-x1}x{y2-y1}"
        )
        print(
            f"Removed  : "
            f"left={x1}, "
            f"right={original_w-x2}, "
            f"top={y1}, "
            f"bottom={original_h-y2}"
        )

    print()
    print(
        f"Saved test crops to:\n"
        f"{output_dir}"
    )

    print(
        "CROP TEST V2 COMPLETE"
    )