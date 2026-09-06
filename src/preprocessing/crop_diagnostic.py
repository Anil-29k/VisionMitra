from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.retinal_crop import crop_retinal_fov


def main():

    project_root = Path(__file__).resolve().parents[2]

    input_dir = (
        project_root
        / "data"
        / "APTOS-19"
        / "train_images"
    )

    output_dir = (
        project_root
        / "data"
        / "crop_diagnostic"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    images = sorted(
        input_dir.glob("*.png")
    )[:10]

    print("=" * 70)
    print("VISIONMITRA - RETINAL CROP DIAGNOSTIC")
    print("=" * 70)

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        cropped = crop_retinal_fov(image)

        # ----------------------------------------------------
        # Find where the crop sits inside the original image
        # ----------------------------------------------------

        crop_h, crop_w = cropped.shape[:2]
        image_h, image_w = image.shape[:2]

        # Re-run the same detection logic to get the box.
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
            (11, 11),
            0
        )

        _, mask = cv2.threshold(
            blurred,
            15,
            255,
            cv2.THRESH_BINARY
        )

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

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        diagnostic = image.copy()

        if num_labels > 1:

            largest_label = (
                1
                + np.argmax(
                    stats[
                        1:,
                        cv2.CC_STAT_AREA
                    ]
                )
            )

            x = stats[
                largest_label,
                cv2.CC_STAT_LEFT
            ]

            y = stats[
                largest_label,
                cv2.CC_STAT_TOP
            ]

            w = stats[
                largest_label,
                cv2.CC_STAT_WIDTH
            ]

            h = stats[
                largest_label,
                cv2.CC_STAT_HEIGHT
            ]

            margin_x = int(w * 0.03)
            margin_y = int(h * 0.03)

            x1 = max(
                0,
                x - margin_x
            )

            y1 = max(
                0,
                y - margin_y
            )

            x2 = min(
                image_w,
                x + w + margin_x
            )

            y2 = min(
                image_h,
                y + h + margin_y
            )

            # Draw detected crop boundary.
            cv2.rectangle(
                diagnostic,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                8
            )

            print()
            print(
                f"{image_path.name}"
            )

            print(
                f"Original : "
                f"{image_w}x{image_h}"
            )

            print(
                f"Detected : "
                f"{x2-x1}x{y2-y1}"
            )

            print(
                f"Removed  : "
                f"left={x1}, "
                f"right={image_w-x2}, "
                f"top={y1}, "
                f"bottom={image_h-y2}"
            )

        # ----------------------------------------------------
        # Save diagnostic
        # ----------------------------------------------------

        output_path = (
            output_dir
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            diagnostic
        )

    print()
    print("=" * 70)
    print(
        f"Diagnostic images saved to:\n{output_dir}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()