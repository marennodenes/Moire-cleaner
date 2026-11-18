import sys
from pathlib import Path

import cv2
import numpy as np

from moire.detection import analyze_image
from moire.adaptive import process_overlapping_patches


def main(image_path):
    # Load BGR image (supports Korean file paths).
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to load the input image.")

    print("Input shape:", image.shape)

    # Step 1: Run Member B's detection.
    print("Running detection...")
    detections = analyze_image(
        image,
        patch_size=512
    )

    print("Detection completed:", len(detections), "regions")

    # Step 2: Run Member C's overlapping filtering.
    print("Running adaptive filtering...")
    filtered = process_overlapping_patches(
        image,
        detections,
        patch_size=512,
        stride=256,
        sigma=2.0,
        threshold=0.2,
        weight_mode="triangular"
    )

    # Step 3: Convert to uint8 for saving.
    filtered_uint8 = np.clip(
        np.rint(filtered), 0, 255
    ).astype(np.uint8)

    # Step 4: Save output alongside the input image.
    output_path = (
        image_path.parent
        / f"{image_path.stem}_filtered.png"
    )

    success, encoded = cv2.imencode(
        ".png", filtered_uint8
    )

    if not success:
        raise RuntimeError("Failed to encode output image.")

    encoded.tofile(str(output_path))

    print("Output saved:", output_path)
    print("Output shape:", filtered_uint8.shape)

    # Step 5: Print a basic change metric.
    difference = np.abs(
        filtered_uint8.astype(np.float64)
        - image.astype(np.float64)
    )

    print("Mean absolute change:", difference.mean())
    print("Maximum absolute change:", difference.max())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: py -m experiments.run_real_filtering IMAGE_PATH"
        )

    main(Path(sys.argv[1]))
