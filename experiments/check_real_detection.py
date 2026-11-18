import sys
from pathlib import Path

import cv2
import numpy as np

from moire.detection import analyze_image
from moire.adaptive import find_best_detection
from moire.patch_processing import divide_into_patches


def main(image_path):
    # Load the image, including paths with Korean characters.
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to load the image.")

    print("Image shape:", image.shape)

    # Member B: detect moire frequencies.
    detections = analyze_image(image, patch_size=512)

    print("Detection regions:", len(detections))

    detected = [
        result for result in detections
        if result.peaks
    ]

    print("Regions with peaks:", len(detected))

    for result in detected[:10]:
        print(
            f"Patch {result.patch_id}: "
            f"position=({result.y}, {result.x}), "
            f"intensity={result.intensity:.3f}, "
            f"peaks={len(result.peaks)}"
        )

    # Member C: create overlapping filtering patches.
    patches = divide_into_patches(
        image, patch_size=512, stride=256
    )

    matched = 0
    active = 0

    for patch, y, x in patches:
        result = find_best_detection(
            y, x, patch.shape[0], detections
        )

        if result is not None:
            matched += 1

            if result.peaks and result.intensity >= 0.2:
                active += 1

    print("Filtering patches:", len(patches))
    print("Patches with detection region:", matched)
    print("Patches selected for filtering:", active)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: py -m experiments.check_real_detection IMAGE_PATH"
        )

    main(Path(sys.argv[1]))
