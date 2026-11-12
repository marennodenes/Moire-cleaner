
import cv2
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output paths
INPUT_PATH = BASE_DIR / "data" / "samples" / "clean.png"
OUTPUT_DIR = BASE_DIR / "data" / "samples"


def add_global_moire(image, frequency=0.08, amplitude=35):
    """Add periodic noise to the entire image."""

    height, width = image.shape[:2]
    y, x = np.indices((height, width))

    pattern = amplitude * np.sin(
        2 * np.pi * frequency * (x + y)
    )

    noisy = (
        image.astype(np.float32)
        + pattern[:, :, np.newaxis]
    )

    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_local_moire(image, frequency=0.08, amplitude=35):
    """Add periodic noise to the center region only."""

    height, width = image.shape[:2]
    y, x = np.indices((height, width))

    pattern = amplitude * np.sin(
        2 * np.pi * frequency * (x + y)
    )

    noisy = image.astype(np.float32).copy()

    # Center region: 25% to 75%
    y1, y2 = height // 4, 3 * height // 4
    x1, x2 = width // 4, 3 * width // 4

    noisy[y1:y2, x1:x2] += pattern[
        y1:y2, x1:x2, np.newaxis
    ]

    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():
    image = cv2.imread(str(INPUT_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Cannot load clean image: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global_moire = add_global_moire(image)
    local_moire = add_local_moire(image)

    cv2.imwrite(
        str(OUTPUT_DIR / "global_moire.png"),
        global_moire
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "local_moire.png"),
        local_moire
    )

    print("Synthetic moire images generated!")
    print("Global:", OUTPUT_DIR / "global_moire.png")
    print("Local:", OUTPUT_DIR / "local_moire.png")


if __name__ == "__main__":
    main()
