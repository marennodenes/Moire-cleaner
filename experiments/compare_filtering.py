from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to load: {path}")

    return image


def save_image(path, image):
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError("Failed to encode image.")

    encoded.tofile(str(path))


def main():
    sample_dir = Path("data/samples")

    original = load_image(sample_dir / "test_image.png")
    filtered = load_image(sample_dir / "test_image_filtered.png")

    if original.shape != filtered.shape:
        raise ValueError("Image shapes do not match.")

    difference = cv2.absdiff(original, filtered)

    # Amplify small differences for visualization.
    amplified = np.clip(
        difference.astype(np.float32) * 20,
        0,
        255
    ).astype(np.uint8)

    save_image(
        sample_dir / "test_image_difference.png",
        amplified
    )

    changed_pixels = np.any(difference != 0, axis=2)

    print("Mean absolute change:", difference.mean())
    print("Maximum absolute change:", difference.max())
    print(
        "Changed pixel ratio:",
        changed_pixels.mean() * 100,
        "%"
    )

    print("Difference image saved.")


if __name__ == "__main__":
    main()
