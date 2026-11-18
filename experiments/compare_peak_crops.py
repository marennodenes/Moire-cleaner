
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR = INPUT_DIR / "crops"

# (x, y, width, height)
CROP_REGIONS = {
    "upper_text": (1700, 1100, 512, 512),
    "lower_left": (900, 2250, 512, 512),
    "lower_center": (1750, 2600, 512, 512),
}


def load_image(path):
    """Read an image safely from a Unicode path."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot decode image: {path}")

    return image


def save_image(path, image):
    """Save a PNG safely to a Unicode path."""
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Cannot encode image.")

    encoded.tofile(str(path))


def crop_image(image, x, y, width, height):
    """Extract a rectangular region without altering the input."""
    image_height, image_width = image.shape[:2]

    if width <= 0 or height <= 0:
        raise ValueError("Crop size must be positive.")

    if (
        x < 0 or y < 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError("Crop exceeds image boundaries.")

    return image[y:y + height, x:x + width].copy()


def calculate_crop_metrics(original, best, merged):
    """Calculate differences for saved uint8 images."""
    if original.shape != best.shape or original.shape != merged.shape:
        raise ValueError("Crop shapes must match.")

    original = original.astype(np.float64)
    best = best.astype(np.float64)
    merged = merged.astype(np.float64)

    return {
        "best_change": float(
            np.mean(np.abs(best - original))
        ),
        "merged_change": float(
            np.mean(np.abs(merged - original))
        ),
        "between_difference": float(
            np.mean(np.abs(merged - best))
        ),
    }


def create_comparison(original, best, merged, scale=2):
    """Create a side-by-side comparison with identical scaling."""
    if not (original.shape == best.shape == merged.shape):
        raise ValueError("Image shapes must match.")

    if scale < 1 or int(scale) != scale:
        raise ValueError("Scale must be a positive integer.")

    labels = ["Original", "Best", "Merged"]
    images = [original, best, merged]

    panels = []

    for label, image in zip(labels, images):
        enlarged = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST
        )

        # Add a separate title area above the crop.
        panel_height = enlarged.shape[0] + 40
        panel_width = enlarged.shape[1]

        panel = np.full(
            (panel_height, panel_width, 3),
            255,
            dtype=np.uint8
        )

        panel[40:, :] = enlarged

        cv2.putText(
            panel,
            label,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        panels.append(panel)

    return np.hstack(panels)


def run_crop_comparison(original, best, merged, regions):
    """Return cropped comparisons and per-region metrics."""
    if not (original.shape == best.shape == merged.shape):
        raise ValueError("Image shapes must match.")

    results = {}

    for name, (x, y, width, height) in regions.items():
        crop_original = crop_image(
            original, x, y, width, height
        )

        crop_best = crop_image(
            best, x, y, width, height
        )

        crop_merged = crop_image(
            merged, x, y, width, height
        )

        comparison = create_comparison(
            crop_original,
            crop_best,
            crop_merged
        )

        metrics = calculate_crop_metrics(
            crop_original,
            crop_best,
            crop_merged
        )

        results[name] = {
            "comparison": comparison,
            "metrics": metrics,
        }

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original = load_image(INPUT_DIR / "original.png")
    best = load_image(INPUT_DIR / "best.png")
    merged = load_image(INPUT_DIR / "merged.png")

    print("Image shape:", original.shape)

    results = run_crop_comparison(
        original,
        best,
        merged,
        CROP_REGIONS
    )

    for name, result in results.items():
        output_path = OUTPUT_DIR / f"{name}_comparison.png"

        save_image(
            output_path,
            result["comparison"]
        )

        print(f"\n=== {name} ===")

        for key, value in result["metrics"].items():
            print(f"{key}: {value:.6f}")

        print("Saved:", output_path)

    print("\nCrop comparison completed.")


if __name__ == "__main__":
    main()
