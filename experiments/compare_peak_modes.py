
from pathlib import Path

import cv2
import numpy as np

from moire.detection import analyze_image
from moire.adaptive import process_overlapping_patches


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "samples" / "test_image.png"
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"


def load_image(path):
    """Load a BGR image, including from Unicode paths."""
    data = np.fromfile(str(path), dtype=np.uint8)

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot decode image: {path}")

    return image


def save_image(path, image):
    """Save image safely to a Unicode path."""
    image_uint8 = np.clip(
        np.rint(image), 0, 255
    ).astype(np.uint8)

    success, encoded = cv2.imencode(
        ".png", image_uint8
    )

    if not success:
        raise ValueError("Failed to encode image.")

    encoded.tofile(str(path))


def calculate_comparison_metrics(original, best, merged):
    """Measure numerical changes between images."""
    original = original.astype(np.float64)
    best = best.astype(np.float64)
    merged = merged.astype(np.float64)

    if not (original.shape == best.shape == merged.shape):
        raise ValueError("Image shapes must match.")

    if not all(
        np.all(np.isfinite(img))
        for img in (original, best, merged)
    ):
        raise ValueError("Images must contain finite values.")

    diff_best = np.abs(best - original)
    diff_merged = np.abs(merged - original)
    diff_between = np.abs(merged - best)

    return {
        "best_mean_change": float(np.mean(diff_best)),
        "merged_mean_change": float(np.mean(diff_merged)),
        "between_mean_difference": float(np.mean(diff_between)),
        "between_max_difference": float(np.max(diff_between)),
    }


def run_comparison(image, detection_results):
    """Run both modes with otherwise identical parameters."""
    params = dict(
        image_bgr=image,
        detection_results=detection_results,
        patch_size=512,
        stride=256,
        sigma=2.0,
        threshold=0.2,
        weight_mode="triangular",
        intensity_mode="best"
    )

    best = process_overlapping_patches(
        **params,
        peak_mode="best"
    )

    merged = process_overlapping_patches(
        **params,
        peak_mode="merged"
    )

    metrics = calculate_comparison_metrics(
        image, best, merged
    )

    return best, merged, metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading image...")
    image = load_image(INPUT_PATH)

    print("Image shape:", image.shape)

    print("Running Moire Detection...")
    detection_results = analyze_image(
        image,
        patch_size=512
    )

    print("Detection regions:", len(detection_results))

    detected_count = sum(
        bool(result.peaks)
        for result in detection_results
    )

    print("Regions with peaks:", detected_count)

    print("Running Best and Merged filtering...")
    best, merged, metrics = run_comparison(
        image, detection_results
    )

    save_image(
        OUTPUT_DIR / "original.png",
        image
    )

    save_image(
        OUTPUT_DIR / "best.png",
        best
    )

    save_image(
        OUTPUT_DIR / "merged.png",
        merged
    )

    print("\n=== Comparison Results ===")

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print("\nSaved images to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
