
from pathlib import Path

import cv2
import numpy as np

from moire.detection import analyze_image
from moire.adaptive import process_overlapping_patches


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "synthetic"
)


def create_clean_image(size=1024):
    """Create a clean BGR image with smooth structures."""
    if size <= 0:
        raise ValueError("Size must be positive.")

    y, x = np.indices((size, size))

    # Smooth background with different color channels.
    background = (
        120.0
        + 25.0 * x / size
        + 20.0 * y / size
    )

    clean = np.stack([
        background,
        background + 8.0,
        background - 8.0
    ], axis=2)

    # Add a simple rectangular object.
    start = size // 4
    end = 3 * size // 4

    clean[start:end, start:end, :] += 20.0

    return np.clip(
        np.rint(clean), 0, 255
    ).astype(np.uint8)


def add_periodic_noise(
    clean,
    amplitude=25.0,
    frequency=0.125
):
    """Add synthetic chromatic sinusoidal noise."""
    if clean.ndim != 3 or clean.shape[2] != 3:
        raise ValueError("Input must be BGR.")

    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Amplitude must be nonnegative.")

    if not np.isfinite(frequency) or not 0 < frequency < 0.5:
        raise ValueError("Frequency must be in (0, 0.5).")

    height, width = clean.shape[:2]

    y, x = np.indices((height, width))

    wave = amplitude * np.sin(
        2.0 * np.pi * frequency * x
    )

    noisy = clean.astype(np.float64).copy()

    # Opposite changes in B and R create color variation.
    noisy[:, :, 0] += wave
    noisy[:, :, 2] -= wave

    return np.clip(
        np.rint(noisy), 0, 255
    ).astype(np.uint8)


def calculate_mse(reference, image):
    """Mean Squared Error between images."""
    if reference.shape != image.shape:
        raise ValueError("Image shapes must match.")

    ref = reference.astype(np.float64)
    img = image.astype(np.float64)

    return float(np.mean((ref - img) ** 2))


def calculate_psnr(reference, image):
    """Calculate PSNR for 8-bit image values."""
    mse = calculate_mse(reference, image)

    if mse == 0:
        return float("inf")

    return float(
        10.0 * np.log10((255.0 ** 2) / mse)
    )


def evaluate_results(clean, noisy, best, merged):
    """Evaluate each image against clean Ground Truth."""
    return {
        "noisy_mse": calculate_mse(clean, noisy),
        "best_mse": calculate_mse(clean, best),
        "merged_mse": calculate_mse(clean, merged),
        "noisy_psnr": calculate_psnr(clean, noisy),
        "best_psnr": calculate_psnr(clean, best),
        "merged_psnr": calculate_psnr(clean, merged),
    }


def run_filtering(noisy, detection_results):
    """Apply Best and Merged with identical settings."""
    params = dict(
        image_bgr=noisy,
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

    return best, merged


def save_image(path, image):
    """Save PNG to a path that may contain Korean characters."""
    image = np.clip(
        np.rint(image), 0, 255
    ).astype(np.uint8)

    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Image encoding failed.")

    encoded.tofile(str(path))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = create_clean_image(size=1024)

    noisy = add_periodic_noise(
        clean,
        amplitude=25.0,
        frequency=0.125
    )

    print("Image shape:", clean.shape)

    print("Running Detection...")

    detections = analyze_image(
        noisy,
        patch_size=512
    )

    detected_count = sum(
        bool(result.peaks) for result in detections
    )

    active_count = sum(
        bool(result.peaks) and result.intensity >= 0.2
        for result in detections
    )

    print("Detection regions:", len(detections))
    print("Regions with peaks:", detected_count)
    print("Regions with active filtering intensity:", active_count)

    for result in detections:
        print(
            f"Patch {result.patch_id}: "
            f"intensity={result.intensity:.4f}, "
            f"peaks={len(result.peaks)}"
        )

    print("Running Best and Merged Filtering...")

    best, merged = run_filtering(
        noisy, detections
    )

    metrics = evaluate_results(
        clean, noisy, best, merged
    )

    save_image(OUTPUT_DIR / "clean.png", clean)
    save_image(OUTPUT_DIR / "noisy.png", noisy)
    save_image(OUTPUT_DIR / "best.png", best)
    save_image(OUTPUT_DIR / "merged.png", merged)

    print("\n=== Ground Truth Evaluation ===")

    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
