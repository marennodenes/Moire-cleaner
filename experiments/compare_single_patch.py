from pathlib import Path

import cv2
import numpy as np

from moire.detection import analyze_image
from moire.fft import apply_fft, apply_ifft
from moire.notch import apply_notch_filter
from moire.adaptive import filter_patch_with_detection


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot load image: {path}")

    return image


def save_image(path, image):
    image_uint8 = np.clip(
        np.rint(image), 0, 255
    ).astype(np.uint8)

    success, encoded = cv2.imencode(".png", image_uint8)

    if not success:
        raise RuntimeError("Failed to encode image.")

    encoded.tofile(str(path))


def apply_full_notch(patch, peaks, sigma=2.0):
    """Apply the notch filter without adaptive blending."""
    channels = []

    for channel_idx in range(3):
        channel = patch[:, :, channel_idx].astype(np.float64)

        fft_image = apply_fft(channel)
        filtered_fft = apply_notch_filter(
            fft_image, peaks, sigma
        )
        filtered_channel = apply_ifft(filtered_fft)

        channels.append(filtered_channel)

    return np.stack(channels, axis=2)


def mean_change(original, processed):
    return float(np.mean(np.abs(
        processed.astype(np.float64)
        - original.astype(np.float64)
    )))


def main():
    image_path = Path("data/samples/test_image.png")
    output_dir = Path("experiments/results/single_patch")
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)

    print("Running detection...")
    detections = analyze_image(image, patch_size=512)

    candidates = [
        result for result in detections
        if result.peaks
    ]

    if not candidates:
        raise ValueError("No detected peaks found.")

    # Select the detection region with highest intensity.
    selected = max(
        candidates,
        key=lambda result: result.intensity
    )

    y, x, size = selected.y, selected.x, selected.size
    patch = image[y:y + size, x:x + size].copy()

    print("Selected Patch ID:", selected.patch_id)
    print("Position:", (y, x))
    print("Intensity:", selected.intensity)
    print("Number of peaks:", len(selected.peaks))

    for peak in selected.peaks:
        print(
            f"Peak: fy={peak.fy:.6f}, "
            f"fx={peak.fx:.6f}, "
            f"ratio={peak.ratio:.3f}"
        )

    # 1. Pure Gaussian Notch Filtering.
    full_notch = apply_full_notch(
        patch, selected.peaks, sigma=2.0
    )

    # 2. Current adaptive blending.
    adaptive = filter_patch_with_detection(
        patch,
        selected,
        sigma=2.0,
        threshold=0.2
    )

    # Save each result separately.
    save_image(output_dir / "01_original.png", patch)
    save_image(output_dir / "02_full_notch.png", full_notch)
    save_image(output_dir / "03_adaptive.png", adaptive)

    # Visualize amplified differences.
    for name, processed in [
        ("full_notch", full_notch),
        ("adaptive", adaptive),
    ]:
        difference = np.abs(
            patch.astype(np.float64) - processed
        )

        amplified = np.clip(
            difference * 20, 0, 255
        ).astype(np.uint8)

        save_image(
            output_dir / f"diff_{name}.png",
            amplified
        )

    print()
    print(
        "Full Notch mean change:",
        mean_change(patch, full_notch)
    )
    print(
        "Adaptive mean change:",
        mean_change(patch, adaptive)
    )
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
