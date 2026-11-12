from pathlib import Path

import numpy as np

from moire.detection import analyze_image
from moire.fft import apply_fft, apply_ifft
from moire.notch import apply_notch_filter

from experiments.compare_single_patch import (
    load_image,
    save_image,
    apply_full_notch,
    mean_change,
)


def apply_notch_preserve_dc(patch, peaks, sigma=4.0):
    channels = []

    for channel_idx in range(3):
        channel = patch[:, :, channel_idx].astype(np.float64)

        fft_image = apply_fft(channel)
        filtered_fft = apply_notch_filter(
            fft_image, peaks, sigma
        )

        # Restore the original DC component.
        h, w = channel.shape
        filtered_fft[h // 2, w // 2] = fft_image[h // 2, w // 2]

        channels.append(apply_ifft(filtered_fft))

    return np.stack(channels, axis=2)


def main():
    image = load_image(Path("data/samples/test_image.png"))
    detections = analyze_image(image, patch_size=512)

    selected = max(
        (d for d in detections if d.peaks),
        key=lambda d: d.intensity
    )

    y, x, size = selected.y, selected.x, selected.size
    patch = image[y:y + size, x:x + size].copy()

    full = apply_full_notch(
        patch, selected.peaks, sigma=4.0
    )

    protected = apply_notch_preserve_dc(
        patch, selected.peaks, sigma=4.0
    )

    output_dir = Path("experiments/results/dc_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(output_dir / "original.png", patch)
    save_image(output_dir / "full_notch.png", full)
    save_image(output_dir / "dc_preserved.png", protected)

    for name, result in [
        ("Full Notch", full),
        ("DC Preserved", protected),
    ]:
        print(f"{name}:")
        print("  Mean brightness:", result.mean())
        print("  Mean change:", mean_change(patch, result))

    print("Original mean brightness:", patch.mean())
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
