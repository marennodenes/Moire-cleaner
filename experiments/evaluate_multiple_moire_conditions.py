
from pathlib import Path

import numpy as np

from moire.detection import analyze_image

from experiments.evaluate_synthetic_moire import (
    create_clean_image,
    add_periodic_noise,
    run_filtering,
    evaluate_results,
    save_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "multiple_conditions"
)


def add_two_frequency_noise(
    clean,
    amplitude_x=18.0,
    amplitude_y=18.0,
    frequency_x=0.125,
    frequency_y=0.09375
):
    """Add two chromatic periodic waves."""
    height, width = clean.shape[:2]

    y, x = np.indices((height, width))

    wave_x = amplitude_x * np.sin(
        2 * np.pi * frequency_x * x
    )

    wave_y = amplitude_y * np.sin(
        2 * np.pi * frequency_y * y
    )

    wave = wave_x + wave_y

    noisy = clean.astype(np.float64).copy()

    noisy[:, :, 0] += wave
    noisy[:, :, 2] -= wave

    return np.clip(
        np.rint(noisy), 0, 255
    ).astype(np.uint8)


def add_spatially_varying_noise(
    clean,
    amplitude=25.0,
    left_frequency=0.125,
    right_frequency=0.0625
):
    """Apply different frequencies on the left and right."""
    height, width = clean.shape[:2]

    x = np.arange(width, dtype=np.float64)

    left_wave = amplitude * np.sin(
        2 * np.pi * left_frequency * x
    )

    right_wave = amplitude * np.sin(
        2 * np.pi * right_frequency * x
    )

    wave = np.where(
        x < width // 2,
        left_wave,
        right_wave
    )

    noisy = clean.astype(np.float64).copy()

    noisy[:, :, 0] += wave[None, :]
    noisy[:, :, 2] -= wave[None, :]

    return np.clip(
        np.rint(noisy), 0, 255
    ).astype(np.uint8)


def create_scenarios(clean):
    """Generate five synthetic test conditions."""
    return {
        "A_single_frequency": add_periodic_noise(
            clean,
            amplitude=25.0,
            frequency=0.125
        ),

        "B_two_frequencies": add_two_frequency_noise(
            clean
        ),

        "C_spatially_varying": add_spatially_varying_noise(
            clean
        ),

        "D_weak_noise": add_periodic_noise(
            clean,
            amplitude=5.0,
            frequency=0.125
        ),

        "E_clean_image": clean.copy(),
    }


def evaluate_one_scenario(clean, noisy):
    """Run detection, filtering, and Ground Truth evaluation."""
    detections = analyze_image(
        noisy,
        patch_size=512
    )

    regions_with_peaks = sum(
        bool(result.peaks)
        for result in detections
    )

    active_regions = sum(
        bool(result.peaks) and result.intensity >= 0.2
        for result in detections
    )

    best, merged = run_filtering(
        noisy,
        detections
    )

    metrics = evaluate_results(
        clean,
        noisy,
        best,
        merged
    )

    metrics["detection_regions"] = len(detections)
    metrics["regions_with_peaks"] = regions_with_peaks
    metrics["active_regions"] = active_regions

    metrics["best_merged_difference"] = float(
        np.mean(
            np.abs(
                best.astype(np.float64)
                - merged.astype(np.float64)
            )
        )
    )

    return best, merged, metrics


def run_all_scenarios(clean, scenarios):
    """Evaluate all provided scenarios."""
    results = {}

    for name, noisy in scenarios.items():
        best, merged, metrics = evaluate_one_scenario(
            clean, noisy
        )

        results[name] = {
            "noisy": noisy,
            "best": best,
            "merged": merged,
            "metrics": metrics
        }

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = create_clean_image(size=1024)

    scenarios = create_scenarios(clean)

    save_image(
        OUTPUT_DIR / "ground_truth.png",
        clean
    )

    print("Ground Truth shape:", clean.shape)
    print("Number of scenarios:", len(scenarios))

    results = {}

    for name, noisy in scenarios.items():
        print(f"\n=== Running {name} ===")

        best, merged, metrics = evaluate_one_scenario(
            clean, noisy
        )

        scenario_dir = OUTPUT_DIR / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        save_image(scenario_dir / "noisy.png", noisy)
        save_image(scenario_dir / "best.png", best)
        save_image(scenario_dir / "merged.png", merged)

        results[name] = metrics

        for key, value in metrics.items():
            if isinstance(value, int):
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value:.6f}")

    print("\n=== Summary ===")

    print(
        "Scenario | Noisy PSNR | Best PSNR | "
        "Merged PSNR | Difference (dB)"
    )

    for name, metrics in results.items():
        best_psnr = metrics["best_psnr"]
        merged_psnr = metrics["merged_psnr"]

        if np.isinf(best_psnr) and np.isinf(merged_psnr):
            difference = 0.0
        else:
            difference = merged_psnr - best_psnr

        print(
            f"{name} | "
            f"{metrics['noisy_psnr']:.4f} | "
            f"{best_psnr:.4f} | "
            f"{merged_psnr:.4f} | "
            f"{difference:+.4f}"
        )

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
