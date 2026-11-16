from pathlib import Path
import csv

from moire.image_io import load_image
from moire.metrics import evaluate_image


def compare_images(reference_path, image_paths, output_csv):
    reference = load_image(reference_path)
    results = []

    for method, image_path in image_paths.items():
        image = load_image(image_path)
        metrics = evaluate_image(reference, image)

        result = {"method": method, **metrics}
        results.append(result)

        print(f"\n[{method}]")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "MSE", "PSNR", "SSIM"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate moire removal performance."
    )
    parser.add_argument(
        "--mode",
        choices=["global", "local"],
        required=True,
        help="Synthetic moire type to evaluate."
    )
    parser.add_argument(
        "--processed",
        default="results/processed_image.png",
        help="Path to the filtered image."
    )

    args = parser.parse_args()

    compare_images(
        reference_path="data/samples/clean.png",
        image_paths={
            "Before_Filtering": f"data/samples/{args.mode}_moire.png",
            "After_Filtering": args.processed
        },
        output_csv=f"results/{args.mode}_filter_comparison.csv"
    )