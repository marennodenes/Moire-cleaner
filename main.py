import sys
from pathlib import Path

from moire.image_io import load_image, save_image
from moire.pipeline import moire_filter, process_image
from moire.metrics import evaluate_image
from moire.visualize import show_comparison


BASE_DIR = Path(__file__).resolve().parent

# Usage: python main.py [image path]
INPUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "results" / "test_image.png"
OUTPUT_PATH = BASE_DIR / "results" / "processed_image.png"
COMPARISON_PATH = BASE_DIR / "results" / "comparison.png"


def main():
    # 1. Load image
    image = load_image(INPUT_PATH)
    print("Image loaded:", image.shape)

    # 2. Process image
    processed = process_image(image, moire_filter)
    print("Image processing completed!")

    # 3. Save processed image
    save_image(OUTPUT_PATH, processed)
    print("Saved:", OUTPUT_PATH)

    # 4. Evaluate image
    metrics = evaluate_image(image, processed)

    print("\n=== Image Quality Evaluation ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 5. Visualize and save comparison
    show_comparison(
        image,
        processed,
        save_path=COMPARISON_PATH,
        show=False
    )

    print("Pipeline execution completed!")


if __name__ == "__main__":
    main()
