import sys
from pathlib import Path

from moire.image_io import load_image, save_image
from moire.pipeline import moire_filter, process_image
from moire.screen_clean import clean_screen_moire
from moire.metrics import evaluate_image
from moire.visualize import show_comparison


BASE_DIR = Path(__file__).resolve().parent

# Usage: python main.py [image path] [--screen-clean]
#   --screen-clean: use moire.screen_clean instead of the notch pipeline
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
USE_SCREEN_CLEAN = "--screen-clean" in sys.argv
INPUT_PATH = Path(ARGS[0]) if ARGS else BASE_DIR / "data" / "real" / "opencv_screen.jpg"
OUTPUT_PATH = BASE_DIR / "results" / "processed_image.png"
COMPARISON_PATH = BASE_DIR / "results" / "comparison.png"


def main():
    # 1. Load image
    image = load_image(INPUT_PATH)
    print("Image loaded:", image.shape)

    # 2. Process image
    if USE_SCREEN_CLEAN:
        processed = process_image(image, lambda img: clean_screen_moire(img, mask=True))
    else:
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
        save_path=COMPARISON_PATH
    )

    print("Pipeline execution completed!")


if __name__ == "__main__":
    main()
