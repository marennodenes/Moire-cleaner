import sys
from pathlib import Path

from moire.image_io import load_image, save_image
from moire.pipeline import (
    moire_filter,
    moire_filter_and_screen_clean,
    process_image,
)
from moire.screen_clean import clean_screen_moire
from moire.metrics import evaluate_image
from moire.visualize import show_comparison


BASE_DIR = Path(__file__).resolve().parent

# Select the input image here. Paths relative to the project root are supported.
INPUT_PATH = BASE_DIR / "data" / "real" / "github_screen.jpg"

# Optional processing mode flags:
#   Default: notch filter followed by screen cleaning
#   --notch-only: use only adaptive notch filtering
#   --screen-clean: use only the screen-clean method
USE_SCREEN_CLEAN = "--screen-clean" in sys.argv
USE_NOTCH_ONLY = "--notch-only" in sys.argv
OUTPUT_PATH = BASE_DIR / "results" / "processed_image.png"
COMPARISON_PATH = BASE_DIR / "results" / "comparison.png"


def main():
    # 1. Load image
    image = load_image(INPUT_PATH)
    print("Image loaded:", image.shape)

    # 2. Process image
    if USE_SCREEN_CLEAN:
        processed = process_image(image, lambda img: clean_screen_moire(img, mask=True))
    elif USE_NOTCH_ONLY:
        processed = process_image(image, moire_filter)
    else:
        processed = process_image(image, moire_filter_and_screen_clean)
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
