import argparse
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
OUTPUT_PATH = BASE_DIR / "results" / "processed_image.png"
COMPARISON_PATH = BASE_DIR / "results" / "comparison.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and remove moiré patterns from a photo of a screen."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="path to the input image (if omitted, the app window opens)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--notch-only",
        action="store_true",
        help="use only adaptive notch filtering",
    )
    mode.add_argument(
        "--screen-clean",
        action="store_true",
        help="use only the screen-clean method",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # No image given: open the app window instead of the command-line mode
    if args.image is None:
        from moire.gui import run
        run(initial_dir=BASE_DIR / "data")
        return

    # 1. Load image
    input_path = args.image
    if not input_path.is_file():
        sys.exit(f"Image not found: {input_path}")

    image = load_image(input_path)
    print("Image loaded:", image.shape)

    # 2. Process image
    if args.screen_clean:
        processed = process_image(image, lambda img: clean_screen_moire(img, mask=True))
    elif args.notch_only:
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
