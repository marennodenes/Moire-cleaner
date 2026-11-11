import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def show_comparison(original, processed, save_path=None):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(processed_rgb)
    axes[1].set_title("Processed Image")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def show_three_images(clean, noisy, filtered, save_path=None):
    images = [clean, noisy, filtered]
    titles = ["Ground Truth", "Noisy Image", "Filtered Image"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, img, title in zip(axes, images, titles):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)