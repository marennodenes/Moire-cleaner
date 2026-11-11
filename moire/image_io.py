import cv2
from pathlib import Path


def load_image(path):
    """Load an image in BGR format."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    return image


def save_image(path, image):
    """Save an image and create directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise IOError(f"Cannot save image: {path}")

    return path