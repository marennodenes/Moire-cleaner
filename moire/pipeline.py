import numpy as np

from moire.adaptive import process_overlapping_patches
from moire.detection import analyze_image


MIN_PATCH = 128  # smallest patch the detector accepts


def moire_filter(image, patch_size=512, sigma=2.0, threshold=0.2):
    """Detect moiré and remove it with adaptive notch filtering per patch.

    Takes and returns a BGR uint8 image. Patches below `threshold` are left untouched.
    """
    # Patches must fit inside the image and be a multiple of 4.
    size = min(patch_size, min(image.shape[:2])) // 4 * 4
    if size < MIN_PATCH:
        return image.copy()

    detections = analyze_image(image, patch_size=size)

    filtered = process_overlapping_patches(
        image,
        detections,
        patch_size=size,
        stride=size // 2,
        sigma=sigma,
        threshold=threshold,
    )

    return np.clip(np.round(filtered), 0, 255).astype(np.uint8)


def process_image(image, filter_function=None):
    """
    Run the image processing pipeline.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in BGR format.
    filter_function : callable, optional
        Filtering function

    Returns
    -------
    numpy.ndarray
        Processed image.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if filter_function is None:
        # Temporary identity filter
        processed = image.copy()
    else:
        processed = filter_function(image)

    if not isinstance(processed, np.ndarray):
        raise TypeError("Filter must return a numpy array.")

    if processed.shape != image.shape:
        raise ValueError("Processed image shape does not match input.")

    return processed
