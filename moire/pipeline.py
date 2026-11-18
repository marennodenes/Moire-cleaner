import numpy as np

from moire.adaptive import process_overlapping_patches
from moire.detection import analyze_image
from moire.screen_clean import clean_screen_moire, moire_mask


MIN_PATCH = 128  # smallest patch the detector accepts


def _notch_filter(image, patch_size, sigma, threshold):
    """Run adaptive notch filtering and return the detections used."""
    # Patches must fit inside the image and be a multiple of 4.
    size = min(patch_size, min(image.shape[:2])) // 4 * 4
    if size < MIN_PATCH:
        return image.copy(), size, None

    detections = analyze_image(image, patch_size=size)

    filtered = process_overlapping_patches(
        image,
        detections,
        patch_size=size,
        stride=size // 2,
        sigma=sigma,
        threshold=threshold,
    )

    filtered = np.clip(np.round(filtered), 0, 255).astype(np.uint8)
    return filtered, size, detections


def moire_filter(image, patch_size=512, sigma=2.0, threshold=0.2):
    """Detect moiré and remove it with adaptive notch filtering per patch.

    Takes and returns a BGR uint8 image. Patches below `threshold` are left untouched.
    """
    filtered, _, _ = _notch_filter(image, patch_size, sigma, threshold)
    return filtered


def moire_filter_and_screen_clean(image, patch_size=512, sigma=2.0, threshold=0.2):
    """Apply adaptive notch filtering, then screen cleaning in detected areas.

    The screen-clean mask uses the original notch detections, so those areas
    remain eligible for cleaning even if notch filtering weakens their peaks.
    """
    filtered, size, detections = _notch_filter(image, patch_size, sigma, threshold)
    if (
        detections is None
        or min(image.shape[:2]) < 256
        or not any(result.peaks for result in detections)
    ):
        return filtered

    mask = moire_mask(image, patch_size=size, detections=detections)
    return clean_screen_moire(filtered, mask=mask)


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
