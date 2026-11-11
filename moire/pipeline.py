import numpy as np


def process_image(image, filter_function=None):
    """
    Run the image processing pipeline.

    Parameters
    ----------
    image : numpy.ndarray
        Input image in BGR format.
    filter_function : callable, optional
        Filtering function provided by teammates.

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