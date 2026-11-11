import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


def evaluate_image(reference, processed):
    if reference.shape != processed.shape:
        raise ValueError("Image shapes do not match.")

    if reference.ndim not in (2, 3):
        raise ValueError("Only grayscale or color images are supported.")

    # MSE
    mse = np.mean(
        (reference.astype(np.float64)
         - processed.astype(np.float64)) ** 2
    )

    # PSNR
    psnr = peak_signal_noise_ratio(
        reference,
        processed,
        data_range=255
    )

    # SSIM
    ssim = structural_similarity(
        reference,
        processed,
        data_range=255,
        channel_axis=-1 if reference.ndim == 3 else None
    )

    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim
    }