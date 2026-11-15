import numpy as np


def peak_to_fft_bins(shape, peak):
    rows, cols = shape

    fy = int(round(peak.fy * rows))
    fx = int(round(peak.fx * cols))

    return fy, fx

def peak_to_coordinates(shape, peak):
    rows, cols = shape

    cy = rows // 2
    cx = cols // 2

    fy, fx = peak_to_fft_bins(shape, peak)

    peak1 = (cy + fy, cx + fx)
    peak2 = (cy - fy, cx - fx)

    return peak1, peak2


def gaussian_notch(shape, center, sigma):
    rows, cols = shape

    y, x = np.ogrid[:rows, :cols]

    py, px = center

    distance_squared = (
        (x - px) ** 2 +
        (y - py) ** 2
    )

    return 1 - np.exp(
        -distance_squared / (2 * sigma**2)
    )



def apply_notch_filter(fft_image, peaks, sigma, min_distance=16):
    """
    Apply Gaussian notch filtering to detected frequency peaks.

    Peaks too close to the DC component are ignored to avoid
    suppressing very low spatial frequencies.
    """
    rows, cols = fft_image.shape

    H = np.ones((rows, cols), dtype=np.float32)

    for peak in peaks:
        fy, fx = peak_to_fft_bins((rows, cols), peak)

        # Ignore peaks too close to the DC component.
        distance_from_dc = np.hypot(fy, fx)

        if distance_from_dc < min_distance:
            continue

        peak1, peak2 = peak_to_coordinates(
            (rows, cols),
            peak
        )

        H *= gaussian_notch(
            (rows, cols),
            peak1,
            sigma
        )

        H *= gaussian_notch(
            (rows, cols),
            peak2,
            sigma
        )

    return fft_image * H
