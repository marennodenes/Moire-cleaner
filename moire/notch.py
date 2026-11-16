import numpy as np


def peak_to_fft_bins(shape, peak):
    """Convert normalized peak frequencies to integer FFT-bin offsets.

    Args:
        shape: The FFT array shape as (rows, columns).
        peak: A detected peak with normalized `fy` and `fx` frequencies.

    Returns:
        The (row, column) frequency-bin offsets from zero frequency.
    """
    rows, cols = shape

    fy = int(round(peak.fy * rows))
    fx = int(round(peak.fx * cols))

    return fy, fx


def peak_to_coordinates(shape, peak):
    """Map a peak and its negative-frequency partner into a shifted FFT.

    The zero-frequency component is at the center of the shifted FFT, so the
    positive and negative bin offsets are added to and subtracted from center.

    Args:
        shape: The shifted FFT shape as (rows, columns).
        peak: A detected peak with normalized `fy` and `fx` frequencies.

    Returns:
        Two (row, column) coordinates for the symmetric frequency pair.
    """
    rows, cols = shape

    cy = rows // 2
    cx = cols // 2

    fy, fx = peak_to_fft_bins(shape, peak)

    peak1 = (cy + fy, cx + fx)
    peak2 = (cy - fy, cx - fx)

    return peak1, peak2


def gaussian_notch(shape, center, sigma):
    """Create a Gaussian attenuation mask centered on one frequency bin.

    The mask is zero at `center` and approaches one farther from it.

    Args:
        shape: The mask shape as (rows, columns).
        center: The notch center as a (row, column) coordinate.
        sigma: The Gaussian width in frequency bins.

    Returns:
        A 2D mask that attenuates frequencies near `center`.
    """
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
    """Apply Gaussian notches to detected peaks in a shifted Fourier transform.

    Each peak is filtered together with its symmetric negative-frequency
    partner. Peaks too close to the DC component are ignored to preserve very
    low spatial frequencies.

    Args:
        fft_image: A shifted 2D Fourier transform.
        peaks: Detected peaks with normalized `fy` and `fx` frequencies.
        sigma: The Gaussian notch width in frequency bins.
        min_distance: Minimum bin distance from DC required to filter a peak.

    Returns:
        The shifted Fourier transform multiplied by the combined notch mask.
    """
    rows, cols = fft_image.shape

    H = np.ones((rows, cols), dtype=np.float32)

    for peak in peaks:
        fy, fx = peak_to_fft_bins((rows, cols), peak)

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
