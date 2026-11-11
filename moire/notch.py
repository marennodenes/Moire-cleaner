import numpy as np

def peak_to_coordinates(shape, peak):
    rows, cols = shape
    cy = rows // 2
    cx = cols // 2
    py = cy + peak.fy
    px = cx + peak.fx
    mirror_y = cy - peak.fy
    mirror_x = cx - peak.fx

    return (py, px), (mirror_y, mirror_x)

def gaussian_notch(shape, center, sigma):
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]

    py, px = center

    distance_squared = (
        (x - px) ** 2 + 
        (y - py) ** 2
        )
    return 1 - np.exp(-distance_squared / (2 * sigma**2))

def apply_notch_filter(fft_image, peaks, sigma):
    rows, cols = fft_image.shape
    H = np.ones((rows, cols), dtype=np.float32)
    for peak in peaks:
        peak1, peak2 = peak_to_coordinates((rows, cols), peak)
        H *= gaussian_notch(fft_image.shape, peak1, sigma)
        H *= gaussian_notch(fft_image.shape, peak2, sigma)
    return fft_image * H