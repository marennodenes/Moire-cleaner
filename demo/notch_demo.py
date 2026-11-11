import cv2
import matplotlib.pyplot as plt

from moire.notch import apply_notch_filter
from moire.fft import apply_fft, get_magnitude_spectrum, apply_ifft

img = cv2.imread(
    "data/samples/github_screen.jpg",
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    raise FileNotFoundError("Fant ikke github_screen.jpg")

fft_result = apply_fft(img)

class DummyPeak:
    def __init__(self, fy, fx):
        self.fy = fy
        self.fx = fx

peaks = [
    DummyPeak(10, 15)
]

filtered_fft = apply_notch_filter(
    fft_result,
    peaks,
    sigma=5
)

filtered_img = apply_ifft(filtered_fft)

spectrum_before = get_magnitude_spectrum(
    fft_result
)

spectrum_after = get_magnitude_spectrum(
    filtered_fft
)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")

plt.subplot(2, 2, 2)
plt.imshow(spectrum_before, cmap="gray")
plt.title("FFT Before Filtering")

plt.subplot(2, 2, 3)
plt.imshow(spectrum_after, cmap="gray")
plt.title("FFT After Filtering")

plt.subplot(2, 2, 4)
plt.imshow(filtered_img, cmap="gray")
plt.title("Filtered Image")

plt.tight_layout()
plt.show()