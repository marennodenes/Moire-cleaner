import cv2
import numpy as np
import matplotlib.pyplot as plt

from moire.detection import analyze_image
from moire.notch import apply_notch_filter
from moire.fft import (
    apply_fft,
    apply_ifft,
    get_magnitude_spectrum,
)

"""A demo of the moiré detection and notch filtering.
Example of how you can use the different modules together."""

img_bgr = cv2.imread(
    "data/samples/github_screen.jpg",
    cv2.IMREAD_COLOR,
)

if img_bgr is None:
    raise FileNotFoundError(
        "Did not find github_screen.jpg"
    )

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB,
)

print("Running moiré-detection...")

results = analyze_image(
    img_bgr,
    patch_size=512,
)

detected_peaks = []

for patch in results:
    detected_peaks.extend(
        patch.peaks
    )

print(
    f"Found {len(detected_peaks)} peaks "
    f"in {len(results)} patches"
)

channels = cv2.split(img_rgb)

filtered_channels = []

spectrum_before = None
spectrum_after = None

for i, channel in enumerate(channels):

    print(
        f"Filtering channel {i + 1}/3..."
    )

    fft_result = apply_fft(channel)

    filtered_fft = apply_notch_filter(
        fft_result,
        detected_peaks,
        sigma=5,
    )

    filtered_channel = apply_ifft(
        filtered_fft
    )

    filtered_channel = np.clip(
        filtered_channel,
        0,
        255,
    ).astype(np.uint8)

    filtered_channels.append(
        filtered_channel
    )

    if i == 0:
        spectrum_before = (
            get_magnitude_spectrum(
                fft_result
            )
        )

        spectrum_after = (
            get_magnitude_spectrum(
                filtered_fft
            )
        )

filtered_img = cv2.merge(
    filtered_channels
)

#showing the results

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(
    spectrum_before,
    cmap="gray"
)
plt.title("FFT Before")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(
    spectrum_after,
    cmap="gray"
)
plt.title("FFT After")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(filtered_img)
plt.title("Filtered")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "notch_result.png",
    dpi=300,
)

plt.show()

print(
    "Lagret som notch_result.png"
)