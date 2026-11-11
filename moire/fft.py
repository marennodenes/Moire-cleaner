import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/samples/cameraman.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Fant ikke cameraman.png")


def apply_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift

def get_magnitude_spectrum(fshift):
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def apply_ifft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.real(img_back)

fft_result = apply_fft(img)
reconstructed = apply_ifft(fft_result)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
#plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed, cmap="gray")
plt.title("Reconstructed Image")
#plt.axis("off")

plt.show()