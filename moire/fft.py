import numpy as np


def apply_fft(image):
    f = np.fft.fft2(image)
    return np.fft.fftshift(f)


def get_magnitude_spectrum(fshift):
    return 20 * np.log(np.abs(fshift) + 1)


def apply_ifft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.real(img_back)

