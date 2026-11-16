import numpy as np


def apply_fft(image):
    """Compute a 2D Fourier transform and center its zero-frequency component.

    Args:
        image: A 2D input image.

    Returns:
        The shifted 2D Fourier transform.
    """
    f = np.fft.fft2(image)
    return np.fft.fftshift(f)


def get_magnitude_spectrum(fshift):
    """Compute a log-scaled magnitude spectrum from a shifted Fourier transform.

    Args:
        fshift: A shifted 2D Fourier transform.

    Returns:
        The log-scaled magnitude spectrum.
    """
    return 20 * np.log(np.abs(fshift) + 1)


def apply_ifft(fshift):
    """Reconstruct the real-valued image from a shifted Fourier transform.

    Args:
        fshift: A shifted 2D Fourier transform.

    Returns:
        The real component of the inverse Fourier transform.
    """
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.real(img_back)
