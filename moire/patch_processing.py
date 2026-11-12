import numpy as np

def get_patch_positions(length, patch_size, stride):
    if patch_size <= 0 or stride <= 0:
        raise ValueError("Patch size and stride must be positive.")

    if patch_size > length:
        raise ValueError("Patch size cannot exceed image length.")

    if stride > patch_size:
        raise ValueError("Stride must not exceed patch size.")

    positions = list(range(0, length - patch_size + 1, stride))

    last_position = length - patch_size

    if positions[-1] != last_position:
        positions.append(last_position)

    return positions



def divide_into_patches(image, patch_size, stride):
    if image.ndim not in (2, 3):
        raise ValueError("Image must be 2D or 3D.")

    height, width = image.shape[:2]

    y_positions = get_patch_positions(height, patch_size, stride)
    x_positions = get_patch_positions(width, patch_size, stride)

    patches = []

    for y in y_positions:
        for x in x_positions:
            patch = image[
                y:y + patch_size,
                x:x + patch_size
            ].copy()

            patches.append((patch, y, x))

    return patches


def merge_patches(patches, image_shape, weight_mode="uniform"):
    if weight_mode not in ("uniform", "triangular", "hann"):
        raise ValueError("Unsupported weight mode.")

    if len(image_shape) not in (2, 3):
        raise ValueError("Image shape must be 2D or 3D.")

    height, width = image_shape[:2]

    accumulated = np.zeros(image_shape, dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    for patch, y, x in patches:
        if patch.ndim != len(image_shape):
            raise ValueError("Patch dimensions do not match image.")

        if len(image_shape) == 3 and patch.shape[2] != image_shape[2]:
            raise ValueError("Patch channel count does not match image.")

        h, w = patch.shape[:2]

        # Validate patch coordinates and boundaries.
        if y < 0 or x < 0 or y + h > height or x + w > width:
            raise ValueError("Patch exceeds image boundaries.")

        if weight_mode == "uniform":
            weights = np.ones((h, w), dtype=np.float64)

        elif weight_mode == "triangular":
            if h != w:
                raise ValueError(
                    "Triangular mode requires square patches."
                )
            weights = create_triangular_weights(h)

        else:  # hann
            if h != w:
                raise ValueError(
                    "Hann mode requires square patches."
                )
            weights = create_hann_weights(h)

        if len(image_shape) == 3:
            weighted_patch = patch.astype(np.float64) * weights[:, :, None]
        else:
            weighted_patch = patch.astype(np.float64) * weights

        accumulated[y:y+h, x:x+w] += weighted_patch
        weight_sum[y:y+h, x:x+w] += weights

    if np.any(weight_sum == 0):
        raise ValueError("Some pixels are not covered by patches.")

    if len(image_shape) == 3:
        return accumulated / weight_sum[:, :, None]

    return accumulated / weight_sum


def process_local_fft(patches, fft_function):
    fft_results = []

    for patch, y, x in patches:
        # Protect the original patch from in-place modifications.
        fft_result = fft_function(patch.copy())

        fft_results.append((fft_result, y, x))

    return fft_results


def decide_filter_strength(moire_intensity, threshold=0.2):
    if not np.isfinite(moire_intensity) or not 0 <= moire_intensity <= 1:
        raise ValueError("Moire intensity must be between 0 and 1.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if moire_intensity < threshold:
        return 0.0

    return float(moire_intensity)


def apply_adaptive_filter(patches, intensities, filter_function,
                          threshold=0.2):
    if len(patches) != len(intensities):
        raise ValueError("Number of patches and intensities must match.")

    filtered_patches = []

    for (patch, y, x), intensity in zip(patches, intensities):
        strength = decide_filter_strength(intensity, threshold)

        if strength == 0.0:
            filtered_patch = patch.copy()
        else:
            filtered_patch = filter_function(patch.copy(), strength)

        # Validate the filter output.
        if not isinstance(filtered_patch, np.ndarray):
            raise ValueError("Filtered patch must be a NumPy array.")

        if filtered_patch.shape != patch.shape:
            raise ValueError("Filtered patch shape must match original.")

        if not np.all(np.isfinite(filtered_patch)):
            raise ValueError("Filtered patch contains NaN or Inf.")

        filtered_patches.append((filtered_patch, y, x))

    return filtered_patches


def create_triangular_weights(patch_size):
    if patch_size <= 0:
        raise ValueError("Patch size must be positive.")

    positions = np.arange(patch_size)

    weights_1d = np.minimum(
        positions + 1,
        patch_size - positions
    )

    weights_2d = np.outer(weights_1d, weights_1d)

    return weights_2d.astype(np.float64)

def create_hann_weights(patch_size, epsilon=0.05):
    if patch_size <= 0:
        raise ValueError("Patch size must be positive.")

    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Epsilon must be in (0, 1].")

    if patch_size == 1:
        return np.ones((1, 1), dtype=np.float64)

    window_1d = np.hanning(patch_size)

    # Ensure positive weights at image boundaries.
    window_1d = (
        epsilon + (1.0 - epsilon) * window_1d
    )

    weights_2d = np.outer(window_1d, window_1d)

    return weights_2d.astype(np.float64)
