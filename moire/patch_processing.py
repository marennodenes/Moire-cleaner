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
    height, width = image.shape

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
    if weight_mode not in ("uniform", "triangular"):
        raise ValueError("Unsupported weight mode.")

    height, width = image_shape

    accumulated = np.zeros((height, width), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    for patch, y, x in patches:
        h, w = patch.shape

        # Validate patch coordinates and boundaries.
        if y < 0 or x < 0 or y + h > height or x + w > width:
            raise ValueError("Patch exceeds image boundaries.")

        if weight_mode == "uniform":
            weights = np.ones((h, w), dtype=np.float64)

        else:
            if h != w:
                raise ValueError("Triangular mode requires square patches.")

            weights = create_triangular_weights(h)

        accumulated[y:y+h, x:x+w] += (
            patch.astype(np.float64) * weights
        )

        weight_sum[y:y+h, x:x+w] += weights

    if np.any(weight_sum == 0):
        raise ValueError("Some pixels are not covered by patches.")

    merged_image = accumulated / weight_sum

    return merged_image


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
