import numpy as np

from moire.patch_processing import (
    get_patch_positions,
    divide_into_patches,
    merge_patches,
    process_local_fft,
    decide_filter_strength,
    apply_adaptive_filter,
    create_triangular_weights,
)

# Patch Position Test
assert get_patch_positions(7, 4, 2) == [0, 2, 3]

image = np.arange(36).reshape(6, 6)

print("Original image:")
print(image)

patches = divide_into_patches(
    image,
    patch_size=4,
    stride=2
)

print("\nNumber of patches:", len(patches))

for i, (patch, y, x) in enumerate(patches):
    print(f"\nPatch {i + 1}: y={y}, x={x}")
    print(patch)

assert len(patches) == 4
assert all(patch.shape == (4, 4) for patch, _, _ in patches)

print("\nPatch Division Test passed!")


# Coverage Test
coverage = np.zeros(image.shape, dtype=np.int32)

for patch, y, x in patches:
    h, w = patch.shape
    coverage[y:y+h, x:x+w] += 1

print("\nCoverage Map:")
print(coverage)

assert np.all(coverage >= 1)

print("Coverage Test passed!")


# 7x7 Boundary Test
image = np.arange(49).reshape(7, 7)

patches = divide_into_patches(
    image,
    patch_size=4,
    stride=2
)

print("\n7x7 Boundary Test")
print("Number of patches:", len(patches))

# Coverage Test
coverage = np.zeros(image.shape, dtype=np.int32)

for patch, y, x in patches:
    coverage[y:y+4, x:x+4] += 1

print("Coverage Map:")
print(coverage)

assert len(patches) == 9
assert np.all(coverage >= 1)

print("Boundary Test passed!")


original = np.arange(36).reshape(6, 6)

patches = divide_into_patches(
    original,
    patch_size=4,
    stride=2
)

# Patch Merging Test
restored = merge_patches(
    patches,
    original.shape
)

print("\nOriginal Image:")
print(original)

print("\nRestored Image:")
print(restored)

assert np.allclose(original, restored)

print("\nPatch Merging Test passed!")

# 7x7 Patch Merging Test
original_7 = np.arange(49).reshape(7, 7)

patches_7 = divide_into_patches(
    original_7,
    patch_size=4,
    stride=2
)

restored_7 = merge_patches(
    patches_7,
    original_7.shape
)

assert np.allclose(original_7, restored_7)

print("7x7 Patch Merging Test passed!")

# Different Patch Values Test
patch_a = np.array([[10, 20],
                    [10, 20]])

patch_b = np.array([[40, 50],
                    [40, 50]])

test_patches = [
    (patch_a, 0, 0),
    (patch_b, 0, 1)
]

result = merge_patches(test_patches, (2, 3))

expected = np.array([
    [10, 30, 50],
    [10, 30, 50]
])

print("\nDifferent Patch Values Test:")
print(result)

assert np.allclose(result, expected)

print("Different Patch Values Test passed!")


# Dummy FFT used to verify the interface.
def dummy_fft(patch):
    return patch.copy()

image = np.arange(36).reshape(6, 6)

patches = divide_into_patches(
    image,
    patch_size=4,
    stride=2
)

fft_results = process_local_fft(
    patches,
    dummy_fft
)

print("\nLocal FFT Interface Test")

for i, (result, y, x) in enumerate(fft_results):
    print(f"Patch {i+1}: y={y}, x={x}, shape={result.shape}")

assert len(fft_results) == len(patches)

for original_patch, fft_item in zip(patches, fft_results):
    patch, y, x = original_patch
    result, result_y, result_x = fft_item

    assert (y, x) == (result_y, result_x)
    assert np.array_equal(patch, result)

print("Local FFT Interface Test passed!")


test_intensities = [0.0, 0.1, 0.3, 0.7, 1.0]

print("\nAdaptive Strength Test")

for intensity in test_intensities:
    strength = decide_filter_strength(
        intensity,
        threshold=0.2
    )

    print(f"Intensity: {intensity:.1f} -> Strength: {strength:.1f}")

expected = [0.0, 0.0, 0.3, 0.7, 1.0]

results = [
    decide_filter_strength(value, 0.2)
    for value in test_intensities
]

assert np.allclose(results, expected)

print("Adaptive Strength Test passed!")


# Dummy filter used to verify strength-dependent processing.
def dummy_filter(patch, strength):
    return patch.astype(np.float64) + strength * 10

image = np.arange(36).reshape(6, 6)

patches = divide_into_patches(image, patch_size=4, stride=2)

# Simulated moire intensities for each patch.
intensities = [0.0, 0.1, 0.5, 1.0]

filtered_patches = apply_adaptive_filter(
    patches,
    intensities,
    dummy_filter,
    threshold=0.2
)

print("\nAdaptive Filtering Interface Test")

for i, ((original, y, x), (filtered, fy, fx)) in enumerate(
    zip(patches, filtered_patches)
):
    difference = filtered.astype(np.float64) - original

    print(f"Patch {i+1}: intensity={intensities[i]:.1f}, "
          f"change={difference[0, 0]:.1f}")

    assert (y, x) == (fy, fx)
    assert filtered.shape == original.shape

expected_changes = [0.0, 0.0, 5.0, 10.0]

for (original, _, _), (filtered, _, _), expected in zip(
    patches, filtered_patches, expected_changes
):
    assert np.allclose(filtered - original, expected)

print("Adaptive Filtering Interface Test passed!")

# Adaptive Filtering + Merging Integration Test

original = np.arange(36).reshape(6, 6)

patches = divide_into_patches(
    original,
    patch_size=4,
    stride=2
)

intensities = [0.0, 0.1, 0.5, 1.0]

filtered_patches = apply_adaptive_filter(
    patches,
    intensities,
    dummy_filter,
    threshold=0.2
)

result = merge_patches(
    filtered_patches,
    original.shape
)

print("\nAdaptive Filtering + Merging Test")
print("Final Image:")
print(result)

assert result.shape == original.shape

assert np.isclose(result[0, 0], 0.0)
assert np.isclose(result[0, 2], 2.0)
assert np.isclose(result[2, 0], 14.5)
assert np.isclose(result[2, 2], 17.75)
assert np.isclose(result[5, 5], 45.0)

print("Adaptive Filtering + Merging Test passed!")


print("\nTriangular Weight Test")

weights = create_triangular_weights(5)

print(weights)

expected = np.array([
    [1, 2, 3, 2, 1],
    [2, 4, 6, 4, 2],
    [3, 6, 9, 6, 3],
    [2, 4, 6, 4, 2],
    [1, 2, 3, 2, 1]
], dtype=np.float64)

assert np.array_equal(weights, expected)
assert np.all(weights > 0)
assert weights.shape == (5, 5)

print("Triangular Weight Test passed!")

# Triangular Weighted Merging Test

original = np.arange(49).reshape(7, 7)

patches = divide_into_patches(
    original,
    patch_size=4,
    stride=2
)

restored = merge_patches(
    patches,
    original.shape,
    weight_mode="triangular"
)

print("\nTriangular Merging Test")
print("Restored Image:")
print(restored)

assert np.allclose(original, restored)

print("Triangular Merging Test passed!")

# Uniform vs Triangular Test

patch_a = np.full((3, 3), 10.0)
patch_b = np.full((3, 3), 30.0)

test_patches = [
    (patch_a, 0, 0),
    (patch_b, 0, 1)
]

uniform_result = merge_patches(
    test_patches,
    (3, 4),
    weight_mode="uniform"
)

triangular_result = merge_patches(
    test_patches,
    (3, 4),
    weight_mode="triangular"
)

print("\nUniform Result:")
print(uniform_result)

print("\nTriangular Result:")
print(triangular_result)

expected_uniform = np.array([
    [10, 20, 20, 30],
    [10, 20, 20, 30],
    [10, 20, 20, 30]
])

expected_triangular = np.array([
    [10, 50/3, 70/3, 30],
    [10, 50/3, 70/3, 30],
    [10, 50/3, 70/3, 30]
])

assert np.allclose(uniform_result, expected_uniform)
assert np.allclose(triangular_result, expected_triangular)

print("Weighted Comparison Test passed!")

# Adaptive Filtering + Triangular Merging Test

original = np.arange(36).reshape(6, 6)

patches = divide_into_patches(
    original,
    patch_size=4,
    stride=2
)

intensities = [0.0, 0.1, 0.5, 1.0]

filtered_patches = apply_adaptive_filter(
    patches,
    intensities,
    dummy_filter,
    threshold=0.2
)

uniform_result = merge_patches(
    filtered_patches,
    original.shape,
    weight_mode="uniform"
)

triangular_result = merge_patches(
    filtered_patches,
    original.shape,
    weight_mode="triangular"
)

print("\nAdaptive + Triangular Test")

print("Uniform Result:")
print(uniform_result)

print("\nTriangular Result:")
print(triangular_result)

assert triangular_result.shape == original.shape

assert not np.allclose(uniform_result, triangular_result)

assert np.isclose(triangular_result[2, 2], 146 / 9)

print("Adaptive + Triangular Test passed!")

# Exception Handling Test

print("\nException Handling Test")

def check_value_error(test_function):
    try:
        test_function()
    except ValueError:
        return True

    return False


# Test 1: Invalid Patch Size
assert check_value_error(
    lambda: get_patch_positions(6, 0, 2)
)

# Test 2: Invalid Stride
assert check_value_error(
    lambda: get_patch_positions(6, 4, 0)
)

# Test 3: Patch Size > Image Size
assert check_value_error(
    lambda: get_patch_positions(6, 8, 2)
)

# Test 4: Stride > Patch Size
assert check_value_error(
    lambda: get_patch_positions(6, 4, 5)
)

print("Patch Parameter Exception Test passed!")

# Adaptive Filtering Exception Test

test_image = np.arange(36).reshape(6, 6)

test_patches = divide_into_patches(
    test_image,
    patch_size=4,
    stride=2
)

# Reject mismatched patch and intensity counts.
assert check_value_error(
    lambda: apply_adaptive_filter(
        test_patches,
        [0.1, 0.5],
        dummy_filter
    )
)

assert check_value_error(
    lambda: decide_filter_strength(1.5)
)

# Reject filter outputs with an incorrect shape.
def wrong_shape_filter(patch, strength):
    return patch[:2, :2]

assert check_value_error(
    lambda: apply_adaptive_filter(
        test_patches,
        [0.5, 0.5, 0.5, 0.5],
        wrong_shape_filter
    )
)

print("Adaptive Filtering Exception Test passed!")

# Original Image Preservation Test

original = np.arange(36).reshape(6, 6)

original_backup = original.copy()

patches = divide_into_patches(
    original,
    patch_size=4,
    stride=2
)

# Simulate a filter that modifies its input in place.
def inplace_dummy_filter(patch, strength):
    patch[:] = 100
    return patch

filtered = apply_adaptive_filter(
    patches,
    [0.5, 0.5, 0.5, 0.5],
    inplace_dummy_filter
)

assert np.array_equal(original, original_backup)

for patch, y, x in patches:
    assert np.array_equal(
        patch,
        original[y:y+4, x:x+4]
    )

for patch, y, x in filtered:
    assert np.all(patch == 100)

print("Original Image Preservation Test passed!")

# Invalid Weight Mode Test

assert check_value_error(
    lambda: merge_patches(
        test_patches,
        test_image.shape,
        weight_mode="invalid"
    )
)

print("Weight Mode Exception Test passed!")

# Additional Validation Tests

print("\nAdditional Validation Tests")

# Invalid patch coordinates
invalid_patch = np.ones((4, 4))

assert check_value_error(
    lambda: merge_patches(
        [(invalid_patch, -1, 0)],
        (4, 4)
    )
)

# Filter returning None
def none_filter(patch, strength):
    return None

assert check_value_error(
    lambda: apply_adaptive_filter(
        test_patches,
        [0.5] * len(test_patches),
        none_filter
    )
)

# Filter returning NaN
def nan_filter(patch, strength):
    return np.full(patch.shape, np.nan)

assert check_value_error(
    lambda: apply_adaptive_filter(
        test_patches,
        [0.5] * len(test_patches),
        nan_filter
    )
)

# FFT input preservation
def inplace_fft_test(patch):
    patch[:] = 0
    return patch

original_patches = [
    (patch.copy(), y, x)
    for patch, y, x in test_patches
]

process_local_fft(test_patches, inplace_fft_test)

for (patch, _, _), (original, _, _) in zip(
    test_patches, original_patches
):
    assert np.array_equal(patch, original)

print("Additional Validation Tests passed!")
