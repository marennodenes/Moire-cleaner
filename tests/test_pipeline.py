import numpy as np
from moire.pipeline import process_image


# Create test image
image = np.full((100, 100, 3), 128, dtype=np.uint8)

# Test 1: No filter
result = process_image(image)

assert np.array_equal(image, result)
print("Test 1 passed: Identity filter")


# Test 2: Dummy filter
def dummy_filter(img):
    return np.clip(
        img.astype(np.int16) + 20, 0, 255
    ).astype(np.uint8)


result = process_image(image, dummy_filter)

assert result.shape == image.shape
assert np.all(result == 148)

print("Test 2 passed: Dummy filter")
print("Pipeline test completed!")