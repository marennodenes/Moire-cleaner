import cv2
import numpy as np

from moire.patch_processing import divide_into_patches, merge_patches
from pathlib import Path

image_path = Path(__file__).resolve().parent / "test_image.png"

data = np.fromfile(str(image_path), dtype=np.uint8)
image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

patch_size = 64
stride = 32

patches = divide_into_patches(image, patch_size, stride)

uniform_result = merge_patches(
    patches, image.shape, weight_mode="uniform"
)

triangular_result = merge_patches(
    patches, image.shape, weight_mode="triangular"
)

print("Image shape:", image.shape)
print("Number of patches:", len(patches))

print("Uniform reconstruction:",
      np.allclose(image, uniform_result))

print("Triangular reconstruction:",
      np.allclose(image, triangular_result))

assert np.allclose(image, uniform_result)
assert np.allclose(image, triangular_result)

print("Real Image Reconstruction Test passed!")

# Reconstruction Error Test

uniform_error = np.max(
    np.abs(image.astype(np.float64) - uniform_result)
)

triangular_error = np.max(
    np.abs(image.astype(np.float64) - triangular_result)
)

print("\nMaximum Reconstruction Error")
print("Uniform:", uniform_error)
print("Triangular:", triangular_error)

assert uniform_error < 1e-10
assert triangular_error < 1e-10

print("Reconstruction Error Test passed!")
