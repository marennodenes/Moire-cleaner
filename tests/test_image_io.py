import numpy as np
from pathlib import Path
from moire.image_io import load_image, save_image

# Create a test image
original = np.full((100, 100, 3), 128, dtype=np.uint8)

# Save the image
output_path = Path("results") / "test_image.png"
save_image(output_path, original)

# Load the image
loaded = load_image(output_path)

# Compare the original and loaded images
assert np.array_equal(original, loaded)

print("Image I/O test completed!")
print("Image shape:", loaded.shape)
print("Saved to:", output_path)