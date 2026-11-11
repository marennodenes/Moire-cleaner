import numpy as np
from pathlib import Path
from moire.visualize import show_three_images

# Create a test image
clean = np.full((100, 100, 3), 128, dtype=np.uint8)

# Add artificial noise
noisy = clean.copy()
noisy[:, ::5] = 200

# No filter applied yet, so copy the noisy image
filtered = noisy.copy()

# Set the output path
output_path = Path("results") / "test_comparison.png"

# Compare and save the images
show_three_images(
    clean,
    noisy,
    filtered,
    save_path=output_path
)

print("Visualization test completed!")
print("Saved to:", output_path)