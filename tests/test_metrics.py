import numpy as np
from moire.metrics import evaluate_image

# Create a clean test image
clean = np.full((100, 100, 3), 128, dtype=np.uint8)

# Create an image with modified pixel values
processed = clean.copy()
processed[20:40, 20:40] = 150

# Evaluate image quality
results = evaluate_image(clean, processed)

for name, value in results.items():
    print(f"{name}: {value:.4f}")