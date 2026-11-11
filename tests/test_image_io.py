import numpy as np
from pathlib import Path
from moire.image_io import load_image, save_image

# 테스트 이미지 생성
original = np.full((100, 100, 3), 128, dtype=np.uint8)

# 이미지 저장
output_path = Path("results") / "test_image.png"
save_image(output_path, original)

# 이미지 불러오기
loaded = load_image(output_path)

# 원본과 불러온 이미지 비교
assert np.array_equal(original, loaded)

print("Image I/O test completed!")
print("Image shape:", loaded.shape)
print("Saved to:", output_path)