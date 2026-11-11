import numpy as np
from pathlib import Path
from moire.visualize import show_three_images

# 테스트용 이미지 생성
clean = np.full((100, 100, 3), 128, dtype=np.uint8)

# 인공적인 줄무늬 추가
noisy = clean.copy()
noisy[:, ::5] = 200

# 아직 필터가 없으므로 원본을 복사
filtered = noisy.copy()

# 결과 저장 경로
output_path = Path("results") / "test_comparison.png"

# 이미지 비교 및 저장
show_three_images(
    clean,
    noisy,
    filtered,
    save_path=output_path
)

print("Visualization test completed!")
print("Saved to:", output_path)