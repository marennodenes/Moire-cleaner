import numpy as np
from moire.metrics import evaluate_image

# 깨끗한 테스트 이미지 생성
clean = np.full((100, 100, 3), 128, dtype=np.uint8)

# 일부 픽셀에 차이를 만든 이미지
processed = clean.copy()
processed[20:40, 20:40] = 150

# 성능 평가
results = evaluate_image(clean, processed)

for name, value in results.items():
    print(f"{name}: {value:.4f}")