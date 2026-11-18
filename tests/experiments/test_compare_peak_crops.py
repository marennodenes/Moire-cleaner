
import numpy as np
import pytest

from experiments.compare_peak_crops import (
    crop_image,
    calculate_crop_metrics,
    create_comparison,
    run_crop_comparison,
    load_image,
    save_image,
)


def test_crop_coordinates():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)

    crop = crop_image(image, 2, 3, 4, 2)

    assert crop.shape == (2, 4)
    assert np.array_equal(
        crop,
        image[3:5, 2:6]
    )


def test_invalid_crop_boundary():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        crop_image(image, 8, 8, 5, 5)


def test_crop_does_not_modify_original():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    crop = crop_image(image, 0, 0, 5, 5)
    crop[:] = 255

    assert np.all(image == 0)


def test_crop_metrics():
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    best = np.full((4, 4, 3), 10, dtype=np.uint8)
    merged = np.full((4, 4, 3), 15, dtype=np.uint8)

    metrics = calculate_crop_metrics(
        original, best, merged
    )

    assert metrics["best_change"] == pytest.approx(10)
    assert metrics["merged_change"] == pytest.approx(15)
    assert metrics["between_difference"] == pytest.approx(5)


def test_comparison_image_shape():
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    result = create_comparison(
        image, image, image, scale=2
    )

    assert result.shape == (56, 48, 3)


def test_all_crops_use_same_region():
    original = np.zeros((10, 10, 3), dtype=np.uint8)
    best = original.copy()
    merged = original.copy()

    best[3:5, 2:6] = 10
    merged[3:5, 2:6] = 20

    regions = {"sample": (2, 3, 4, 2)}

    results = run_crop_comparison(
        original, best, merged, regions
    )

    metrics = results["sample"]["metrics"]

    assert metrics["best_change"] == pytest.approx(10)
    assert metrics["merged_change"] == pytest.approx(20)
    assert metrics["between_difference"] == pytest.approx(10)


def test_unicode_png_roundtrip(tmp_path):
    image = np.full((8, 8, 3), 120, dtype=np.uint8)

    path = tmp_path / "한글_crop.png"

    save_image(path, image)
    loaded = load_image(path)

    assert np.array_equal(image, loaded)
