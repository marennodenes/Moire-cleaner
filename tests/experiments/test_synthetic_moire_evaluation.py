
import numpy as np
import pytest

import experiments.evaluate_synthetic_moire as exp


def test_clean_image_shape():
    clean = exp.create_clean_image(size=128)

    assert clean.shape == (128, 128, 3)
    assert clean.dtype == np.uint8


def test_noise_changes_image():
    clean = exp.create_clean_image(size=128)

    noisy = exp.add_periodic_noise(
        clean,
        amplitude=25.0,
        frequency=0.125
    )

    assert noisy.shape == clean.shape
    assert not np.array_equal(clean, noisy)


def test_zero_amplitude_preserves_image():
    clean = exp.create_clean_image(size=128)

    noisy = exp.add_periodic_noise(
        clean,
        amplitude=0.0,
        frequency=0.125
    )

    assert np.array_equal(clean, noisy)


def test_periodic_noise_repeats():
    clean = np.full(
        (32, 32, 3), 128, dtype=np.uint8
    )

    noisy = exp.add_periodic_noise(
        clean,
        amplitude=20.0,
        frequency=0.125
    )

    # Frequency 0.125 means an 8-pixel period.
    assert np.array_equal(
        noisy[:, 0, :],
        noisy[:, 8, :]
    )


def test_identical_images_have_zero_mse():
    image = np.full(
        (16, 16, 3), 100, dtype=np.uint8
    )

    assert exp.calculate_mse(image, image) == 0.0
    assert exp.calculate_psnr(image, image) == float("inf")


def test_mse_and_psnr_calculation():
    clean = np.zeros((4, 4, 3), dtype=np.uint8)

    noisy = np.full(
        (4, 4, 3), 10, dtype=np.uint8
    )

    mse = exp.calculate_mse(clean, noisy)
    psnr = exp.calculate_psnr(clean, noisy)

    expected_psnr = 10 * np.log10(
        (255 ** 2) / 100
    )

    assert mse == pytest.approx(100.0)
    assert psnr == pytest.approx(expected_psnr)


def test_invalid_image_shape():
    clean = np.zeros((4, 4, 3))
    other = np.zeros((8, 8, 3))

    with pytest.raises(ValueError):
        exp.calculate_mse(clean, other)


def test_filtering_uses_same_parameters(monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    detections = []
    calls = []

    def fake_process(**kwargs):
        calls.append(kwargs.copy())
        return image.astype(np.float64)

    monkeypatch.setattr(
        exp,
        "process_overlapping_patches",
        fake_process
    )

    exp.run_filtering(image, detections)

    assert len(calls) == 2
    assert calls[0]["peak_mode"] == "best"
    assert calls[1]["peak_mode"] == "merged"

    assert calls[0]["detection_results"] is detections
    assert calls[1]["detection_results"] is detections

    for key in calls[0]:
        if key == "peak_mode":
            continue

        if key in ("image_bgr", "detection_results"):
            assert calls[0][key] is calls[1][key]
        else:
            assert calls[0][key] == calls[1][key]


def test_evaluation_metrics():
    clean = np.zeros((4, 4, 3), dtype=np.uint8)
    noisy = np.full((4, 4, 3), 20)
    best = np.full((4, 4, 3), 10)
    merged = np.full((4, 4, 3), 5)

    metrics = exp.evaluate_results(
        clean, noisy, best, merged
    )

    assert metrics["noisy_mse"] == pytest.approx(400)
    assert metrics["best_mse"] == pytest.approx(100)
    assert metrics["merged_mse"] == pytest.approx(25)

    assert metrics["merged_psnr"] > metrics["best_psnr"]
    assert metrics["best_psnr"] > metrics["noisy_psnr"]


def test_unicode_image_save(tmp_path):
    image = np.full(
        (16, 16, 3), 120, dtype=np.uint8
    )

    path = tmp_path / "합성_테스트.png"

    exp.save_image(path, image)

    assert path.exists()
    assert path.stat().st_size > 0
