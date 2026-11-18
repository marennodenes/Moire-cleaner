
import numpy as np
import pytest

import experiments.compare_peak_modes as experiment


def test_identical_outputs_have_zero_difference():
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    metrics = experiment.calculate_comparison_metrics(
        image, image.copy(), image.copy()
    )

    assert metrics["best_mean_change"] == 0.0
    assert metrics["merged_mean_change"] == 0.0
    assert metrics["between_mean_difference"] == 0.0
    assert metrics["between_max_difference"] == 0.0


def test_comparison_metrics():
    original = np.zeros((2, 2, 3), dtype=np.uint8)

    best = np.full(
        (2, 2, 3), 10.0, dtype=np.float64
    )

    merged = np.full(
        (2, 2, 3), 15.0, dtype=np.float64
    )

    metrics = experiment.calculate_comparison_metrics(
        original, best, merged
    )

    assert metrics["best_mean_change"] == pytest.approx(10.0)
    assert metrics["merged_mean_change"] == pytest.approx(15.0)
    assert metrics["between_mean_difference"] == pytest.approx(5.0)
    assert metrics["between_max_difference"] == pytest.approx(5.0)


def test_invalid_image_shape():
    original = np.zeros((4, 4, 3))
    best = np.zeros((2, 2, 3))

    with pytest.raises(ValueError):
        experiment.calculate_comparison_metrics(
            original, best, original
        )


def test_run_comparison_uses_same_parameters(monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    calls = []

    def fake_process(**kwargs):
        calls.append(kwargs.copy())

        if kwargs["peak_mode"] == "best":
            value = 10.0
        else:
            value = 20.0

        return np.full(
            image.shape, value, dtype=np.float64
        )

    monkeypatch.setattr(
        experiment,
        "process_overlapping_patches",
        fake_process
    )

    best, merged, metrics = experiment.run_comparison(
        image, []
    )

    assert len(calls) == 2

    assert calls[0]["peak_mode"] == "best"
    assert calls[1]["peak_mode"] == "merged"

    params_best = calls[0].copy()
    params_merged = calls[1].copy()

    params_best.pop("peak_mode")
    params_merged.pop("peak_mode")

    assert params_best.keys() == params_merged.keys()

    for key in params_best:
        if key in ("image_bgr", "detection_results"):
            assert params_best[key] is params_merged[key]
        else:
            assert params_best[key] == params_merged[key]

    assert np.all(best == 10.0)
    assert np.all(merged == 20.0)

    assert metrics["between_mean_difference"] == 10.0


def test_unicode_image_save_and_load(tmp_path):
    image = np.full(
        (8, 8, 3), 120, dtype=np.uint8
    )

    path = tmp_path / "한글_테스트.png"

    experiment.save_image(path, image)

    loaded = experiment.load_image(path)

    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)
