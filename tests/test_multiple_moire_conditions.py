
import numpy as np
import pytest

import experiments.evaluate_multiple_moire_conditions as exp


def make_flat_image(size=64):
    return np.full(
        (size, size, 3),
        128,
        dtype=np.uint8
    )


def test_two_frequency_noise_changes_image():
    clean = make_flat_image()

    noisy = exp.add_two_frequency_noise(clean)

    assert noisy.shape == clean.shape
    assert noisy.dtype == np.uint8
    assert not np.array_equal(clean, noisy)


def test_two_frequency_noise_affects_both_axes():
    clean = make_flat_image()

    noisy = exp.add_two_frequency_noise(
        clean,
        amplitude_x=20,
        amplitude_y=20
    )

    # Horizontal and vertical variation should both exist.
    assert not np.array_equal(
        noisy[0, :, :],
        noisy[1, :, :]
    )

    assert not np.array_equal(
        noisy[:, 0, :],
        noisy[:, 1, :]
    )


def test_zero_amplitudes_preserve_image():
    clean = make_flat_image()

    noisy = exp.add_two_frequency_noise(
        clean,
        amplitude_x=0,
        amplitude_y=0
    )

    assert np.array_equal(clean, noisy)


def test_spatially_varying_noise_changes_image():
    clean = make_flat_image()

    noisy = exp.add_spatially_varying_noise(clean)

    assert noisy.shape == clean.shape
    assert not np.array_equal(clean, noisy)


def test_spatial_frequencies_differ():
    clean = make_flat_image(size=128)

    noisy = exp.add_spatially_varying_noise(
        clean,
        amplitude=24,
        left_frequency=0.125,
        right_frequency=0.0625
    )

    # Compare repeated values after different periods.
    left = noisy[0, :64, 0]
    right = noisy[0, 64:, 0]

    assert left[1] == left[9]
    assert right[1] == right[17]

    assert not np.array_equal(
        left[:16],
        right[:16]
    )


def test_five_scenarios_created():
    clean = exp.create_clean_image(size=128)

    scenarios = exp.create_scenarios(clean)

    assert len(scenarios) == 5

    assert set(scenarios.keys()) == {
        "A_single_frequency",
        "B_two_frequencies",
        "C_spatially_varying",
        "D_weak_noise",
        "E_clean_image",
    }

    for image in scenarios.values():
        assert image.shape == clean.shape


def test_clean_scenario_is_unchanged():
    clean = exp.create_clean_image(size=128)

    scenarios = exp.create_scenarios(clean)

    assert np.array_equal(
        scenarios["E_clean_image"],
        clean
    )


def test_clean_scenario_is_independent_copy():
    clean = exp.create_clean_image(size=128)

    scenarios = exp.create_scenarios(clean)

    scenarios["E_clean_image"][:] = 0

    assert np.any(clean != 0)


def test_weak_noise_has_lower_mse_than_strong():
    clean = exp.create_clean_image(size=128)

    scenarios = exp.create_scenarios(clean)

    strong_mse = exp.evaluate_results(
        clean,
        scenarios["A_single_frequency"],
        clean,
        clean
    )["noisy_mse"]

    weak_mse = exp.evaluate_results(
        clean,
        scenarios["D_weak_noise"],
        clean,
        clean
    )["noisy_mse"]

    assert weak_mse < strong_mse


def test_evaluate_one_scenario_uses_pipeline(monkeypatch):
    clean = make_flat_image(size=16)
    noisy = clean.copy()

    class FakeDetection:
        peaks = [object()]
        intensity = 0.8

    calls = []

    def fake_analyze(image, patch_size=512):
        calls.append(("detect", patch_size))
        return [FakeDetection()]

    def fake_filter(image, detections):
        calls.append(("filter", detections))

        best = image.astype(np.float64)
        merged = image.astype(np.float64)

        return best, merged

    monkeypatch.setattr(
        exp,
        "analyze_image",
        fake_analyze
    )

    monkeypatch.setattr(
        exp,
        "run_filtering",
        fake_filter
    )

    best, merged, metrics = exp.evaluate_one_scenario(
        clean,
        noisy
    )

    assert calls[0] == ("detect", 512)
    assert calls[1][0] == "filter"

    assert metrics["detection_regions"] == 1
    assert metrics["regions_with_peaks"] == 1
    assert metrics["active_regions"] == 1
    assert metrics["best_merged_difference"] == 0.0

    assert np.array_equal(best, clean)
    assert np.array_equal(merged, clean)
