
import csv

import numpy as np

from moire.datatypes import Peak, PatchResult

from experiments.analyze_peak_mapping import (
    count_new_frequencies,
    analyze_peak_mapping,
    summarize_mapping,
    save_csv,
)


def make_image():
    return np.zeros((4, 8, 3), dtype=np.uint8)


def make_detections(intensity=0.8):
    return [
        PatchResult(
            patch_id=0,
            y=0,
            x=0,
            size=4,
            peaks=[Peak(0.25, 0.0, 5.0)],
            intensity=intensity
        ),
        PatchResult(
            patch_id=1,
            y=0,
            x=4,
            size=4,
            peaks=[Peak(0.0, 0.25, 8.0)],
            intensity=intensity
        ),
    ]


def test_patch_count():
    records = analyze_peak_mapping(
        make_image(),
        make_detections(),
        patch_size=4,
        stride=2
    )

    # Filtering patches start at x=0, 2, 4.
    assert len(records) == 3
    assert [r["patch_x"] for r in records] == [
        0, 2, 4
    ]


def test_best_and_merged_peak_counts():
    records = analyze_peak_mapping(
        make_image(),
        make_detections(),
        patch_size=4,
        stride=2
    )

    center_patch = records[1]

    assert center_patch["patch_x"] == 2
    assert center_patch["overlap_regions"] == 2

    assert center_patch["best_peak_count"] == 1
    assert center_patch["merged_peak_count"] == 2
    assert center_patch["new_frequency_count"] == 1


def test_symmetric_peak_is_not_new():
    best = [Peak(0.125, 0.02, 5.0)]
    merged = [Peak(-0.125, -0.02, 8.0)]

    count = count_new_frequencies(
        best, merged, patch_size=512
    )

    assert count == 0


def test_different_frequency_is_new():
    best = [Peak(0.25, 0.0, 5.0)]
    merged = [
        Peak(0.25, 0.0, 5.0),
        Peak(0.0, 0.25, 8.0)
    ]

    count = count_new_frequencies(
        best, merged, patch_size=4
    )

    assert count == 1


def test_low_intensity_disables_filtering():
    records = analyze_peak_mapping(
        make_image(),
        make_detections(intensity=0.1),
        patch_size=4,
        stride=2,
        threshold=0.2
    )

    assert all(
        record["best_active"] == 0
        for record in records
    )

    assert all(
        record["merged_active"] == 0
        for record in records
    )


def test_empty_detection_results():
    records = analyze_peak_mapping(
        make_image(),
        [],
        patch_size=4,
        stride=2
    )

    assert len(records) == 3
    assert all(
        r["best_peak_count"] == 0
        for r in records
    )
    assert all(
        r["merged_peak_count"] == 0
        for r in records
    )


def test_summary_counts():
    records = analyze_peak_mapping(
        make_image(),
        make_detections(),
        patch_size=4,
        stride=2
    )

    summary = summarize_mapping(records)

    assert summary["total_patches"] == 3
    assert summary["best_active_patches"] == 3
    assert summary["merged_active_patches"] == 3

    assert summary["total_best_peaks"] == 3
    assert summary["total_merged_peaks"] == 4

    assert summary["total_new_frequencies"] == 1
    assert summary["patches_with_new_frequencies"] == 1


def test_input_detection_unchanged():
    detections = make_detections()
    original = [list(d.peaks) for d in detections]

    analyze_peak_mapping(
        make_image(),
        detections,
        patch_size=4,
        stride=2
    )

    for detection, expected in zip(
        detections, original
    ):
        assert detection.peaks == expected


def test_save_csv(tmp_path):
    records = analyze_peak_mapping(
        make_image(),
        make_detections(),
        patch_size=4,
        stride=2
    )

    output = tmp_path / "mapping.csv"
    save_csv(output, records)

    assert output.exists()

    with open(
        output, encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 3
    assert rows[1]["patch_x"] == "2"
    assert rows[1]["new_frequency_count"] == "1"
