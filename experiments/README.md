# Experiments

This directory contains experiment and evaluation scripts
used during the development of Moire-cleaner.

## 1. Detection and peak analysis
- analyze_peak_mapping.py: Analyze detected frequency peak mappings.
- check_real_detection.py: Inspect detection results on real images.
- evaluate_real.py: Evaluate detections on real-image datasets.

## 2. Filtering and comparison
- compare.py: Compare image quality metrics.
- compare_peak_modes.py: Compare best and merged peak modes.
- compare_peak_crops.py: Compare cropped regions of saved results.
- run_real_filtering.py: Apply detection and adaptive filtering to a real image.

## 3. Evaluation
- evaluate_multiple_moire_conditions.py: Evaluate multiple synthetic moire conditions.
- evaluate_synthetic_moire.py: Evaluate filtering with synthetic moire.

## 4. Exploration and data generation
- explore_real.py: Explore moire patterns in real images.
- explore_spectrum.py: Explore frequency spectra.
- make_synthetic.py: Generate synthetic moire images.

## Notes
- Run experiments from the repository root.
- Some scripts require input files that are not tracked in this repository.
- compare_peak_modes.py expects data/samples/test_image.png.
- compare_peak_crops.py requires previously generated images in experiments/results/.
- Tests for these scripts are in `tests/experiments/` and import modules from
  this directory.
- Generated outputs should not be committed.
- These scripts have not all been independently validated against the current datasets.

## Testing
Run the complete test suite from the repository root:

    py -m pytest tests -q
