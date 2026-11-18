import numpy as np

from dataclasses import replace
from moire.fft import apply_fft, apply_ifft
from moire.notch import apply_notch_filter
from moire.patch_processing import decide_filter_strength, divide_into_patches, merge_patches

def find_best_detection(patch_y, patch_x, patch_size, detection_results):
    """
    Find the detection region with the largest overlap
    with a filtering patch.

    Parameters
    ----------
    patch_y, patch_x : int
        Top-left coordinates of the filtering patch.
    patch_size : int
        Size of the square filtering patch.
    detection_results : list[PatchResult]
        Detection results returned by analyze_image().

    Returns
    -------
    PatchResult or None
        Detection result with the largest overlap.
    """
    best_result = None
    max_overlap = 0

    for result in detection_results:
        y1 = max(patch_y, result.y)
        x1 = max(patch_x, result.x)

        y2 = min(
            patch_y + patch_size,
            result.y + result.size
        )
        x2 = min(
            patch_x + patch_size,
            result.x + result.size
        )

        overlap_height = max(0, y2 - y1)
        overlap_width = max(0, x2 - x1)

        overlap_area = overlap_height * overlap_width

        if overlap_area > max_overlap:
            max_overlap = overlap_area
            best_result = result

    return best_result


def filter_patch_with_detection(
    patch,
    detection_result,
    sigma=2.0,
    threshold=0.2
):
    """
    Apply adaptive Gaussian Notch Filtering to one BGR patch.
    """
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError("Patch must be a BGR image.")

    if not np.all(np.isfinite(patch)):
        raise ValueError("Patch contains NaN or Inf.")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be positive and finite.")

    original = patch.astype(np.float64)

    # No corresponding detection region.
    if detection_result is None:
        return original.copy()

    strength = decide_filter_strength(
        detection_result.intensity,
        threshold
    )

    # Skip filtering when no significant moire is detected.
    if strength == 0.0 or not detection_result.peaks:
        return original.copy()

    filtered_channels = []

    for channel_idx in range(3):
        channel = original[:, :, channel_idx]

        fft_image = apply_fft(channel)

        filtered_fft = apply_notch_filter(
            fft_image,
            detection_result.peaks,
            sigma
        )

        filtered_channel = apply_ifft(filtered_fft)

        filtered_channels.append(filtered_channel)

    filtered_bgr = np.stack(filtered_channels, axis=2)

    if not np.all(np.isfinite(filtered_bgr)):
        raise ValueError("Filtered result contains NaN or Inf.")

    # Blend original and filtered images according to intensity.
    result = (
        (1.0 - strength) * original
        + strength * filtered_bgr
    )

    return result



def process_overlapping_patches(
    image_bgr,
    detection_results,
    patch_size=512,
    stride=256,
    sigma=2.0,
    threshold=0.2,
    weight_mode="triangular",
    intensity_mode="best",
    peak_mode="best"
):
    """
    Process overlapping BGR patches using detection-guided
    adaptive notch filtering.

    intensity_mode:
        "best"     - intensity from the largest-overlap region
        "weighted" - overlap-area-weighted intensity
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input must be a BGR image.")

    if intensity_mode not in ("best", "weighted"):
        raise ValueError("Unsupported intensity mode.")

    if peak_mode not in ("best", "merged"):
        raise ValueError("Unsupported peak mode.")

    patches = divide_into_patches(
        image_bgr, patch_size, stride
)

    filtered_patches = []

    for patch, y, x in patches:
        detection = find_best_detection(
            y, x, patch_size, detection_results
        )

        if detection is not None and intensity_mode == "weighted":
            weighted_intensity = calculate_weighted_intensity(
                y, x, patch_size, detection_results
            )

            # Keep the selected peaks, but use weighted intensity.
            detection = replace(
                detection,
                intensity=weighted_intensity
            )

        if detection is not None and peak_mode == "merged":
            overlaps = get_detection_overlaps(
                y, x, patch_size, detection_results
            )

            local_detections = [
                result for result, area, ratio in overlaps
            ]

            merged_peaks = merge_local_detection_peaks(
                local_detections,
                patch_size=patch_size,
                max_distance_bins=1.0
            )

            detection = replace(
                detection,
                peaks=merged_peaks
            )

        filtered_patch = filter_patch_with_detection(
            patch,
            detection,
            sigma=sigma,
            threshold=threshold
        )

        filtered_patches.append((filtered_patch, y, x))

    return merge_patches(
        filtered_patches,
        image_bgr.shape,
        weight_mode=weight_mode
    )

def get_detection_overlaps(
    patch_y,
    patch_x,
    patch_size,
    detection_results
):
    """
    Return detection regions overlapping a filtering patch.

    Each result contains:
        (PatchResult, overlap_area, overlap_ratio)

    overlap_ratio is relative to the filtering patch area.
    """
    if patch_size <= 0:
        raise ValueError("Patch size must be positive.")

    overlaps = []
    patch_area = patch_size * patch_size

    for result in detection_results:
        y1 = max(patch_y, result.y)
        x1 = max(patch_x, result.x)

        y2 = min(
            patch_y + patch_size,
            result.y + result.size
        )
        x2 = min(
            patch_x + patch_size,
            result.x + result.size
        )

        overlap_height = max(0, y2 - y1)
        overlap_width = max(0, x2 - x1)

        overlap_area = overlap_height * overlap_width

        if overlap_area > 0:
            overlap_ratio = overlap_area / patch_area

            overlaps.append(
                (result, overlap_area, overlap_ratio)
            )

    return overlaps


def calculate_weighted_intensity(
    patch_y,
    patch_x,
    patch_size,
    detection_results
):
    """
    Calculate overlap-area-weighted moire intensity.

    Returns 0.0 if no detection region overlaps.
    """
    overlaps = get_detection_overlaps(
        patch_y,
        patch_x,
        patch_size,
        detection_results
    )

    if not overlaps:
        return 0.0

    total_area = sum(
        area for _, area, _ in overlaps
    )

    weighted_sum = sum(
        result.intensity * area
        for result, area, _ in overlaps
    )

    return float(weighted_sum / total_area)


def calculate_peak_distance(peak_a, peak_b):
    """
    Calculate frequency distance between two peaks,
    accounting for FFT conjugate symmetry.
    """
    direct_distance = (
        (peak_a.fy - peak_b.fy) ** 2
        + (peak_a.fx - peak_b.fx) ** 2
    ) ** 0.5

    symmetric_distance = (
        (peak_a.fy + peak_b.fy) ** 2
        + (peak_a.fx + peak_b.fx) ** 2
    ) ** 0.5

    return min(direct_distance, symmetric_distance)

def match_mutual_peaks(
    peaks_a,
    peaks_b,
    patch_size=512,
    max_distance_bins=1.0
):
    """
    Match mutually nearest peaks within a distance threshold.

    Returns:
        list of (index_a, index_b, distance_bins)
    """
    if patch_size <= 0:
        raise ValueError("Patch size must be positive.")

    if not np.isfinite(max_distance_bins) or max_distance_bins < 0:
        raise ValueError("Invalid peak distance threshold.")

    if not peaks_a or not peaks_b:
        return []

    distances = np.array([
        [
            calculate_peak_distance(a, b) * patch_size
            for b in peaks_b
        ]
        for a in peaks_a
    ])

    nearest_b = np.argmin(distances, axis=1)
    nearest_a = np.argmin(distances, axis=0)

    matches = []

    for i, j in enumerate(nearest_b):
        j = int(j)

        distance = float(distances[i, j])

        if (
            nearest_a[j] == i
            and distance <= max_distance_bins + 1e-9
        ):
            matches.append((i, j, distance))

    return matches

def merge_matched_peaks(
    peaks_a,
    peaks_b,
    patch_size=512,
    max_distance_bins=1.0
):
    """
    Merge mutually matched peaks by selecting the peak
    with the higher ratio.

    Unmatched peaks are preserved.
    """
    matches = match_mutual_peaks(
        peaks_a,
        peaks_b,
        patch_size=patch_size,
        max_distance_bins=max_distance_bins
    )

    matched_a = set()
    matched_b = set()
    merged_peaks = []

    # 1. Merge matched peak pairs.
    for i, j, distance in matches:
        peak_a = peaks_a[i]
        peak_b = peaks_b[j]

        if peak_a.ratio >= peak_b.ratio:
            representative = peak_a
        else:
            representative = peak_b

        merged_peaks.append(representative)

        matched_a.add(i)
        matched_b.add(j)

    # 2. Preserve unmatched peaks from A.
    for i, peak in enumerate(peaks_a):
        if i not in matched_a:
            merged_peaks.append(peak)

    # 3. Preserve unmatched peaks from B.
    for j, peak in enumerate(peaks_b):
        if j not in matched_b:
            merged_peaks.append(peak)

    return merged_peaks

def merge_local_detection_peaks(
    detection_results,
    patch_size=512,
    max_distance_bins=1.0
):
    """
    Merge peaks from multiple local detection regions.

    Only mutually matched peaks are considered.
    Groups are merged only when every cross-group
    peak pair is within the distance threshold.

    Returns a list of representative Peak objects.
    """
    if patch_size <= 0:
        raise ValueError("Patch size must be positive.")

    if not np.isfinite(max_distance_bins) or max_distance_bins < 0:
        raise ValueError("Invalid peak distance threshold.")

    # 1. Collect all peaks with their region indices.
    nodes = []
    region_nodes = []

    for region_idx, result in enumerate(detection_results):
        indices = []

        for peak in result.peaks:
            indices.append(len(nodes))
            nodes.append((region_idx, peak))

        region_nodes.append(indices)

    if not nodes:
        return []

    # 2. Find mutual matches between different regions.
    edges = []

    for a in range(len(detection_results)):
        for b in range(a + 1, len(detection_results)):
            matches = match_mutual_peaks(
                detection_results[a].peaks,
                detection_results[b].peaks,
                patch_size=patch_size,
                max_distance_bins=max_distance_bins
            )

            for i, j, distance in matches:
                node_a = region_nodes[a][i]
                node_b = region_nodes[b][j]
                edges.append((distance, node_a, node_b))

    # Process nearest matches first.
    edges.sort()

    # 3. Start with one group per peak.
    groups = [{i} for i in range(len(nodes))]

    for distance, node_a, node_b in edges:
        group_a = next(g for g in groups if node_a in g)
        group_b = next(g for g in groups if node_b in g)

        if group_a is group_b:
            continue

        # Never merge two peaks from the same region.
        regions_a = {nodes[i][0] for i in group_a}
        regions_b = {nodes[i][0] for i in group_b}

        if regions_a & regions_b:
            continue

        # Complete-link condition.
        all_close = all(
            calculate_peak_distance(
                nodes[i][1],
                nodes[j][1]
            ) * patch_size <= max_distance_bins + 1e-9
            for i in group_a
            for j in group_b
        )

        if not all_close:
            continue

        group_a.update(group_b)
        groups.remove(group_b)

    # 4. Select the highest-ratio representative.
    merged_peaks = []

    for group in sorted(groups, key=lambda g: min(g)):
        best_idx = max(
            group,
            key=lambda i: (nodes[i][1].ratio, -i)
        )
        merged_peaks.append(nodes[best_idx][1])

    return merged_peaks
