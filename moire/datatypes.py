"""Shared data types between the pipeline steps."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Peak:
    """One moiré frequency, in cycles per pixel of the original image.

    One peak per mirror pair: the filter must also suppress (-fy, -fx).
    FFT bin in an N x M region (after fftshift):
        row = N // 2 + round(fy * N),  col = M // 2 + round(fx * M)
    """
    fy: float
    fx: float
    ratio: float  # how much the peak stands out (magnitude / local median)


@dataclass
class PatchResult:
    """Detection result for one patch. Returned even when nothing is found."""
    patch_id: int   # row * n_cols + col
    y: int          # top-left corner and size of the patch,
    x: int          #   in original image pixels
    size: int
    peaks: list[Peak] = field(default_factory=list)   # strongest first
    intensity: float = 0.0  # 0 = no moiré, 1 = strong moiré
