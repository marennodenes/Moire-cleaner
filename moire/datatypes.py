"""Shared data types between the pipeline steps."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Peak:
    """One moiré frequency in a patch.

    fy, fx are in cycles per pixel of the ORIGINAL image (-0.5 .. 0.5),
    (0, 0) is DC and fy is positive downwards. Only one peak of each mirror
    pair is stored (fy > 0, or fy == 0 and fx > 0); the filter must also
    suppress (-fy, -fx).

    To get the FFT bin in an N x M region (after fftshift):
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
