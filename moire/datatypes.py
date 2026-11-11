"""Shared data types between the pipeline steps."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Peak:
    """One moiré peak in a patch spectrum.

    Coordinates use the fftshift convention: (0, 0) is the centre of the
    spectrum and fy is positive downwards. Only one peak of each mirror pair
    is stored; the filter must also suppress (-fy, -fx).
    """
    fy: int
    fx: int
    ratio: float  # magnitude / local median, ~1 is background


@dataclass
class PatchResult:
    """Detection result for one patch. Returned even when nothing is found."""
    patch_id: int
    peaks: list[Peak] = field(default_factory=list)
    intensity: float = 0.0  # 0 = no moiré, 1 = strong moiré
