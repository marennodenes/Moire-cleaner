# Proposal: detection → filter interface

**Status:** proposal, not agreed yet. Replaces `analyze_patch(patch, patch_id)`.

## Summary

Detection takes the **whole colour image** and returns **one result per patch**:
where the patch is, which moiré frequencies it has, and how strong the moiré is.

It needs the whole colour image because tests on real photos showed:

- **Colour** separates screen moiré (green/magenta stripes) from natural
  stripes like wood grain and leaves, and lets moiré be found behind black text.
- **The area around a patch** is needed to find coarse moiré, which has too
  few stripes inside one patch.

## API

```python
# moire/detection.py
def analyze_image(img_bgr, patch_size=512) -> list[PatchResult]:
    """img_bgr: colour image from cv2.imread (H x W x 3, uint8).
    patch_size: patch side in original image pixels.
    Returns one PatchResult per patch, row by row, including patches
    without moiré."""
```

```python
# moire/datatypes.py
@dataclass(frozen=True)
class Peak:
    fy: float      # cycles per pixel of the original image, -0.5..0.5
    fx: float      #   (0, 0) = DC, fy positive downwards
    ratio: float   # how much the peak stands out (~1 = background)

@dataclass
class PatchResult:
    patch_id: int       # row * n_cols + col
    y: int              # top-left corner and size of the patch,
    x: int              #   in original image pixels
    size: int
    peaks: list[Peak]   # one per mirror pair, strongest first; [] if none
    intensity: float    # 0 = no moiré, 1 = strong moiré
```

## Using it in the filter

```python
for res in analyze_image(img):
    region = img[res.y:res.y + res.size, res.x:res.x + res.size]
    N, M = region.shape[:2]
    for p in res.peaks:
        # FFT bin after fftshift, centre at (N//2, M//2)
        row, col = N // 2 + round(p.fy * N), M // 2 + round(p.fx * M)
        # the mirror peak must be suppressed too
        row2, col2 = N // 2 - round(p.fy * N), M // 2 - round(p.fx * M)
```

Frequencies are in cycles per pixel, not FFT bins, so they work for any
region size the filter chooses.

## To decide together

1. **Find more, or harm less?** Measured on 4 screen photos and 2 photos
   without screens (wood, plants):

   | Variant | Moiré found | False hits |
   |---|---|---|
   | A: brightness only | ~55 % | ~2 % |
   | B: also colour channels | ~80 % | ~8 % |

   A misses most moiré behind text. B also filters ~1 in 12 patches of
   coloured texture that is not moiré. Which is worse for the filter?
   (Numbers come from the photos the settings were tuned on, and will be
   checked on new photos.)
2. **Patch size:** 512 original pixels was tested. Does it suit the filter?
3. **Edges:** patches that do not fit at the right/bottom edge are dropped
   now. Shift the last row/column inwards instead?
4. **`patches.py`:** with this proposal detection splits the image itself,
   and the filter only reads `y`, `x`, `size`. Is that OK?
5. **Intensity threshold:** is 0.2 still sensible? Will be checked on real
   photos once the new detector is built.

## Known limits

- Moiré finer than a period of about 8 pixels is not reported (detection
  runs on the image scaled to 1/4).
- Large, coarse swirls are still often missed.
- Bigger windows can mark a patch next to strong moiré (e.g. the dark
  screen edge) as moiré too.
