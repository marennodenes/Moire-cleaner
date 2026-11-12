# Detection → filter interface

**Status:** implemented in `moire/detection.py`

## Overview

`analyze_image` takes the whole colour photo and returns one result per
patch: where the patch is, which moiré frequencies it has, and how strong
the moiré is. The filter does not need to split the image or run its own
detection.

## API

```python
from moire.detection import analyze_image

results = analyze_image(img_bgr, patch_size=512)   # list[PatchResult]
```

| Argument | Meaning |
|---|---|
| `img_bgr` | colour image as loaded by `cv2.imread` (H × W × 3) |
| `patch_size` | patch side in original image pixels; multiple of 4, at least 128 |

```python
# moire/datatypes.py
@dataclass(frozen=True)
class Peak:
    fy: float      # frequency in cycles per pixel of the original image
    fx: float      #   (0, 0) = DC, fy positive downwards
    ratio: float   # how much the peak stands out (~1 = background)

@dataclass
class PatchResult:
    patch_id: int       # row * n_cols + col
    y: int              # top-left corner of the patch, original pixels
    x: int
    size: int           # = patch_size
    peaks: list[Peak]   # strongest first; [] if no moiré
    intensity: float    # 0 = no moiré, 1 = strong moiré
```

## What the filter can rely on

- **Every patch gets a result**, also without moiré (`peaks == []`,
  `intensity == 0.0`). Results come row by row.
- **Patches that do not fit** at the right or bottom edge are left out.
- **Frequencies are in cycles per pixel**, not FFT bins, so they work for
  any region size. They are at most 0.125 in each direction.
- **One peak per mirror pair** (`fy > 0`, or `fy == 0` and `fx > 0`).
  The filter must suppress both `(fy, fx)` and `(-fy, -fx)`.
- **Runtime:** about 0.3–0.6 s for a 24-megapixel phone photo.

## Example: from peak to FFT bin

```python
for res in analyze_image(img):
    region = img[res.y:res.y + res.size, res.x:res.x + res.size]
    N, M = region.shape[:2]
    F = np.fft.fftshift(np.fft.fft2(region_channel))   # centre at (N//2, M//2)
    for p in res.peaks:
        dy, dx = round(p.fy * N), round(p.fx * M)
        peak   = (N // 2 + dy, M // 2 + dx)
        mirror = (N // 2 - dy, M // 2 - dx)
        # suppress F around both peak and mirror, more when res.intensity is high
```

## Results on real photos

Measured with `experiments/evaluate_real.py` (512-pixel patches):

| Photos | Moiré found | False hits |
|---|---|---|
| 4 screens + wood + plants (settings tuned on these) | 80 % | 4 % |
| 5 new photos (only 1 screen with clear moiré) | 72 % | 8 % |

Most false hits on the new photos are rows of light text on a dark screen.

## Known limits

- **Only coloured stripes are reported.** Screen moiré is coloured;
  brightness-only stripes (wood grain, fabric) are ignored on purpose.
- **Light text on a dark screen** can be reported as moiré. Suppressing it
  may blur the text, so a gentle filter at low intensity is safer.
- **Large, coarse swirls** are still often missed.
- **Very fine moiré** (period under about 8 pixels) is not reported,
  because detection runs on the image scaled to 1/4.
- **Spill-over:** a patch next to strong moiré (e.g. the dark screen edge)
  can be marked too, because detection also looks at the area around it.

## Open questions

1. **Patch size:** is 512 pixels right for the filter?
2. **Edges:** keep dropping the patches that do not fit, or shift the last
   row/column inwards so the whole image is covered?
3. **`patches.py`:** detection splits the image itself now. Is a separate
   patch module still needed?
4. **Intensity:** it is the strongest peak ratio on a log scale
   (3.5 → 0, 50 → 1). On the 4 screen photos, patches with moiré get
   0.01–0.81, and 4–46 % of them (about 30 % overall) are below 0.2.
   Lower the threshold, or change the scale?
