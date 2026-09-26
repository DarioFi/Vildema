"""Turn a bitmap into a terminal-renderable image, with integer scaling only.

A "bitmap" is whatever a visualizer hands back — a numpy array, a PIL image, or
a path to an image file. We normalize it to an ``H×W×3`` uint8 array and render
it with Unicode upper-half-block characters (``▀``): each character cell stacks
two vertical pixels (foreground = top pixel, background = bottom pixel), so one
text row shows two image rows.

Scaling is strictly integer — nearest-neighbour repeat to enlarge, block-mean
to shrink — so pixels stay crisp and aligned (no fractional resampling). numpy
and PIL are imported lazily so the rest of the TUI does not depend on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

from rich.color import Color
from rich.style import Style
from rich.text import Text

if TYPE_CHECKING:
  import numpy as np

_HALF_BLOCK = "▀"


@dataclass
class Rendered:
  """The terminal rendering of a bitmap plus a description of how it was scaled."""

  text: Text
  source_w: int
  source_h: int
  display_w: int
  display_h: int
  factor: int
  mode: str  # "actual" | "upscaled" | "downscaled"

  def describe(self) -> str:
    if self.mode == "upscaled":
      scale = f"upscaled {self.factor}×"
    elif self.mode == "downscaled":
      scale = f"downscaled 1/{self.factor}"
    else:
      scale = "1:1"
    return (
      f"original {self.source_w}×{self.source_h}px · "
      f"shown {self.display_w}×{self.display_h}px · {scale}"
    )


def to_rgb_array(bitmap: Any) -> "np.ndarray":
  """Normalize any supported bitmap to an ``H×W×3`` uint8 array.

  Accepts a path / PIL image / array-like. Arrays may be grayscale (``H×W`` or
  ``H×W×1``), RGB (``H×W×3``), RGBA (``H×W×4``), or a channel-first tensor
  layout (``1×H×W`` / ``3×H×W``). Float data is assumed to be in ``[0, 1]`` if
  its max is ``<= 1`` and ``[0, 255]`` otherwise.
  """
  import numpy as np

  if isinstance(bitmap, (str, Path)):
    from PIL import Image

    with Image.open(bitmap) as im:
      return np.asarray(im.convert("RGB"), dtype=np.uint8)

  # PIL image (duck-typed to avoid importing PIL when not needed).
  if hasattr(bitmap, "convert") and hasattr(bitmap, "size"):
    return np.asarray(bitmap.convert("RGB"), dtype=np.uint8)

  arr = np.asarray(bitmap)

  # Channel-first (C×H×W) tensor layout -> channels-last (H×W×C).
  if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
    arr = np.transpose(arr, (1, 2, 0))

  if arr.ndim == 2:
    arr = arr[:, :, None]
  if arr.ndim != 3:
    raise ValueError(f"cannot interpret array of shape {arr.shape} as an image")

  channels = arr.shape[2]
  if channels == 1:
    arr = np.repeat(arr, 3, axis=2)
  elif channels == 4:
    arr = arr[:, :, :3]
  elif channels != 3:
    raise ValueError(f"unsupported channel count: {channels}")

  if np.issubdtype(arr.dtype, np.floating):
    scale = 255.0 if float(np.nanmax(arr, initial=0.0)) <= 1.0 else 1.0
    arr = arr * scale

  return np.clip(arr, 0, 255).astype(np.uint8)


# Zoom is a signed integer "level": ``z >= 1`` upscales by ``z`` (``1`` is 1:1),
# ``z <= -1`` downscales by ``-z + 1`` (``-1`` is 1/2, ``-2`` is 1/3, …). ``0``
# is skipped so stepping moves smoothly across 1:1.


def fit_zoom(rgb: "np.ndarray", max_cols: int, max_rows: int) -> int:
  """The largest integer zoom level whose result fits the cell budget."""
  height, width = rgb.shape[:2]
  budget_w = max(1, max_cols)
  budget_h = max(1, max_rows * 2)  # two stacked pixels per text row
  if width <= budget_w and height <= budget_h:
    return min(budget_w // width, budget_h // height)  # >= 1 (upscale / 1:1)
  factor = max(math.ceil(width / budget_w), math.ceil(height / budget_h))  # >= 2
  return -(factor - 1)  # downscale


def step_zoom(zoom: int, delta: int) -> int:
  """Move ``zoom`` by ``delta`` levels, skipping the unused ``0``."""
  zoom += delta
  if zoom == 0:
    zoom += delta if delta else 1
  return zoom


def clamp_zoom(zoom: int, rgb: "np.ndarray") -> int:
  """Keep ``zoom`` within sane bounds for ``rgb`` (no >32× up, never below 1px)."""
  smallest = max(1, min(rgb.shape[0], rgb.shape[1]))
  return max(-(smallest - 1), min(32, zoom))


def _apply_zoom(rgb: "np.ndarray", zoom: int) -> Tuple["np.ndarray", int, str]:
  """Scale ``rgb`` by ``zoom``; returns (scaled, factor, mode)."""
  import numpy as np

  if zoom >= 1:
    if zoom == 1:
      return rgb, 1, "actual"
    return np.repeat(np.repeat(rgb, zoom, axis=0), zoom, axis=1), zoom, "upscaled"

  factor = -zoom + 1
  height, width = rgb.shape[:2]
  crop = rgb[: height // factor * factor, : width // factor * factor]
  reduced = crop.reshape(
    max(1, crop.shape[0] // factor), factor, max(1, crop.shape[1] // factor), factor, 3
  )
  return reduced.mean(axis=(1, 3)).astype(np.uint8), factor, "downscaled"


def _to_text(rgb: "np.ndarray") -> Text:
  import numpy as np

  width = rgb.shape[1]
  if rgb.shape[0] % 2:  # pad to an even number of rows so pairs line up
    rgb = np.vstack([rgb, np.zeros((1, width, 3), dtype=rgb.dtype)])

  text = Text(no_wrap=True, overflow="crop")
  for y in range(0, rgb.shape[0], 2):
    for x in range(width):
      top = rgb[y, x]
      bottom = rgb[y + 1, x]
      text.append(
        _HALF_BLOCK,
        Style(
          color=Color.from_rgb(int(top[0]), int(top[1]), int(top[2])),
          bgcolor=Color.from_rgb(int(bottom[0]), int(bottom[1]), int(bottom[2])),
        ),
      )
    if y + 2 < rgb.shape[0]:
      text.append("\n")
  return text


def render_source(source: "np.ndarray", zoom: int) -> Rendered:
  """Render an already-normalized RGB array at an explicit ``zoom`` level."""
  scaled, factor, mode = _apply_zoom(source, zoom)
  return Rendered(
    text=_to_text(scaled),
    source_w=int(source.shape[1]),
    source_h=int(source.shape[0]),
    display_w=int(scaled.shape[1]),
    display_h=int(scaled.shape[0]),
    factor=factor,
    mode=mode,
  )


def render_halfblocks(bitmap: Any, max_cols: int, max_rows: int) -> Rendered:
  """Convert ``bitmap`` and render it auto-fitted to the given cell budget."""
  source = to_rgb_array(bitmap)
  return render_source(source, fit_zoom(source, max_cols, max_rows))
