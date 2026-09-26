"""Cell formatting and the sort key used by the browser.

Kept free of any Textual import so it can be unit-tested on its own.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

#: A per-column formatter turns a raw cell value into its displayed string.
Formatter = Callable[[Any], str]

#: Default cap on rendered cell width; longer values are truncated.
DEFAULT_MAX_LEN = 40

#: Shown for a column that an entry simply does not have.
MISSING_CELL = ""

#: Ellipsis appended to truncated cells.
_ELLIPSIS = "…"


def format_value(value: object, max_len: int = DEFAULT_MAX_LEN) -> str:
  """Render ``value`` as a single, width-capped line.

  Strings pass through; everything else goes through ``str``. The result is
  truncated to ``max_len`` characters with a trailing ellipsis so heavy fields
  (checkpoints, trajectories) never blow out the table width.
  """
  text = value if isinstance(value, str) else str(value)
  text = text.replace("\n", " ")
  if max_len > 0 and len(text) > max_len:
    text = text[: max_len - 1] + _ELLIPSIS
  return text


#: Stops of the value gradient: low → mid → high. Chosen to read on the
#: solarized-light background (a muddy direct red→green is avoided by routing
#: through yellow).
_GRADIENT_LOW = (0xc0, 0x39, 0x2b)  # red
_GRADIENT_MID = (0xb7, 0x8a, 0x1f)  # amber
_GRADIENT_HIGH = (0x2e, 0x8b, 0x33)  # green


def _lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
  return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def gradient_hex(t: float) -> str:
  """Hex colour on a red→amber→green gradient for ``t`` clamped to ``[0, 1]``.

  ``0`` is red (low), ``0.5`` amber, ``1`` green (high).
  """
  t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
  if t < 0.5:
    r, g, b = _lerp(_GRADIENT_LOW, _GRADIENT_MID, t * 2)
  else:
    r, g, b = _lerp(_GRADIENT_MID, _GRADIENT_HIGH, (t - 0.5) * 2)
  return f"#{r:02x}{g:02x}{b:02x}"


def sort_key(value: object) -> Tuple[int, float, str]:
  """A total order over heterogeneous cell values.

  Returns a 3-tuple ``(bucket, number, text)`` so that comparisons never raise
  on mixed types: numbers sort numerically within their bucket, everything else
  sorts as text, and missing values (``None``) sort last.
  """
  if value is None:
    return (2, 0.0, "")
  if isinstance(value, bool):
    return (0, float(value), "")
  if isinstance(value, (int, float)):
    return (0, float(value), "")
  return (1, 0.0, str(value))
