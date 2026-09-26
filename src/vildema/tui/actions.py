"""The pluggable pieces a project attaches to a `Browser`.

These types carry *no* Textual dependency on purpose: a project can define
visualizers and actions in their own modules, import them, and register them on
a `Browser` from elsewhere (the incremental / cross-file workflow). They are
also plain enough to unit-test without ever starting the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol

from vildema.data import ExperimentData

#: Anything the image renderer can turn into pixels: a numpy array (``H×W``
#: grayscale, ``H×W×3`` RGB, ``H×W×4`` RGBA, or a ``C×H×W`` tensor layout), a
#: PIL ``Image``, or a path to an image file. Kept as ``Any`` so this module —
#: and the core library — need no numpy/PIL import. See `vildema.tui.image`.
Bitmap = Any

#: A ``matplotlib.figure.Figure``. Kept as ``Any`` so this module never imports
#: matplotlib — it is only touched when a `Plotter` actually fires. See
#: `vildema.tui.display.open_figure_window`.
Figure = Any


@dataclass
class Selection:
  """What an action operates on when its key is pressed.

  v1 carries a single focused row plus the focused column path. ``entries``
  already exposes the row as a list so action code can be written against the
  eventual multi-select world without changing later.
  """

  focused: ExperimentData
  column: Optional[str] = None

  @property
  def entries(self) -> List[ExperimentData]:
    return [self.focused]


class BrowserContext(Protocol):
  """The safe surface an action may call back into.

  Deliberately tiny: actions get to notify the user, pop an image, or show a
  block of text, and nothing else. The running `Browser` implements this
  protocol; tests can supply a trivial stand-in.
  """

  def notify(self, message: str) -> None: ...

  def show_bitmap(self, bitmap: Bitmap, title: str = "image") -> None: ...

  def show_image(self, path: Path) -> None: ...

  def open_figure(self, figure: Figure, title: str = "plot") -> None: ...

  def open_text(self, text: str) -> None: ...


#: A key-bound callback. Must return quickly; heavy work should be acknowledged
#: via ``ctx.notify`` rather than blocking the event loop.
Action = Callable[[Selection, "BrowserContext"], None]


@dataclass
class Visualizer:
  """An action that produces an image for the focused entry and previews it.

  ``render`` returns a `Bitmap` — a numpy array, a PIL image, or a path to an
  image file. The `Browser` renders it in a popup inside the terminal (integer
  scaling only, no fractional resampling). The point is that a project builds a
  visualizer by handing back pixels (e.g. ``tensor.cpu().numpy()``) without
  writing any terminal-specific code.

  If ``column`` is set, the key only fires while that column is focused — use
  it to tie a visualizer to the column whose data it depicts, so it doesn't
  trigger from an unrelated cell. Left ``None``, it fires on any column.
  """

  key: str
  name: str
  render: Callable[[ExperimentData], Bitmap]
  column: Optional[str] = None

  def as_action(self) -> Action:
    def _action(selection: Selection, ctx: BrowserContext) -> None:
      if self.column is not None and selection.column != self.column:
        ctx.notify(f"Focus column '{self.column}' to view {self.name}.")
        return
      ctx.show_bitmap(self.render(selection.focused), title=self.name)

    return _action


@dataclass
class Plotter:
  """An action that builds a matplotlib figure for the focused entry and opens it.

  ``plot`` returns a ``matplotlib.figure.Figure`` (build it with
  ``matplotlib.figure.Figure()`` / ``fig.subplots()`` — no ``pyplot`` needed).
  The `Browser` pops it in a **real, interactive** matplotlib window, spawned in
  a separate process so the TUI is never blocked. The project supplies the plot;
  vildema handles the windowing.

  ``column`` works exactly as on `Visualizer`: set it to fire the key only while
  that column is focused, leave ``None`` to fire on any column.
  """

  key: str
  name: str
  plot: Callable[[ExperimentData], Figure]
  column: Optional[str] = None

  def as_action(self) -> Action:
    def _action(selection: Selection, ctx: BrowserContext) -> None:
      if self.column is not None and selection.column != self.column:
        ctx.notify(f"Focus column '{self.column}' to view {self.name}.")
        return
      ctx.open_figure(self.plot(selection.focused), title=self.name)

    return _action
