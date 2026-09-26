"""Image display backends for the browser.

v1 ships the *external viewer* path only: a visualizer hands us a PNG on disk
and we open it with the platform's default image viewer. Inline
terminal-graphics rendering is intentionally deferred (see the CLI UI spec).
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def open_image(path: Path) -> None:
  """Open ``path`` in the system's default image viewer.

  Non-blocking: the viewer is spawned and we return immediately so the TUI
  event loop is never stalled.
  """
  if sys.platform == "darwin":
    subprocess.Popen(["open", str(path)])
  elif os.name == "nt":
    # `startfile` only exists on Windows; guarded by the branch above.
    os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: F821
  else:
    subprocess.Popen(["xdg-open", str(path)])


def open_figure_window(figure: Any, title: str = "plot") -> None:
  """Open a matplotlib ``figure`` in a real, interactive window.

  Non-blocking: the figure is pickled to a temp file and shown by a fresh
  interpreter (``python -m vildema.tui._plot_viewer``), so the calling TUI keeps
  running and the window gets its own matplotlib event loop. matplotlib is only
  imported here, never at TUI import time. The viewer deletes the temp file once
  it has loaded it.
  """
  # Detach to a plain Agg canvas so pickling never trips over a live GUI canvas
  # the caller may have attached (e.g. a pyplot figure).
  from matplotlib.backends.backend_agg import FigureCanvasAgg

  FigureCanvasAgg(figure)
  fd, name = tempfile.mkstemp(suffix=".fig.pkl", prefix="vildema_")
  with os.fdopen(fd, "wb") as fh:
    pickle.dump(figure, fh)
  subprocess.Popen([sys.executable, "-m", "vildema.tui._plot_viewer", name, title])
