"""Stand-alone matplotlib window for a pickled figure.

Run as a fresh interpreter so it never touches the parent TUI's terminal/event
loop::

    python -m vildema.tui._plot_viewer <figure.pkl>

`vildema.tui.display.open_figure_window` pickles a figure to a temp file and
spawns this module; we unpickle it, re-home it onto a fresh pyplot-managed
window (an unpickled figure is not registered with pyplot, so ``plt.show()``
would otherwise display nothing), and block on the interactive window. The temp
file is deleted as soon as it is read.
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

# Interactive backends in preference order, each paired with the import that has
# to succeed for it to work. matplotlib otherwise falls back to the non-GUI
# ``agg`` backend, where ``plt.show()`` is a silent no-op (the "non-interactive
# window" symptom). The fallback below covers a venv with no GUI binding at all.
_INTERACTIVE_BACKENDS = [
  ("QtAgg", ("PyQt6", "PySide6", "PyQt5", "PySide2")),
  ("TkAgg", ("tkinter",)),
  ("GTK4Agg", ("gi",)),
]


def _use_interactive_backend() -> bool:
  """Switch matplotlib to the first importable interactive backend; report success."""
  import matplotlib

  for backend, modules in _INTERACTIVE_BACKENDS:
    if any(importlib.util.find_spec(m) is not None for m in modules):
      try:
        matplotlib.use(backend, force=True)
        return True
      except Exception:
        continue
  return False


def main() -> int:
  if not 2 <= len(sys.argv) <= 3:
    print("usage: python -m vildema.tui._plot_viewer <figure.pkl> [title]", file=sys.stderr)
    return 2

  path = Path(sys.argv[1])
  title = sys.argv[2] if len(sys.argv) == 3 else "plot"
  with path.open("rb") as fh:
    figure = pickle.load(fh)
  try:
    path.unlink()
  except OSError:
    pass

  if not _use_interactive_backend():
    # No GUI binding available: write a PNG and hand it to the system image
    # viewer so something still shows. Install a Qt backend (``pip install
    # PyQt6``) for a real, interactive matplotlib window.
    import tempfile

    from .display import open_image

    png = Path(tempfile.mkstemp(suffix=".png", prefix="vildema_plot_")[1])
    figure.savefig(png)
    print(
      "no interactive matplotlib backend (tkinter/Qt missing); opened a PNG "
      f"instead. `pip install PyQt6` for an interactive window. -> {png}",
      file=sys.stderr,
    )
    open_image(png)
    return 0

  import matplotlib.pyplot as plt

  # An unpickled figure carries no pyplot figure manager, so attach it to a
  # fresh one and reuse that window — otherwise plt.show() shows an empty frame.
  manager = plt.figure().canvas.manager
  manager.canvas.figure = figure
  figure.set_canvas(manager.canvas)
  try:
    manager.set_window_title(title)
  except Exception:
    pass

  # Close the window on `q` / Esc. (`q` is also matplotlib's built-in quit key,
  # but bind it explicitly so it works regardless of the active keymap.)
  def _on_key(event):
    if event.key in ("q", "escape"):
      plt.close(figure)

  figure.canvas.mpl_connect("key_press_event", _on_key)

  plt.show()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
