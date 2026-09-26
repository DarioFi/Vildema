"""Interactive, ``htop``-style terminal browser for a `vildema.Database`.

Importing this package pulls in Textual; install it with the ``tui`` extra::

    pip install vildema[tui]

See ``cli_ui_specification.md`` for the design. Typical usage::

    from vildema import Database
    from vildema.tui import Browser

    Browser(Database().load_from_disk(...)).run()
"""

from .actions import Action, BrowserContext, Plotter, Selection, Visualizer
from .browser import Browser
from .formatting import Formatter, format_value

__all__ = [
  "Browser",
  "Visualizer",
  "Plotter",
  "Action",
  "Selection",
  "BrowserContext",
  "Formatter",
  "format_value",
]
