"""The `Browser` builder and the Textual app that renders it.

Textual is imported here (and only here) so the rest of `vildema.tui` — and the
core library — stay import-light. ``Browser`` itself does no UI work until
``run()`` is called.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import ScrollableContainer, VerticalScroll
from textual.coordinate import Coordinate
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import DataTable, Footer, Header, Input, Static, Tree
from textual.widgets.data_table import CursorType
from textual.widgets.tree import TreeNode
from rich.style import Style
from rich.text import Text

from vildema.constraints import C
from vildema.data import ExperimentData
from vildema.database import Database

from .actions import Action, Plotter, Selection, Visualizer
from .formatting import DEFAULT_MAX_LEN, Formatter, format_value, gradient_hex, sort_key

_HINTS = (
  "j/k move · s sort · e expand · [ ] fold/unfold · g fold all · u color · h hide · "
  "c cols · / filter · enter inspect · ? help · q quit"
)

#: Built-in theme we derive ours from, and the name of the derived theme.
_BASE_THEME = "solarized-light"
_THEME_NAME = "vildema-solarized"

#: A darker, fully-opaque gray for body text than solarized-light's default
#: (#586e75), for stronger contrast on the light background.
_TEXT_GRAY = "#33444b"


def _build_theme(base: Theme) -> Theme:
  """Solarized-light with darker, crisper body text."""
  return dataclasses.replace(
    base, name=_THEME_NAME, foreground=_TEXT_GRAY, text_alpha=1.0
  )


class _CrosshairDataTable(DataTable[str]):
  """A `DataTable` whose cell cursor also faintly tints its row and column.

  The focused cell keeps the normal, strong ``datatable--cursor`` highlight; the
  rest of its row and column get a fainter ``datatable--crosshair`` wash painted
  *underneath*, so the exact cell still stands out from its crosshair while the
  crosshair makes position easy to track across a wide table.
  """

  COMPONENT_CLASSES = DataTable.COMPONENT_CLASSES | {"datatable--crosshair"}

  @property
  def _crosshair_style(self) -> Style:
    return self.get_component_styles("datatable--crosshair").rich_style

  def _on_crosshair(self, row_index: int) -> bool:
    return (
            self.cursor_type == "cell"
            and self.show_cursor
            and row_index >= 0
            and row_index == self.cursor_coordinate.row
    )

  def _get_row_style(self, row_index: int, base_style: Style) -> Style:
    style = super()._get_row_style(row_index, base_style)
    if self._on_crosshair(row_index):
      style = style + self._crosshair_style  # faint tint along the focused row
    return style

  def _render_cell(
          self,
          row_index: int,
          column_index: int,
          base_style: Style,
          width: int,
          cursor: bool = False,
          hover: bool = False,
  ) -> "List[List[Any]]":
    if (
            self.cursor_type == "cell"
            and self.show_cursor
            and row_index >= 0
            and column_index >= 0
            and column_index == self.cursor_coordinate.column
    ):
      base_style = base_style + self._crosshair_style  # faint tint down the column
    return super()._render_cell(
      row_index, column_index, base_style, width, cursor=cursor, hover=hover
    )

  def watch_cursor_coordinate(self, old: Coordinate, new: Coordinate) -> None:
    # The base class only refreshes the single old/new cell; a crosshair needs
    # the full old and new row + column repainted.
    if self.cursor_type == "cell" and old != new:
      for coord in (old, new):
        self.refresh_row(coord.row)
        self.refresh_column(coord.column)
    super().watch_cursor_coordinate(old, new)


#: Sentinel for "this entry has no such column", kept distinct from a real
#: ``None`` value so a column present-here/absent-there isn't seen as constant.
_MISSING = object()


def _column_value(entry: ExperimentData, col: str) -> object:
  """The raw value of ``col`` for ``entry``, or ``_MISSING`` if absent."""
  return entry.get(col, _MISSING)


def _top_group(col: str) -> str:
  """The top-level group of a column: the path up to the first `C.SEP`."""
  return col.split(C.SEP, 1)[0]


def _under(col: str, prefix: str) -> bool:
  """Whether ``col`` is *strictly* nested under ``prefix`` (a proper ancestor).

  Unlike `_in_group`, an exact match is not "under": ``params//a`` is under
  ``params`` but ``params`` is not under itself.
  """
  return col.startswith(prefix + C.SEP)


def _parent_prefix(path: str) -> Optional[str]:
  """The path with its last `C.SEP` segment dropped, or ``None`` at top level."""
  head, sep, _ = path.rpartition(C.SEP)
  return head if sep else None


def _child_prefix(col: str, ancestor: str) -> str:
  """The immediate child of ``ancestor`` on the way down to ``col``.

  For ``col='a//b//c//d'`` and ``ancestor='a'`` this is ``'a//b'``: one level
  deeper than ``ancestor`` towards ``col``.
  """
  rest = col[len(ancestor) + len(C.SEP):]
  return ancestor + C.SEP + rest.split(C.SEP, 1)[0]


@dataclasses.dataclass
class _DisplayColumn:
  """One column as actually shown: a single leaf, or a collapsed group of them."""

  key: str  # leaf path, or group name when collapsed
  is_group: bool
  members: List[str]  # the leaf(s) this display column stands for


class Browser:
  """A builder for an interactive, ``htop``-style view of a `Database`.

  Every registration method mutates the builder and returns ``self``, so a
  project script can chain calls or extend the same object across files. The UI
  is not started until `run` is called.
  """

  def __init__(self, *, title: Optional[str] = None) -> None:
    self._db: Database = Database()
    self._title = title
    self._visible_override: Optional[List[str]] = None
    self._hidden: List[str] = []
    self._hide_constant: bool = False
    self._hide_patterns: List[str] = []
    self._formatters: Dict[str, Formatter] = {}
    self._truncate_len = DEFAULT_MAX_LEN
    self._bindings: Dict[str, Tuple[str, Action]] = {}
    self._visualizers: List[Visualizer] = []
    self._plotters: List[Plotter] = []
    self._sort: Optional[Tuple[str, bool]] = None
    self._colorized: List[str] = []

  # -- registration API (all return self) ----------------------------------

  def attach_db(self, db: Database) -> "Browser":
    self._db = db
    return self

  def show_columns(self, cols: Sequence[str]) -> "Browser":
    """Set the ordered, visible column set (overrides the default of *all*)."""
    self._visible_override = list(cols)
    return self

  def hide_columns(self, cols: Sequence[str]) -> "Browser":
    """Drop ``cols`` from whatever the visible set would otherwise be.

    Each entry is treated as a group prefix: hiding ``"params"`` hides
    ``params`` and every ``params//…`` column at once. Hidden columns stay in
    the pool, so they can still be re-enabled from the `c` picker.
    """
    self._hidden.extend(cols)
    return self

  def hide_columns_from_pattern(self, pattern: str) -> "Browser":
    """Hide every column matched by ``pattern`` (see `C.expand`): a literal
    group-prefix if every `C.SEP`-segment of ``pattern`` is plain
    word/digit/underscore/dot characters (equivalent to passing it to
    `hide_columns`), otherwise a regex matched with `re.search` anywhere in
    the column path unless anchored. Matched columns stay in the pool, so
    they can still be re-enabled from the `c` picker.

    Like `hide_constant_columns`, this only records the pattern: the matching
    columns are computed lazily from the pool when the view is resolved (at
    `run`), so the result does not depend on when the database is attached or on
    where in the builder chain this is called.
    """
    self._hide_patterns.append(pattern)
    return self

  def hide_constant_columns(self) -> "Browser":
    """Hide every column whose value is identical across all entries.

    A QoL filter that drops the fields which don't vary (shared hyperparams,
    fixed settings, all-missing columns) so only the columns that actually
    distinguish runs remain.

    This only records the *intent*: the constant columns are computed lazily
    from the entries when the view is resolved (at `run`), not at call time. So
    the result is independent of where in the builder chain this is called and
    of when the database is attached — the settings are permutation invariant.
    """
    self._hide_constant = True
    return self

  def format(self, col: str, fn: Formatter) -> "Browser":
    """Use ``fn`` to render cells of ``col`` instead of the default truncation."""
    self._formatters[col] = fn
    return self

  def truncate(self, max_len: int) -> "Browser":
    """Set the global cell-width cap used by the default formatter."""
    self._truncate_len = max_len
    return self

  def bind(self, key: str, name: str, action: Action) -> "Browser":
    """Bind ``key`` to ``action`` (overriding any previous bind on that key)."""
    self._bindings[key] = (name, action)
    return self

  def add_visualizer(self, viz: Visualizer) -> "Browser":
    """Register an image visualizer under its own key."""
    self._visualizers.append(viz)
    return self

  def add_plotter(self, plotter: Plotter) -> "Browser":
    """Register a matplotlib plotter under its own key."""
    self._plotters.append(plotter)
    return self

  def sort_default(self, col: str, descending: bool = False) -> "Browser":
    """Sort by ``col`` on startup."""
    self._sort = (col, descending)
    return self

  def colorize_columns(self, cols: "str | Sequence[str]") -> "Browser":
    """Start with ``cols`` colour-coded on a red→green value gradient.

    The same per-column effect as the in-app ``u`` key, applied on startup. Each
    entry is a group prefix, so ``colorize_columns("metrics")`` colours every
    ``metrics//…`` leaf at once (just like `hide_columns`). A bare string is
    accepted as a single column/group. Only numeric cells take colour (others are
    left as-is), and the gradient is scaled to each leaf column's own min/max over
    the visible rows. Calls accumulate.
    """
    if isinstance(cols, str):
      cols = [cols]
    self._colorized.extend(cols)
    return self

  def pipe(self, *presets: Callable[["Browser"], Any]) -> "Browser":
    """Apply reusable preset functions to this browser, in order, and return self.

    A composition point for project-specific defaults: each ``preset`` takes the
    browser and configures it (its return value is ignored, so a preset may
    either chain-and-return or just mutate). This lets functional presets slot
    into the fluent chain instead of breaking it::

        Browser(title="…").attach_db(db).pipe(apply_formatting, hide_useless_cols)
    """
    for preset in presets:
      preset(self)
    return self

  # -- derived state --------------------------------------------------------

  def all_columns(self) -> List[str]:
    """The full pool of columns offered by the picker (override, else all DB columns).

    Hidden columns stay in this pool so the user can re-enable them with ``c``;
    hiding only affects which columns are *initially* visible.
    """
    if self._visible_override is not None:
      return list(self._visible_override)
    return self._db.columns()

  def resolve_columns(self) -> List[str]:
    """The initially visible columns: the full pool minus any hidden group.

    The `hide_constant_columns` and `hide_columns_from_pattern` settings are
    resolved here (against the current pool/entries) and hidden too — so the
    result is the same no matter when those toggles or the database attachment
    happened.
    """
    hidden = list(self._hidden)
    if self._hide_constant:
      hidden.extend(self._constant_columns())
    pool = self.all_columns()
    hidden.extend(self._pattern_hidden_columns())
    hidden_names = {m.name for group in hidden for m in C(group).expand(pool)}
    return [c for c in pool if c not in hidden_names]

  def _constant_columns(self) -> List[str]:
    """Columns whose value is identical across all current entries.

    Computed on demand from `self._db.entries`; empty with fewer than two
    entries (nothing can be "constant across runs" then).
    """
    entries = self._db.entries
    if len(entries) < 2:
      return []
    constant: List[str] = []
    for col in self._db.columns():
      first = _column_value(entries[0], col)
      if all(_column_value(e, col) == first for e in entries[1:]):
        constant.append(col)
    return constant

  def _pattern_hidden_columns(self) -> List[str]:
    """Columns in the current pool matching any registered hide pattern."""
    if not self._hide_patterns:
      return []
    pool = self.all_columns()
    return sorted({m.name for p in self._hide_patterns for m in C(p).expand(pool)})

  def action_map(self) -> Dict[str, Tuple[str, Action]]:
    """Key -> (label, action); explicit binds override plotters, which override visualizers."""
    merged: Dict[str, Tuple[str, Action]] = {
      viz.key: (viz.name, viz.as_action()) for viz in self._visualizers
    }
    merged.update({p.key: (p.name, p.as_action()) for p in self._plotters})
    merged.update(self._bindings)
    return merged

  # -- entry point ----------------------------------------------------------

  def run(self) -> None:
    """Enter the interactive TUI (blocks until the user quits)."""
    _BrowserApp(self).run()


class _TextScreen(ModalScreen[None]):
  """A scrollable, read-only block of text (inspect view, help, action text)."""

  BINDINGS = [("escape,q", "dismiss", "Close")]

  def __init__(self, title: str, body: str) -> None:
    super().__init__()
    self._title = title
    self._body = body

  def compose(self) -> ComposeResult:
    with VerticalScroll(id="textbox"):
      yield Static(f"[b]{self._title}[/b]\n\n{self._body}", id="textbody")
    yield Static("↑/↓ scroll · q / esc close", id="texthint")

  def on_mount(self) -> None:
    # Focus the scroller so long content (e.g. the full keybinding list) can be
    # scrolled to with the arrow / page keys instead of being cut off.
    self.query_one("#textbox").focus()


class _ImageScreen(ModalScreen[None]):
  """An in-terminal image preview (integer-scaled half-blocks).

  Opens auto-fitted to the viewport; ``+``/``-`` change the integer zoom and
  ``f`` returns to fit. When the image is larger than the viewport the arrow
  keys / page keys scroll it.
  """

  BINDINGS = [
    Binding("escape,q", "dismiss", "Close"),
    Binding("plus,equals_sign,kp_plus", "zoom(1)", "Zoom in"),
    Binding("minus,underscore,kp_minus", "zoom(-1)", "Zoom out"),
    Binding("f,0", "fit", "Fit"),
  ]

  def __init__(self, bitmap: object, title: str) -> None:
    super().__init__()
    self._bitmap = bitmap
    self._title = title
    self._source: Any = None  # normalized RGB array (numpy), filled on mount
    self._zoom = 1
    self._error: Optional[str] = None

  def compose(self) -> ComposeResult:
    with ScrollableContainer(id="imagebox"):
      yield Static(id="image")
    yield Static(id="imagehint")

  def on_mount(self) -> None:
    from .image import fit_zoom, to_rgb_array

    try:
      self._source = to_rgb_array(self._bitmap)
    except Exception as exc:  # noqa: BLE001 - surface any conversion failure
      self._error = str(exc)
      self.query_one("#image", Static).update(f"Cannot render image: {exc}")
      self.query_one("#imagehint", Static).update(
        f"[b]{self._title}[/b]  —  q / esc: close"
      )
      return
    size = self.app.size
    self._zoom = fit_zoom(self._source, size.width - 2, size.height - 3)
    self._refresh_image()
    self.query_one("#imagebox").focus()  # so arrow / page keys scroll

  def _refresh_image(self) -> None:
    from .image import render_source

    rendered = render_source(self._source, self._zoom)
    self.query_one("#image", Static).update(rendered.text)
    self.query_one("#imagehint", Static).update(
      f"[b]{self._title}[/b]  —  {rendered.describe()}"
      "  —  +/- zoom · f fit · q/esc close"
    )

  def action_zoom(self, delta: int) -> None:
    from .image import clamp_zoom, step_zoom

    if self._error is not None:
      return
    self._zoom = clamp_zoom(step_zoom(self._zoom, delta), self._source)
    self._refresh_image()

  def action_fit(self) -> None:
    from .image import fit_zoom

    if self._error is not None:
      return
    size = self.app.size
    self._zoom = fit_zoom(self._source, size.width - 2, size.height - 3)
    self._refresh_image()


class _FilterScreen(ModalScreen[Optional[str]]):
  """Prompt for a substring filter; dismiss returns the entered text."""

  BINDINGS = [("escape", "cancel", "Cancel")]

  def __init__(self, current: str) -> None:
    super().__init__()
    self._current = current

  def compose(self) -> ComposeResult:
    yield Input(value=self._current, placeholder="filter rows (substring)…")

  def on_input_submitted(self, event: Input.Submitted) -> None:
    self.dismiss(event.value)

  def action_cancel(self) -> None:
    self.dismiss(None)


class _ColumnScreen(ModalScreen[Optional[List[str]]]):
  """Show/hide picker as a `C.SEP` tree; dismiss returns the new visible set.

  ``enter``, ``space``, or a mouse click toggles the highlighted node — a leaf
  column, or a whole group when on a group node. ``escape`` applies and closes
  (changes are always kept; there is no discard). Group nodes show a
  ``(selected/total)`` tally. Fold/unfold a group with ←/→ or tab (or by
  clicking its arrow).
  """

  BINDINGS = [
    Binding("escape", "done", "Done"),
    # Take space before the Tree's own fold binding so it toggles selection.
    Binding("space", "toggle_selected", "Toggle", priority=True),
    # Folding moved off space onto the arrows / tab.
    Binding("right", "unfold", "Unfold", show=False),
    Binding("left", "fold", "Fold", show=False),
    Binding("tab", "toggle_fold", "Fold/unfold", priority=True, show=False),
  ]

  def __init__(self, all_columns: Sequence[str], visible: Sequence[str]) -> None:
    super().__init__()
    self._all = list(all_columns)
    self._selected: set[str] = {c for c in visible if c in set(all_columns)}
    self._group_seg: Dict[TreeNode[Optional[str]], str] = {}

  def compose(self) -> ComposeResult:
    yield Static(
      "[b]Columns[/b]  —  enter / space / click: toggle · ←/→ or tab: fold · escape: done"
    )
    yield Tree("columns", id="coltree")
    yield Footer()

  def on_mount(self) -> None:
    tree: Tree[Optional[str]] = self.query_one(Tree)
    tree.show_root = False
    tree.root.expand()
    nodes: Dict[Tuple[str, ...], TreeNode[Optional[str]]] = {}
    for col in self._all:
      parts = col.split(C.SEP)
      parent = tree.root
      prefix: Tuple[str, ...] = ()
      for seg in parts[:-1]:
        prefix = prefix + (seg,)
        if prefix not in nodes:
          group_node = parent.add(seg, data=None, expand=True)
          self._group_seg[group_node] = seg
          nodes[prefix] = group_node
        parent = nodes[prefix]
      parent.add_leaf(parts[-1], data=col)
    self._refresh(tree.root)

  def _descendant_cols(self, node: "TreeNode[Optional[str]]") -> List[str]:
    cols: List[str] = [] if node.data is None else [str(node.data)]
    for child in node.children:
      cols.extend(self._descendant_cols(child))
    return cols

  def _refresh(self, node: "TreeNode[Optional[str]]") -> None:
    # Labels are Text, not markup strings, so "[x]" renders literally.
    if node.data is not None:  # leaf column
      col = str(node.data)
      mark = "[x]" if col in self._selected else "[ ]"
      node.set_label(Text(f"{mark} {col.split(C.SEP)[-1]}"))
    elif node in self._group_seg:  # group (the hidden root is skipped)
      cols = self._descendant_cols(node)
      chosen = sum(c in self._selected for c in cols)
      node.set_label(Text(f"{self._group_seg[node]}  ({chosen}/{len(cols)})"))
    for child in node.children:
      self._refresh(child)

  def _toggle(self, node: "TreeNode[Optional[str]]") -> None:
    cols = self._descendant_cols(node)
    if not cols:
      return
    if all(c in self._selected for c in cols):
      self._selected.difference_update(cols)  # turn the whole subtree off
    else:
      self._selected.update(cols)  # turn it on
    self._refresh(self.query_one(Tree).root)

  def on_tree_node_selected(self, event: "Tree.NodeSelected[Optional[str]]") -> None:
    self._toggle(event.node)  # enter key or mouse click

  def action_toggle_selected(self) -> None:
    node = self.query_one(Tree).cursor_node  # space key
    if node is not None:
      self._toggle(node)

  def action_unfold(self) -> None:
    node = self.query_one(Tree).cursor_node  # right arrow
    if node is not None and node.allow_expand and node.children:
      node.expand()

  def action_fold(self) -> None:
    node = self.query_one(Tree).cursor_node  # left arrow
    if node is not None and node.children:
      node.collapse()

  def action_toggle_fold(self) -> None:
    node = self.query_one(Tree).cursor_node  # tab
    if node is not None and node.children:
      node.collapse() if node.is_expanded else node.expand()

  def action_done(self) -> None:
    self.dismiss([c for c in self._all if c in self._selected])


class _ShortcutCommands(Provider):
  """Surfaces every browser shortcut in the command palette (ctrl+p)."""

  def _commands(self) -> List[Tuple[str, str, Callable[[], None]]]:
    return cast("_BrowserApp", self.app).shortcut_commands()

  async def discover(self) -> Hits:
    for title, help_text, callback in self._commands():
      yield DiscoveryHit(title, callback, help=help_text)

  async def search(self, query: str) -> Hits:
    matcher = self.matcher(query)
    for title, help_text, callback in self._commands():
      score = matcher.match(title)
      if score > 0:
        yield Hit(score, matcher.highlight(title), callback, help=help_text)


class _BrowserApp(App[None]):
  """The Textual application that renders a `Browser`."""

  COMMANDS = App.COMMANDS | {_ShortcutCommands}

  CSS = """
  Static#hints { dock: bottom; color: $text-muted; padding: 0 1; }
  _TextScreen #textbox { background: $panel; padding: 1 2; }
  _TextScreen #texthint { dock: bottom; color: $text-muted; padding: 0 1; }
  /* Two-level crosshair: a faint wash along the focused row + column, and a
     stronger (but still non-bold) tint on the exact focused cell so it stands
     out from its crosshair. */
  DataTable > .datatable--crosshair {
    background: $primary 12%;
  }
  DataTable > .datatable--cursor,
  DataTable:focus > .datatable--cursor {
    background: $primary 35%;
    color: $foreground;
    text-style: none;
  }
  /* Default header-cursor is dark text on dark accent (unreadable on this
     light theme); use a light tint with dark, bold text instead. */
  DataTable > .datatable--header-cursor {
    background: $primary 25%;
    color: $foreground;
    text-style: bold;
  }
  /* Image preview popup. */
  _ImageScreen #imagebox { align: center middle; background: $surface; }
  _ImageScreen #image { width: auto; height: auto; }
  _ImageScreen #imagehint { dock: bottom; color: $text-muted; padding: 0 1; }
  """

  def __init__(self, browser: Browser) -> None:
    super().__init__()
    self._browser = browser
    self._all_columns: List[str] = browser.all_columns()
    self._visible: List[str] = browser.resolve_columns()
    self._expanded: set[str] = set()
    # Group prefixes (at any tree depth) folded into a single column. When two
    # nested prefixes both fold a leaf, the shortest (outermost) one wins.
    self._collapsed: set[str] = set()
    # Colorized leaf columns. Builder entries are group prefixes, expanded here
    # to the concrete leaves they cover so the `u` toggle works per-subcolumn.
    self._colorized: set[str] = {
      m.name for prefix in browser._colorized for m in C(prefix).expand(self._all_columns)
    }
    self._color_ranges: Dict[str, Tuple[float, float]] = {}  # col -> (min, max)
    self._display_cols: List[_DisplayColumn] = []  # what each table column maps to
    self._filter = ""
    self._rows: List[ExperimentData] = []
    self._actions = browser.action_map()
    if browser._sort is not None:
      self._sort_col: Optional[str] = browser._sort[0]
      self._sort_desc = browser._sort[1]
    else:
      self._sort_col = None
      self._sort_desc = False

  # -- layout ---------------------------------------------------------------

  def compose(self) -> ComposeResult:
    yield Header()
    yield _CrosshairDataTable(
      cursor_type="cell", zebra_stripes=True, cursor_background_priority="css"
    )
    yield Static(_HINTS, id="hints")
    yield Footer()

  def on_mount(self) -> None:
    self.register_theme(_build_theme(self.available_themes[_BASE_THEME]))
    self.theme = _THEME_NAME
    self.title = self._browser._title or "vildema"
    table = self.query_one(DataTable)
    table.focus()
    self._rebuild()

  # -- table construction ---------------------------------------------------

  def _entry_value(self, entry: ExperimentData, col: str) -> object:
    """The raw value of ``col`` for ``entry``, or ``None`` if it has no such key."""
    return entry.get(col, None)

  def _cell(self, entry: ExperimentData, col: str) -> str:
    value = self._entry_value(entry, col)
    if value is None:
      return ""
    fmt = self._browser._formatters.get(col)
    if fmt is not None:
      return fmt(value)
    # Expanded columns bypass truncation (max_len <= 0 disables it).
    max_len = 0 if col in self._expanded else self._browser._truncate_len
    return format_value(value, max_len)

  def _current_entries(self) -> List[ExperimentData]:
    entries = list(self._browser._db.entries)
    if self._filter:
      needle = self._filter.lower()
      entries = [
        e for e in entries if any(needle in self._cell(e, c).lower() for c in self._visible)
      ]
    if self._sort_col is not None:
      col = self._sort_col
      entries.sort(key=lambda e: sort_key(self._entry_value(e, col)), reverse=self._sort_desc)
    return entries

  def _header_label(self, col: str) -> str:
    """Column header, with a sort arrow appended when it is the sort key."""
    if col == self._sort_col:
      return f"{col} {'▼' if self._sort_desc else '▲'}"
    return col

  def _display_key(self, col: str) -> str:
    """How ``col`` is shown: its outermost collapsed ancestor, else ``col`` itself."""
    ancestors = [p for p in self._collapsed if _under(col, p)]
    return min(ancestors, key=len) if ancestors else col

  def _build_display(self) -> List["_DisplayColumn"]:
    """Map the visible leaves to display columns, folding collapsed groups.

    A collapsed prefix can sit at any depth, so several leaves may share one
    display key; consecutive leaves with the same key become a single group
    column.
    """
    display: List[_DisplayColumn] = []
    seen: set[str] = set()
    for col in self._visible:
      key = self._display_key(col)
      if key == col:
        display.append(_DisplayColumn(key=col, is_group=False, members=[col]))
      elif key not in seen:
        seen.add(key)
        members = [c for c in self._visible if self._display_key(c) == key]
        display.append(_DisplayColumn(key=key, is_group=True, members=members))
    return display

  def _display_header(self, dc: "_DisplayColumn") -> str:
    if dc.is_group:
      return f"▸ {dc.key}"  # collapsed-group marker
    return self._header_label(dc.key)

  def _display_cell(self, entry: ExperimentData, dc: "_DisplayColumn") -> "str | Text":
    if dc.is_group:
      return f"#{len(dc.members)} cols"
    text = self._cell(entry, dc.key)
    if dc.key in self._colorized:
      return self._colorize(entry, dc.key, text)
    return text

  def _numeric(self, value: object) -> Optional[float]:
    """``value`` as a float if it is a real number (``bool`` excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      return None
    return float(value)

  def _colorize(self, entry: ExperimentData, col: str, text: str) -> "str | Text":
    """Tint ``text`` on a red→green gradient by the cell's value within ``col``."""
    value = self._numeric(self._entry_value(entry, col))
    rng = self._color_ranges.get(col)
    if value is None or rng is None:
      return text  # non-numeric / missing cells stay uncoloured
    lo, hi = rng
    t = 0.5 if hi == lo else (value - lo) / (hi - lo)
    return Text(text, style=gradient_hex(t))

  def _compute_color_ranges(self) -> None:
    """Cache each colourised column's numeric (min, max) over the visible rows.

    Ranges are per leaf, so a colorized group (e.g. ``metrics``) gives every
    subcolumn its own gradient scaled to that subcolumn's values.
    """
    self._color_ranges = {}
    for col in self._colorized:
      nums = [
        n
        for e in self._rows
        if (n := self._numeric(self._entry_value(e, col))) is not None
      ]
      if nums:
        self._color_ranges[col] = (min(nums), max(nums))

  def _rebuild(self) -> None:
    table = self.query_one(DataTable)
    coordinate = table.cursor_coordinate
    table.clear(columns=True)
    self._display_cols = self._build_display()
    if not self._display_cols:
      self._rows = []
      return
    table.add_columns(*(self._display_header(dc) for dc in self._display_cols))
    self._rows = self._current_entries()
    self._compute_color_ranges()
    for index, entry in enumerate(self._rows, start=1):
      # The row label is a non-selectable left gutter, here a 1-based row number.
      cells = [self._display_cell(entry, dc) for dc in self._display_cols]
      table.add_row(*cells, label=str(index))
    if table.row_count:
      max_row = table.row_count - 1
      max_col = len(self._display_cols) - 1
      table.cursor_coordinate = coordinate.__class__(
        min(coordinate.row, max_row), min(coordinate.column, max_col)
      )

  # -- focus helpers --------------------------------------------------------

  def _focused_entry(self) -> Optional[ExperimentData]:
    row = self.query_one(DataTable).cursor_row
    if 0 <= row < len(self._rows):
      return self._rows[row]
    return None

  def _focused_display(self) -> Optional["_DisplayColumn"]:
    col = self.query_one(DataTable).cursor_column
    if 0 <= col < len(self._display_cols):
      return self._display_cols[col]
    return None

  def _focused_column(self) -> Optional[str]:
    dc = self._focused_display()
    return dc.key if dc is not None else None

  # -- command palette ------------------------------------------------------

  def shortcut_commands(self) -> List[Tuple[str, str, Callable[[], None]]]:
    """(title, help, callback) for every shortcut, for the ctrl+p palette."""
    commands: List[Tuple[str, str, Callable[[], None]]] = [
      ("Sort by focused column", "key: s", self._sort_focused),
      ("Expand / collapse focused cell text", "key: e", self._expand_focused),
      ("Collapse focused column one level", "key: [", self._collapse_one),
      ("Expand focused column one level", "key: ]", self._expand_one),
      ("Collapse / expand all groups", "key: g", self._collapse_all),
      ("Colour-code focused column (red→green gradient)", "key: u", self._colorize_focused),
      ("Hide focused column or group", "key: h", self._hide_focused),
      ("Choose columns (tree)", "key: c", self._open_columns),
      ("Filter rows", "key: /", self._open_filter),
      ("Inspect focused row", "key: enter", self._inspect_focused),
      ("Show keybindings help", "key: ?", self._open_help),
      ("Quit", "key: q", self.exit),
    ]
    for key, (label, action) in sorted(self._actions.items()):
      commands.append((label, f"key: {key}", self._action_runner(action)))
    return commands

  def _action_runner(self, action: Action) -> Callable[[], None]:
    """A zero-arg callback (for the palette) that runs ``action`` on the selection."""
    return lambda: self._run_action(action)

  # -- key handling ---------------------------------------------------------

  def on_key(self, event: events.Key) -> None:
    # A modal screen is open: let it handle its own keys.
    if len(self.screen_stack) > 1:
      return

    key = event.key
    if key in self._actions:
      event.stop()
      self._run_action(self._actions[key][1])
    elif key == "j":
      event.stop()
      self.query_one(DataTable).action_cursor_down()
    elif key == "k":
      event.stop()
      self.query_one(DataTable).action_cursor_up()
    elif key == "s":
      event.stop()
      self._sort_focused()
    elif key == "e":
      event.stop()
      self._expand_focused()
    elif key == "left_square_bracket":
      event.stop()
      self._collapse_one()
    elif key == "right_square_bracket":
      event.stop()
      self._expand_one()
    elif key == "g":
      event.stop()
      self._collapse_all()
    elif key == "u":
      event.stop()
      self._colorize_focused()
    elif key == "h":
      event.stop()
      self._hide_focused()
    elif key == "c":
      event.stop()
      self._open_columns()
    elif key == "slash":
      event.stop()
      self._open_filter()
    elif key == "enter":
      event.stop()
      self._inspect_focused()
    elif key == "question_mark":
      event.stop()
      self._open_help()
    elif key == "q":
      event.stop()
      self.exit()

  def _run_action(self, action: Action) -> None:
    entry = self._focused_entry()
    if entry is None:
      self.notify("No row selected.")
      return
    action(Selection(entry, self._focused_column()), self)

  def _sort_focused(self) -> None:
    dc = self._focused_display()
    if dc is None:
      return
    if dc.is_group:
      self.notify(f"'{dc.key}' is a collapsed group — press ] to expand, then sort.")
      return
    col = dc.key
    if self._sort_col == col:
      self._sort_desc = not self._sort_desc
    else:
      self._sort_col = col
      self._sort_desc = False
    self._rebuild()
    arrow = "↓" if self._sort_desc else "↑"
    self.notify(f"Sorted by {col} {arrow}")

  def _expand_focused(self) -> None:
    dc = self._focused_display()
    if dc is None:
      return
    if dc.is_group:
      self.notify(f"'{dc.key}' is a collapsed group — press g to expand it.")
      return
    col = dc.key
    if col in self._expanded:
      self._expanded.discard(col)
      self.notify(f"Collapsed column {col}")
    else:
      self._expanded.add(col)
      self.notify(f"Expanded column {col}")
    self._rebuild()

  def _collapse_one(self) -> None:
    """Fold the focused column one level shallower (toward the tree root).

    On a leaf this folds it with its siblings into their parent group; on an
    already-folded group it folds that group up into *its* parent, one step at
    a time, so repeated presses walk steadily up the tree.
    """
    dc = self._focused_display()
    if dc is None:
      return
    target = _parent_prefix(dc.key)
    if target is None:
      self.notify(f"'{dc.key}' is already at the top level.")
      return
    # Drop any deeper collapses now subsumed by (folded inside) the new group.
    self._collapsed = {
      p for p in self._collapsed if p != target and not _under(p, target)
    }
    self._collapsed.add(target)
    self._rebuild()
    self.notify(f"Collapsed to {target}")

  def _expand_one(self) -> None:
    """Unfold the focused group by exactly one level (toward the leaves).

    Each immediate child that still has nesting is re-folded so the reveal stops
    after a single level; pressing again drills down further.
    """
    dc = self._focused_display()
    if dc is None:
      return
    if not dc.is_group:
      self.notify(f"'{dc.key}' is already fully expanded.")
      return
    self._collapsed.discard(dc.key)
    for col in dc.members:
      child = _child_prefix(col, dc.key)
      if _under(col, child):  # the child is itself a group, not a bare leaf
        self._collapsed.add(child)
    self._rebuild()
    self.notify(f"Expanded {dc.key}")

  def _collapse_all(self) -> None:
    """Toggle between a fully folded (top-level groups) and fully unfolded view."""
    if self._collapsed:
      self._collapsed.clear()
      self.notify("Expanded all groups")
      self._rebuild()
      return
    groups = {
      _top_group(c) for c in self._visible if _top_group(c) != c
    }
    self._collapsed = {
      g for g in groups if sum(_under(c, g) for c in self._visible) > 1
    }
    if self._collapsed:
      self.notify("Collapsed all to top level")
    else:
      self.notify("Nothing to collapse.")
    self._rebuild()

  def _colorize_focused(self) -> None:
    """Toggle the focused column's red→green value gradient (a per-column flag)."""
    dc = self._focused_display()
    if dc is None:
      return
    if dc.is_group:
      self.notify(f"'{dc.key}' is a collapsed group — press ] to expand it.")
      return
    col = dc.key
    if col in self._colorized:
      self._colorized.discard(col)
      self.notify(f"Uncoloured column {col}")
    else:
      self._colorized.add(col)
      self.notify(f"Colour-coded column {col}")
    self._rebuild()

  def _hide_focused(self) -> None:
    dc = self._focused_display()
    if dc is None:
      return
    remaining = [c for c in self._visible if c not in dc.members]
    if not remaining:  # never hide the last visible column(s)
      return
    self._visible = remaining
    # Drop the folded group and any deeper folds inside it (no-op for a leaf).
    self._collapsed = {
      p for p in self._collapsed if p != dc.key and not _under(p, dc.key)
    }
    self._rebuild()
    label = f"group {dc.key}" if dc.is_group else f"column {dc.key}"
    self.notify(f"Hid {label}")

  def _open_columns(self) -> None:
    def apply(result: Optional[List[str]]) -> None:
      if result is not None and result:
        self._visible = result
        self._rebuild()

    self.push_screen(_ColumnScreen(self._all_columns, self._visible), apply)

  def _open_filter(self) -> None:
    def apply(result: Optional[str]) -> None:
      if result is not None:
        self._filter = result
        self._rebuild()

    self.push_screen(_FilterScreen(self._filter), apply)

  def _inspect_focused(self) -> None:
    entry = self._focused_entry()
    if entry is None:
      return
    origin = f"origin: {entry.file_origin}\n\n" if entry.file_origin is not None else ""
    self.push_screen(_TextScreen("Row detail", origin + _pretty(entry.data)))

  def _open_help(self) -> None:
    lines = [
      "j / k, ↑ / ↓     move row",
      "← / →            move column",
      "s                sort by focused column (toggle)",
      "e                expand/collapse focused column (full text)",
      "[ / ]            collapse / expand focused column one tree level",
      "g                collapse / expand all groups (toggle)",
      "u                colour-code focused column (red→green, toggle)",
      "h                hide focused column (or whole group)",
      "c                show/hide columns",
      "/                filter rows",
      "enter            inspect focused row",
      "ctrl+p           command palette (search all actions)",
      "?                this help",
      "q                quit",
    ]
    custom = [f"{key:<16} {label}" for key, (label, _) in sorted(self._actions.items())]
    if custom:
      lines.append("")
      lines.append("-- project keys --")
      lines.extend(custom)
    else:
      lines.append("")
      lines.append("(no project keys registered)")
    self.push_screen(_TextScreen("Keybindings", "\n".join(lines)))

  # -- BrowserContext implementation ----------------------------------------

  def show_bitmap(self, bitmap: object, title: str = "image") -> None:
    self.push_screen(_ImageScreen(bitmap, title))

  def show_image(self, path: Path) -> None:
    from .display import open_image

    open_image(path)
    self.notify(f"Opened {path.name}")

  def open_figure(self, figure: object, title: str = "plot") -> None:
    from .display import open_figure_window

    open_figure_window(figure, title)
    self.notify(f"Opened {title} window")

  def open_text(self, text: str) -> None:
    self.push_screen(_TextScreen("Output", text))


def _pretty(value: object, indent: int = 0) -> str:
  """Render an entry (nested mappings / sequences) as readable, full text."""
  pad = "  " * indent
  if isinstance(value, Mapping):
    if not value:
      return "{}"
    parts = []
    for key, val in value.items():
      rendered = _pretty(val, indent + 1)
      sep = "\n" if isinstance(val, (Mapping, tuple, list)) and val else " "
      parts.append(f"{pad}  {key}:{sep}{rendered}")
    return "\n".join(parts) if indent == 0 else "\n" + "\n".join(parts)
  if isinstance(value, (tuple, list)):
    if not value:
      return "[]"
    return ", ".join(_pretty(v, indent + 1) for v in value)
  return str(value)
