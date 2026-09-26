import asyncio
import unittest
from pathlib import Path

from vildema.database import Database, ExperimentData
from vildema.tui import Browser, Plotter, Selection, Visualizer, format_value
from vildema.tui.browser import _BrowserApp, _pretty
from vildema.tui.formatting import sort_key


def _db() -> Database:
  db = Database()
  db.entries = [
    ExperimentData({"name": "a", "cfg": {"lr": 0.1}, "acc": 0.9}),
    ExperimentData({"name": "b", "cfg": {"lr": 0.01}, "acc": 0.5}),
    ExperimentData({"name": "c", "acc": 0.7}),  # missing cfg//lr
  ]
  return db


class FormattingTest(unittest.TestCase):
  def test_strings_pass_through(self):
    self.assertEqual(format_value("hello"), "hello")

  def test_long_values_truncated_with_ellipsis(self):
    out = format_value("x" * 100, max_len=10)
    self.assertEqual(len(out), 10)
    self.assertTrue(out.endswith("…"))

  def test_newlines_flattened(self):
    self.assertEqual(format_value("a\nb"), "a b")

  def test_sort_key_orders_numbers_then_text_then_missing(self):
    keys = [sort_key(1.0), sort_key("z"), sort_key(None)]
    self.assertEqual(keys, sorted(keys))


class BuilderTest(unittest.TestCase):
  def test_resolve_columns_defaults_to_all(self):
    cols = Browser().attach_db(_db()).resolve_columns()
    self.assertEqual(cols, _db().columns())

  def test_show_and_hide_columns(self):
    b = Browser().attach_db(_db()).show_columns(["name", "acc"]).hide_columns(["acc"])
    self.assertEqual(b.resolve_columns(), ["name"])

  def test_hide_columns_hides_whole_group_by_prefix(self):
    db = Database()
    db.entries = [
      ExperimentData(
        {
          "params": {"a": 1, "b": 2},
          "parameters": {"z": 9},  # must NOT be hit by "params"
          "acc": 0.5,
        }
      )
    ]
    b = Browser().attach_db(db).hide_columns(["params"])
    # "params//a" and "params//b" gone; "parameters//z" and "acc" remain.
    self.assertEqual(b.resolve_columns(), ["acc", "parameters//z"])
    # the group's leaves are still in the pool for the `c` picker
    self.assertIn("params//a", b.all_columns())

  def test_hide_group_then_reshow_leaf_via_picker_pool(self):
    db = Database()
    db.entries = [ExperimentData({"params": {"a": 1, "b": 2}})]
    b = Browser().attach_db(db).hide_columns(["params"])
    self.assertEqual(b.resolve_columns(), [])
    self.assertEqual(b.all_columns(), ["params//a", "params//b"])

  def test_hide_columns_from_pattern(self):
    db = Database()
    db.entries = [
      ExperimentData(
        {
          "metrics": {"acc": 0.5, "loss": 0.1},
          "name": "run",
        }
      )
    ]
    # Regex matches anywhere in the path, so "metrics//" hits both leaves.
    b = Browser().attach_db(db).hide_columns_from_pattern(r"^metrics//")
    self.assertEqual(b.resolve_columns(), ["name"])
    # matched columns stay in the pool for the `c` picker
    self.assertIn("metrics//acc", b.all_columns())

  def test_hide_columns_from_pattern_is_permutation_invariant(self):
    db = Database()
    db.entries = [ExperimentData({"metrics": {"acc": 0.5, "loss": 0.1}, "name": "run"})]
    # Pattern registered before vs. after attaching the db must agree, since the
    # match is computed lazily at resolve/run time.
    before = Browser().hide_columns_from_pattern(r"^metrics//").attach_db(db).resolve_columns()
    after = Browser().attach_db(db).hide_columns_from_pattern(r"^metrics//").resolve_columns()
    self.assertEqual(before, ["name"])
    self.assertEqual(after, ["name"])

  def test_hide_columns_from_pattern_accumulates(self):
    db = Database()
    db.entries = [ExperimentData({"a": 1, "b": 2, "c": 3})]
    b = (Browser().attach_db(db)
         .hide_columns_from_pattern(r"^a$")
         .hide_columns_from_pattern(r"^b$"))
    self.assertEqual(b.resolve_columns(), ["c"])

  def test_hide_columns_from_pattern_with_a_plain_word_is_group_prefix_not_substring(self):
    # A pattern with no regex metacharacters is auto-detected as a literal
    # group-prefix (see C.expand): "loss" hides the top-level "loss" leaf
    # itself, but leaves "metrics//loss" alone even though it also contains
    # "loss" -- unlike the old unconditional re.search, which hit both.
    db = Database()
    db.entries = [ExperimentData({"loss": 0.1, "metrics": {"loss": 0.2, "acc": 0.9}, "name": "run"})]
    b = Browser().attach_db(db).hide_columns_from_pattern("loss")
    self.assertEqual(sorted(b.resolve_columns()), ["metrics//acc", "metrics//loss", "name"])

  def test_hide_constant_columns(self):
    db = Database()
    db.entries = [
      ExperimentData({"dataset": "mnist", "lr": 0.1, "seed": 1}),
      ExperimentData({"dataset": "mnist", "lr": 0.2, "seed": 1}),
    ]
    # "dataset" and "seed" are constant; only "lr" varies.
    self.assertEqual(Browser().attach_db(db).hide_constant_columns().resolve_columns(), ["lr"])

  def test_hide_constant_columns_treats_missing_as_a_value(self):
    db = Database()
    db.entries = [
      ExperimentData({"a": 1, "b": 2}),
      ExperimentData({"a": 1}),  # "b" missing here -> b is NOT constant
    ]
    self.assertEqual(Browser().attach_db(db).hide_constant_columns().resolve_columns(), ["b"])

  def test_hide_constant_columns_noop_with_single_entry(self):
    db = Database()
    db.entries = [ExperimentData({"a": 1, "b": 2})]
    self.assertEqual(Browser().attach_db(db).hide_constant_columns().resolve_columns(), ["a", "b"])

  def test_hidden_columns_stay_in_pool_for_picker(self):
    db = Database()
    db.entries = [
      ExperimentData({"dataset": "mnist", "lr": 0.1}),
      ExperimentData({"dataset": "mnist", "lr": 0.2}),
    ]
    b = Browser().attach_db(db).hide_constant_columns()
    self.assertEqual(b.resolve_columns(), ["lr"])  # initially visible
    self.assertEqual(b.all_columns(), ["dataset", "lr"])  # still offered by `c`

  def test_hide_constant_columns_is_permutation_invariant(self):
    db = Database()
    db.entries = [
      ExperimentData({"dataset": "mnist", "lr": 0.1, "seed": 1}),
      ExperimentData({"dataset": "mnist", "lr": 0.2, "seed": 1}),
    ]
    # Toggle before vs. after attaching the db must give the same result, since
    # the constants are computed lazily at resolve/run time.
    before = Browser().hide_constant_columns().attach_db(db).resolve_columns()
    after = Browser().attach_db(db).hide_constant_columns().resolve_columns()
    self.assertEqual(before, ["lr"])
    self.assertEqual(after, ["lr"])

  def test_hide_constant_columns_reflects_entries_attached_after_toggle(self):
    # The flag is set on an empty browser; entries arrive only afterwards, yet
    # the constants are still computed correctly at resolve time.
    b = Browser().hide_constant_columns()
    db = Database()
    db.entries = [
      ExperimentData({"dataset": "mnist", "lr": 0.1}),
      ExperimentData({"dataset": "mnist", "lr": 0.2}),
    ]
    b.attach_db(db)
    self.assertEqual(b.resolve_columns(), ["lr"])

  def test_registration_methods_are_chainable(self):
    b = Browser().attach_db(_db())
    self.assertIs(b.truncate(10), b)
    self.assertIs(b.format("acc", str), b)

  def test_pipe_applies_presets_in_order_and_returns_self(self):
    calls = []

    def preset_a(br):
      calls.append("a")
      return br.hide_columns(["acc"])  # chain-and-return style

    def preset_b(br):
      calls.append("b")
      br.hide_columns(["name"])  # mutate-only style (no return)

    b = Browser().attach_db(_db())
    self.assertIs(b.pipe(preset_a, preset_b), b)
    self.assertEqual(calls, ["a", "b"])
    self.assertEqual(b.resolve_columns(), ["cfg//lr"])

  def test_explicit_bind_overrides_visualizer_on_same_key(self):
    seen = []
    viz = Visualizer("v", "viz", render=lambda e: Path("/tmp/x.png"))
    b = Browser().attach_db(_db()).add_visualizer(viz).bind("v", "custom", lambda s, c: seen.append(s))
    label, action = b.action_map()["v"]
    self.assertEqual(label, "custom")

  def test_visualizer_action_calls_show_bitmap(self):
    calls = []

    class Ctx:
      def notify(self, m): ...
      def show_bitmap(self, b, title="image"): calls.append((b, title))
      def show_image(self, p): ...
      def open_text(self, t): ...

    bitmap = object()
    viz = Visualizer("v", "viz", render=lambda e: bitmap)
    viz.as_action()(Selection(_db().entries[0]), Ctx())
    self.assertEqual(calls, [(bitmap, "viz")])

  def test_column_scoped_visualizer_only_fires_on_its_column(self):
    calls = []
    notes = []

    class Ctx:
      def notify(self, m): notes.append(m)
      def show_bitmap(self, b, title="image"): calls.append(b)
      def show_image(self, p): ...
      def open_text(self, t): ...

    bitmap = object()
    viz = Visualizer("v", "viz", render=lambda e: bitmap, column="acc")
    action = viz.as_action()
    entry = _db().entries[0]
    action(Selection(entry, column="name"), Ctx())  # wrong column
    self.assertEqual(calls, [])
    self.assertEqual(len(notes), 1)
    action(Selection(entry, column="acc"), Ctx())  # right column
    self.assertEqual(calls, [bitmap])

  def test_plotter_registers_under_its_key(self):
    plotter = Plotter("t", "trajectory", plot=lambda e: object())
    b = Browser().attach_db(_db()).add_plotter(plotter)
    self.assertIn("t", b.action_map())
    self.assertEqual(b.action_map()["t"][0], "trajectory")

  def test_plotter_action_calls_open_figure(self):
    calls = []

    class Ctx:
      def notify(self, m): ...
      def show_bitmap(self, b, title="image"): ...
      def show_image(self, p): ...
      def open_figure(self, f, title="plot"): calls.append((f, title))
      def open_text(self, t): ...

    figure = object()
    plotter = Plotter("t", "trajectory", plot=lambda e: figure)
    plotter.as_action()(Selection(_db().entries[0]), Ctx())
    self.assertEqual(calls, [(figure, "trajectory")])

  def test_column_scoped_plotter_only_fires_on_its_column(self):
    calls = []
    notes = []

    class Ctx:
      def notify(self, m): notes.append(m)
      def show_bitmap(self, b, title="image"): ...
      def show_image(self, p): ...
      def open_figure(self, f, title="plot"): calls.append(f)
      def open_text(self, t): ...

    figure = object()
    plotter = Plotter("t", "trajectory", plot=lambda e: figure, column="acc")
    action = plotter.as_action()
    entry = _db().entries[0]
    action(Selection(entry, column="name"), Ctx())  # wrong column
    self.assertEqual(calls, [])
    self.assertEqual(len(notes), 1)
    action(Selection(entry, column="acc"), Ctx())  # right column
    self.assertEqual(calls, [figure])

  def test_explicit_bind_overrides_plotter_on_same_key(self):
    plotter = Plotter("t", "trajectory", plot=lambda e: object())
    b = Browser().attach_db(_db()).add_plotter(plotter).bind("t", "custom", lambda s, c: None)
    self.assertEqual(b.action_map()["t"][0], "custom")


class PrettyTest(unittest.TestCase):
  def test_nested_mapping_renders_all_leaves(self):
    text = _pretty(_db().entries[0].data)
    self.assertIn("name:", text)
    self.assertIn("lr:", text)
    self.assertIn("0.1", text)


class AppPilotTest(unittest.TestCase):
  """Drive the real Textual app headlessly through a Pilot."""

  def _run(self, coro):
    asyncio.run(coro)

  def test_table_populates_and_sorts(self):
    async def scenario():
      app = _BrowserApp(Browser().attach_db(_db()).sort_default("acc", descending=True))
      async with app.run_test() as pilot:
        from textual.widgets import DataTable

        table = app.query_one(DataTable)
        self.assertEqual(table.row_count, 3)
        # sorted by acc desc -> first row is the 0.9 entry ("a")
        self.assertEqual(app._rows[0].data["name"], "a")
        # 'q' exits
        await pilot.press("q")

    self._run(scenario())

  def test_filter_reduces_rows(self):
    async def scenario():
      app = _BrowserApp(Browser().attach_db(_db()))
      async with app.run_test() as pilot:
        app._filter = "name//"  # no effect; ensure rebuild path is exercised
        app._filter = ""
        app._rebuild()
        self.assertEqual(len(app._rows), 3)
        app._filter = "0.9"
        app._rebuild()
        self.assertEqual(len(app._rows), 1)
        await pilot.press("q")

    self._run(scenario())

  def test_custom_action_receives_focused_entry(self):
    received = []

    async def scenario():
      browser = Browser().attach_db(_db()).bind("x", "grab", lambda s, c: received.append(s.focused))
      app = _BrowserApp(browser)
      async with app.run_test() as pilot:
        await pilot.press("x")
        await pilot.press("q")

    self._run(scenario())
    self.assertEqual(len(received), 1)
    self.assertIn("name", received[0].data)

  def test_solarized_light_theme_with_darker_text_is_default(self):
    from vildema.tui.browser import _TEXT_GRAY, _THEME_NAME

    async def scenario():
      app = _BrowserApp(Browser().attach_db(_db()))
      async with app.run_test() as pilot:
        self.assertEqual(app.theme, _THEME_NAME)
        self.assertEqual(app.current_theme.foreground, _TEXT_GRAY)
        await pilot.press("q")

    self._run(scenario())

  def test_expand_toggle_is_column_specific(self):
    long = "z" * 200
    db = Database()
    db.entries = [ExperimentData({"blob": long, "name": "a"})]

    async def scenario():
      app = _BrowserApp(Browser().attach_db(db).show_columns(["blob", "name"]).truncate(20))
      async with app.run_test() as pilot:
        # focused column starts at "blob"
        self.assertNotIn("blob", app._expanded)
        self.assertEqual(app._cell(db.entries[0], "blob"), "z" * 19 + "…")
        await pilot.press("e")
        self.assertIn("blob", app._expanded)
        self.assertEqual(app._cell(db.entries[0], "blob"), long)  # full, untruncated
        # other columns are unaffected
        self.assertNotIn("name", app._expanded)
        await pilot.press("e")  # toggles back off
        self.assertNotIn("blob", app._expanded)
        await pilot.press("q")

    self._run(scenario())

  def test_colorize_toggle_is_column_specific(self):
    from rich.text import Text

    db = Database()
    db.entries = [
      ExperimentData({"acc": 0.1, "name": "a"}),
      ExperimentData({"acc": 0.9, "name": "b"}),
    ]

    async def scenario():
      app = _BrowserApp(Browser().attach_db(db).show_columns(["acc", "name"]))
      async with app.run_test() as pilot:
        self.assertNotIn("acc", app._colorized)
        await pilot.press("u")  # focused column starts at "acc"
        self.assertIn("acc", app._colorized)
        # min maps to red, max to green; both rendered as styled Text.
        lo = app._display_cell(db.entries[0], app._display_cols[0])
        hi = app._display_cell(db.entries[1], app._display_cols[0])
        self.assertIsInstance(lo, Text)
        self.assertEqual(lo.style, "#c0392b")  # gradient low (red)
        self.assertEqual(hi.style, "#2e8b33")  # gradient high (green)
        # other columns are unaffected
        self.assertNotIn("name", app._colorized)
        await pilot.press("u")  # toggles back off
        self.assertNotIn("acc", app._colorized)
        await pilot.press("q")

    self._run(scenario())

  def test_colorize_columns_seeds_gradient_at_startup(self):
    from rich.text import Text

    db = Database()
    db.entries = [
      ExperimentData({"acc": 0.1, "name": "a"}),
      ExperimentData({"acc": 0.9, "name": "b"}),
    ]

    async def scenario():
      app = _BrowserApp(
        Browser().attach_db(db).show_columns(["acc", "name"]).colorize_columns(["acc"])
      )
      async with app.run_test() as pilot:
        # No key press: "acc" is colour-coded from construction.
        self.assertIn("acc", app._colorized)
        lo = app._display_cell(db.entries[0], app._display_cols[0])
        hi = app._display_cell(db.entries[1], app._display_cols[0])
        self.assertIsInstance(lo, Text)
        self.assertEqual(lo.style, "#c0392b")  # gradient low (red)
        self.assertEqual(hi.style, "#2e8b33")  # gradient high (green)
        self.assertNotIn("name", app._colorized)
        await pilot.press("q")

    self._run(scenario())

  def test_colorize_columns_group_prefix_covers_subcolumns(self):
    from rich.text import Text

    db = Database()
    db.entries = [
      ExperimentData({"metrics": {"acc": 0.1, "loss": 9.0}, "name": "a"}),
      ExperimentData({"metrics": {"acc": 0.9, "loss": 1.0}, "name": "b"}),
    ]

    async def scenario():
      # bare string, group prefix -> both metrics//* leaves are colour-coded,
      # each on its own min/max gradient.
      app = _BrowserApp(Browser().attach_db(db).colorize_columns("metrics"))
      async with app.run_test() as pilot:
        cols = {dc.key: i for i, dc in enumerate(app._display_cols)}
        acc_i, loss_i, name_i = cols["metrics//acc"], cols["metrics//loss"], cols["name"]

        # acc: row0 is the min (red), row1 the max (green)
        acc_lo = app._display_cell(db.entries[0], app._display_cols[acc_i])
        acc_hi = app._display_cell(db.entries[1], app._display_cols[acc_i])
        self.assertEqual(acc_lo.style, "#c0392b")
        self.assertEqual(acc_hi.style, "#2e8b33")
        # loss: row0 is the max (green), row1 the min (red) — its own scale
        loss_hi = app._display_cell(db.entries[0], app._display_cols[loss_i])
        loss_lo = app._display_cell(db.entries[1], app._display_cols[loss_i])
        self.assertEqual(loss_hi.style, "#2e8b33")
        self.assertEqual(loss_lo.style, "#c0392b")
        # a non-metrics column stays uncoloured
        self.assertNotIsInstance(app._display_cell(db.entries[0], app._display_cols[name_i]), Text)
        await pilot.press("q")

    self._run(scenario())

  def test_colorize_toggle_off_on_seeded_subcolumn(self):
    db = Database()
    db.entries = [
      ExperimentData({"metrics": {"acc": 0.1, "loss": 9.0}}),
      ExperimentData({"metrics": {"acc": 0.9, "loss": 1.0}}),
    ]

    async def scenario():
      # Seed the whole "metrics" group, then `u` on one subcolumn must toggle
      # just that leaf off (punch a hole), leaving its sibling coloured.
      app = _BrowserApp(Browser().attach_db(db).colorize_columns("metrics"))
      async with app.run_test() as pilot:
        self.assertEqual(app._display_cols[0].key, "metrics//acc")  # focused column
        self.assertIn("metrics//acc", app._colorized)
        self.assertIn("metrics//loss", app._colorized)
        await pilot.press("u")  # toggle the focused subcolumn off
        self.assertNotIn("metrics//acc", app._colorized)
        self.assertIn("metrics//loss", app._colorized)  # sibling unaffected
        await pilot.press("q")

    self._run(scenario())

  def test_column_tree_picker_toggles_leaf(self):
    from textual.widgets import Tree
    from vildema.tui.browser import _ColumnScreen

    async def scenario():
      app = _BrowserApp(Browser().attach_db(_db()))
      async with app.run_test() as pilot:
        await pilot.press("c")  # open tree picker
        await pilot.pause()
        self.assertIsInstance(app.screen, _ColumnScreen)
        tree = app.screen.query_one(Tree)
        tree.move_cursor(tree.root.children[0])  # first leaf
        await pilot.pause()
        await pilot.press("enter")  # toggle it off
        await pilot.press("escape")  # apply + close
        await pilot.pause()
        self.assertEqual(len(app._visible), len(_db().columns()) - 1)
        await pilot.press("q")

    self._run(scenario())

  def test_column_tree_space_also_toggles(self):
    from textual.widgets import Tree
    from vildema.tui.browser import _ColumnScreen

    async def scenario():
      app = _BrowserApp(Browser().attach_db(_db()))
      async with app.run_test() as pilot:
        await pilot.press("c")
        await pilot.pause()
        tree = app.screen.query_one(Tree)
        tree.move_cursor(tree.root.children[0])  # first leaf
        await pilot.pause()
        await pilot.press("space")  # toggle off via space (not just enter)
        await pilot.press("escape")
        await pilot.pause()
        self.assertEqual(len(app._visible), len(_db().columns()) - 1)
        await pilot.press("q")

    self._run(scenario())

  def test_column_tree_arrows_and_tab_fold_groups(self):
    from textual.widgets import Tree

    db = Database()
    db.entries = [ExperimentData({"params": {"a": 1, "b": 2}, "acc": 0.9})]

    async def scenario():
      app = _BrowserApp(Browser().attach_db(db))
      async with app.run_test() as pilot:
        await pilot.press("c")
        await pilot.pause()
        tree = app.screen.query_one(Tree)
        group = tree.root.children[1]  # the "params" group
        tree.move_cursor(group)
        await pilot.pause()
        self.assertTrue(group.is_expanded)
        await pilot.press("left")
        await pilot.pause()
        self.assertFalse(group.is_expanded)  # ← folds
        await pilot.press("right")
        await pilot.pause()
        self.assertTrue(group.is_expanded)  # → unfolds
        await pilot.press("tab")
        await pilot.pause()
        self.assertFalse(group.is_expanded)  # tab toggles
        await pilot.press("escape")
        await pilot.press("q")

    self._run(scenario())

  def test_help_lists_builtin_and_custom_keys(self):
    from textual.widgets import Static

    async def scenario():
      browser = Browser().attach_db(_db()).bind("d", "diff runs", lambda s, c: None)
      app = _BrowserApp(browser)
      async with app.run_test() as pilot:
        await pilot.press("question_mark")
        await pilot.pause()
        body = str(app.screen.query_one("#textbody", Static).render())
        self.assertIn("ctrl+p", body)  # palette mentioned
        self.assertIn("-- project keys --", body)
        self.assertIn("diff runs", body)  # custom bind present
        await pilot.press("q")

    self._run(scenario())

  def test_column_tree_groups_nested_paths(self):
    from textual.widgets import Tree

    db = Database()
    db.entries = [ExperimentData({"params": {"a": 1, "b": 2}, "acc": 0.9})]

    async def scenario():
      app = _BrowserApp(Browser().attach_db(db))
      async with app.run_test() as pilot:
        await pilot.press("c")
        await pilot.pause()
        labels = [str(n.label) for n in app.screen.query_one(Tree).root.children]
        # a single "params" group node (with a tally) instead of two flat leaves
        self.assertEqual(labels, ["[x] acc", "params  (2/2)"])
        await pilot.press("escape")
        await pilot.press("q")

    self._run(scenario())

  def test_command_palette_lists_shortcuts(self):
    async def scenario():
      browser = Browser().attach_db(_db()).bind("x", "My custom action", lambda s, c: None)
      app = _BrowserApp(browser)
      async with app.run_test() as pilot:
        titles = [title for title, _help, _cb in app.shortcut_commands()]
        self.assertIn("Sort by focused column", titles)
        self.assertIn("Collapse / expand all groups", titles)
        self.assertIn("Collapse focused column one level", titles)
        self.assertIn("Expand focused column one level", titles)
        self.assertIn("My custom action", titles)  # custom binds appear too
        await pilot.press("q")

    self._run(scenario())

  def test_sort_arrow_in_header_label(self):
    async def scenario():
      app = _BrowserApp(Browser().attach_db(_db()).sort_default("acc", descending=True))
      async with app.run_test() as pilot:
        self.assertEqual(app._header_label("acc"), "acc ▼")
        self.assertEqual(app._header_label("name"), "name")  # unsorted: no arrow
        await pilot.press("q")

    self._run(scenario())

  def test_focused_column_passed_to_action(self):
    seen = []

    async def scenario():
      browser = Browser().attach_db(_db()).show_columns(["name", "acc"])
      browser.bind("x", "grab", lambda s, c: seen.append(s.column))
      app = _BrowserApp(browser)
      async with app.run_test() as pilot:
        await pilot.press("x")  # cursor starts on first column "name"
        await pilot.press("q")

    self._run(scenario())
    self.assertEqual(seen, ["name"])

  def test_crosshair_has_three_distinct_tint_levels(self):
    from textual.widgets import DataTable

    db = Database()
    db.entries = [ExperimentData({"a": i, "b": i * 2, "c": i * 3}) for i in range(4)]

    async def scenario():
      app = _BrowserApp(Browser().attach_db(db))
      async with app.run_test(size=(80, 24)) as pilot:
        table = app.query_one(DataTable)
        table.move_cursor(row=1, column=1)
        await pilot.pause()

        def bg(r, c):
          coord = table.cursor_coordinate.__class__(r, c)
          style = table._get_row_style(r, table.rich_style)
          lines = table._render_cell(
            r, c, style, table.ordered_columns[c].get_render_width(table),
            cursor=table._should_highlight(table.cursor_coordinate, coord, "cell"),
          )
          return next((s.style.bgcolor for s in lines[0] if s.style and s.style.bgcolor), None)

        cursor, same_row, same_col, other = bg(1, 1), bg(1, 0), bg(0, 1), bg(0, 0)
        # the focused cell, its crosshair, and the rest are all different shades
        self.assertEqual(same_row, same_col)  # crosshair is uniform
        self.assertNotEqual(cursor, same_row)  # focused cell stands out from it
        self.assertNotEqual(same_row, other)  # crosshair stands out from the rest
        await pilot.press("q")

    self._run(scenario())

  def test_group_collapse_and_expand(self):
    from textual.widgets import DataTable

    db = Database()
    db.entries = [
      ExperimentData({"params": {"a": 1, "b": 2}, "acc": 0.9}),
      ExperimentData({"params": {"a": 3, "b": 4}, "acc": 0.5}),
    ]

    async def scenario():
      app = _BrowserApp(Browser().attach_db(db))
      async with app.run_test(size=(100, 24)) as pilot:
        table = app.query_one(DataTable)
        # columns sorted: acc, params//a, params//b -> focus a params leaf
        table.move_cursor(column=1)
        await pilot.pause()
        await pilot.press("g")  # collapse the params group
        await pilot.pause()
        headers = [str(c.label) for c in table.ordered_columns]
        self.assertEqual(headers, ["acc", "▸ params"])
        self.assertEqual(table.get_row_at(0)[1], "#2 cols")
        self.assertEqual(app._focused_column(), "params")
        await pilot.press("g")  # expand again
        await pilot.pause()
        headers = [str(c.label) for c in table.ordered_columns]
        self.assertEqual(headers, ["acc", "params//a", "params//b"])
        await pilot.press("q")

    self._run(scenario())

  def test_recursive_collapse_and_expand_one_level(self):
    from textual.widgets import DataTable

    db = Database()
    db.entries = [
      ExperimentData(
        {
          "cfg": 
            {
              "optim": {"lr": 0.1, "mom": 0.9},
              "data": {"bs": 32},
            }
          ,
          "acc": 0.9,
        }
      ),
    ]

    def headers(table):
      return [str(c.label) for c in table.ordered_columns]

    async def scenario():
      app = _BrowserApp(Browser().attach_db(db))
      async with app.run_test(size=(120, 24)) as pilot:
        table = app.query_one(DataTable)
        # sorted: acc, cfg//data//bs, cfg//optim//lr, cfg//optim//mom
        table.move_cursor(column=2)  # focus cfg//optim//lr
        await pilot.pause()
        await pilot.press("left_square_bracket")  # fold up to cfg//optim
        await pilot.pause()
        self.assertEqual(
          headers(table), ["acc", "cfg//data//bs", "▸ cfg//optim"]
        )
        self.assertEqual(app._focused_column(), "cfg//optim")
        await pilot.press("left_square_bracket")  # fold up again to cfg
        await pilot.pause()
        self.assertEqual(headers(table), ["acc", "▸ cfg"])
        self.assertEqual(table.get_row_at(0)[1], "#3 cols")
        await pilot.press("right_square_bracket")  # reveal exactly one level
        await pilot.pause()
        self.assertEqual(headers(table), ["acc", "▸ cfg//data", "▸ cfg//optim"])
        # drilling into a single-leaf child reveals its leaf
        table.move_cursor(column=1)  # focus ▸ cfg//data
        await pilot.pause()
        await pilot.press("right_square_bracket")
        await pilot.pause()
        self.assertEqual(
          headers(table), ["acc", "cfg//data//bs", "▸ cfg//optim"]
        )
        await pilot.press("q")

    self._run(scenario())

  def test_hiding_collapsed_group_removes_all_members(self):
    from textual.widgets import DataTable

    db = Database()
    db.entries = [
      ExperimentData({"params": {"a": 1, "b": 2}, "acc": 0.9}),
      ExperimentData({"params": {"a": 3, "b": 4}, "acc": 0.5}),
    ]

    async def scenario():
      app = _BrowserApp(Browser().attach_db(db))
      async with app.run_test(size=(100, 24)) as pilot:
        table = app.query_one(DataTable)
        table.move_cursor(column=1)
        await pilot.pause()
        await pilot.press("g")  # collapse params
        await pilot.pause()
        await pilot.press("h")  # hide the whole collapsed group
        await pilot.pause()
        self.assertEqual(app._visible, ["acc"])
        self.assertNotIn("params", app._collapsed)
        await pilot.press("q")

    self._run(scenario())

  def test_rows_have_number_labels(self):
    from textual.widgets import DataTable

    async def scenario():
      app = _BrowserApp(Browser().attach_db(_db()))
      async with app.run_test() as pilot:
        table = app.query_one(DataTable)
        labels = [row.label.plain for row in table.ordered_rows]
        self.assertEqual(labels, ["1", "2", "3"])
        await pilot.press("q")

    self._run(scenario())

  def test_visualizer_key_opens_and_closes_image_popup(self):
    import numpy as np
    from vildema.tui.browser import _ImageScreen

    img = np.random.rand(8, 8, 3)

    async def scenario():
      browser = Browser().attach_db(_db()).add_visualizer(Visualizer("v", "img", render=lambda e: img))
      app = _BrowserApp(browser)
      async with app.run_test() as pilot:
        await pilot.press("v")
        await pilot.pause()
        self.assertIsInstance(app.screen, _ImageScreen)
        await pilot.press("escape")
        await pilot.pause()
        self.assertNotIsInstance(app.screen, _ImageScreen)
        await pilot.press("q")

    self._run(scenario())

  def test_image_popup_zoom_controls(self):
    import numpy as np
    from vildema.tui.browser import _ImageScreen

    img = np.zeros((8, 8, 3), dtype=np.uint8)

    async def scenario():
      browser = Browser().attach_db(_db()).add_visualizer(Visualizer("v", "img", render=lambda e: img))
      app = _BrowserApp(browser)
      async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("v")
        await pilot.pause()
        screen = app.screen
        self.assertIsInstance(screen, _ImageScreen)
        fit = screen._zoom
        await pilot.press("plus")
        await pilot.pause()
        self.assertEqual(screen._zoom, fit + 1)  # zoom in
        await pilot.press("minus", "minus")
        await pilot.pause()
        self.assertEqual(screen._zoom, fit - 1)  # zoom out past fit
        await pilot.press("f")
        await pilot.pause()
        self.assertEqual(screen._zoom, fit)  # back to fit
        await pilot.press("q")

    self._run(scenario())


class ImageTest(unittest.TestCase):
  def test_grayscale_float_becomes_rgb_uint8(self):
    import numpy as np
    from vildema.tui.image import to_rgb_array

    out = to_rgb_array(np.linspace(0, 1, 16).reshape(4, 4))
    self.assertEqual(out.shape, (4, 4, 3))
    self.assertEqual(out.dtype, np.dtype("uint8"))
    self.assertEqual(out.max(), 255)

  def test_channel_first_tensor_layout_is_transposed(self):
    import numpy as np
    from vildema.tui.image import to_rgb_array

    self.assertEqual(to_rgb_array(np.zeros((3, 5, 7))).shape, (5, 7, 3))

  def test_rgba_drops_alpha(self):
    import numpy as np
    from vildema.tui.image import to_rgb_array

    self.assertEqual(to_rgb_array(np.zeros((6, 6, 4), dtype=np.uint8)).shape, (6, 6, 3))

  def test_render_uses_integer_downscaling_to_fit(self):
    import numpy as np
    from vildema.tui.image import render_halfblocks

    rendered = render_halfblocks(np.zeros((300, 300, 3), dtype=np.uint8), 80, 24)
    rows = rendered.text.plain.split("\n")
    self.assertLessEqual(len(rows[0]), 80)  # within column budget
    self.assertLessEqual(len(rows), 24)  # within row budget
    # square image stays square: width(px) == height(px) == 2 * text rows
    self.assertEqual(len(rows[0]), len(rows) * 2)

  def test_render_reports_scaling_metadata(self):
    import numpy as np
    from vildema.tui.image import render_halfblocks

    # 300x300 into an 80x24 budget -> integer downscale by 7.
    big = render_halfblocks(np.zeros((300, 300, 3), dtype=np.uint8), 80, 24)
    self.assertEqual((big.source_w, big.source_h), (300, 300))
    self.assertEqual(big.mode, "downscaled")
    self.assertEqual(big.factor, 7)
    self.assertEqual((big.display_w, big.display_h), (42, 42))
    self.assertIn("original 300×300px", big.describe())
    self.assertIn("downscaled 1/7", big.describe())

    # 8x8 into a roomy budget -> integer upscale.
    small = render_halfblocks(np.zeros((8, 8, 3), dtype=np.uint8), 80, 24)
    self.assertEqual(small.mode, "upscaled")
    self.assertEqual(small.display_w, 8 * small.factor)
    self.assertIn("upscaled", small.describe())

  def test_render_path_via_pil(self):
    import tempfile
    import numpy as np
    from PIL import Image
    from vildema.tui.image import render_halfblocks

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
      Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(f.name)
      rendered = render_halfblocks(Path(f.name), 80, 24)
    self.assertTrue(rendered.text.plain)


if __name__ == "__main__":
  unittest.main()
