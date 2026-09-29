from __future__ import annotations

import copy
from typing import Callable, Iterable, List, Sequence

from .config import PathPattern, VildemaConfig, resolve_config
from .constraints import C, ImplementedC
from .data import ExperimentData, collect_paths
from .loaders import load_pattern

__all__ = ["Database", "ExperimentData", "Query"]


class Database:
  """
  Database object collecting the data read from disk. All the data is loaded in an internal list. The db exposes apis
  to extract specific columns and make edits. In particular, it is possible to apply queries.

  The design is meant to work with data with wildly different attributes, so methods like `columns` iterate through the list
  and perform a set union.
  """

  def __init__(self, config: VildemaConfig | None = None) -> None:
    self.entries: List[ExperimentData] = []
    self.config: VildemaConfig = resolve_config(config)
    self._column_view: List[str] | None = None

  def load_from_disk(self, paths: Sequence[PathPattern]) -> None:
    """
    Load entries from disk and append them to the internal list.
    It handles errors in the parsing according to the config passed when instantiating the database.

    :param paths: Sequence of PathPattern paths to load.
    """
    for pattern, serializer in paths:
      for file_path, data in load_pattern(pattern, serializer, on_failed_parse=self.config.on_failed_parse):
        has_own_file = len(data) == 1
        for d in data:
          d.file_origin = file_path
          d.has_own_file = has_own_file
        self.entries.extend(data)

  def __getitem__(self, item: str) -> List[object]:
    """
    Access experiment data by key (a `C.SEP`-joined path). Returns one value
    per entry, in entry order: a scalar/list if `item` is a leaf, or a `Node`
    (see `vildema.data.Node`) if it lands on an intermediate group -- walking
    each entry's own tree exactly, with no reference to `Database.columns()`.
    For regex matching, or to project onto a whole group/pattern's columns
    across entries, use `Database.select(...)` instead.
    """
    return [entry[item] for entry in self.entries]

  def select(self, *patterns: str) -> "Database":
    """
    Column-scoped view onto this Database: a new `Database` sharing this
    one's entries by reference (not copied), whose `columns()` reports only
    the leaves matched by `patterns` (group-prefix or regex, via `C.expand`,
    resolved against this database's current column pool).

    ``{c: view[c] for c in view.columns()}`` feeds straight into e.g.
    ``pandas.DataFrame(...)``.
    """
    pool = self.columns()
    matched = sorted({m.name for p in patterns for m in C(p).expand(pool)})
    view = Database(self.config)
    view.entries = self.entries
    view._column_view = matched
    return view

  def columns(self) -> List[str]:
    """
    The sorted list of leaf paths, each joined by ``C.SEP`` so it can be fed
    back into `C` / `__getitem__`. Explores recursively through all the
    entries in the database, unless this `Database` is a `select(...)` view,
    in which case the view's own column set is returned directly.
    """
    if self._column_view is not None:
      return list(self._column_view)
    cols: set[str] = set()
    for entry in self.entries:
      collect_paths(entry.data, (), cols)
    return sorted(cols)

  def add_column(self, name: str, func: Callable[[ExperimentData], object], show_progress=False, multi_process=False) -> None:
    """
    Adds a new column to the database by applying a function to each entry.

    `func` takes an `ExperimentData` and returns the value for the new column.
    `name` is a `C.SEP`-joined path, so nested columns (e.g.
    ``f"derived{C.SEP}score"``) are created and become accessible through
    `__getitem__`, exactly like loaded data. Entries are mutated in place
    via `ExperimentData.__setitem__`.
    """
    if multi_process:
      from concurrent.futures import ProcessPoolExecutor
      from multiprocessing import get_context

      # Python 3.14 defaults to forkserver on Linux, which requires the caller
      # to use an `if __name__ == "__main__"` guard. Explicit fork preserves
      # support for existing unguarded scripts.
      with ProcessPoolExecutor(mp_context=get_context("fork")) as executor:
        values: Iterable[object] = executor.map(func, self.entries, chunksize=1)

        if show_progress:
          from tqdm import tqdm
          values = tqdm(values, total=len(self.entries), desc=name)

        for entry, value in zip(self.entries, values):
          entry[name] = value
    else:
      entries: Iterable[ExperimentData] = self.entries
      if show_progress:
        from tqdm import tqdm
        entries = tqdm(entries, desc=name)

      for entry in entries:
        entry[name] = func(entry)


class Query:
  """
  Builder class that joins multiple constraints that can then be applied to a database.
  Designed with a Builder Pattern.
  Inspired by the Django ORM

  The query is applied in the following order:
    - standard filters added through `add_filters`
    - callback filters added through `add_callbacks`
  following the order in which both were added to the lists.

  Shortcircuiting is applied, so for queries on fields that might not exist it is necessary to either test for existance first
  or to write error-safe callbacks.
  """

  def __init__(self) -> None:
    self._filters: List[ImplementedC] = []
    self._callbacks: List[Callable[[ExperimentData], bool]] = []

  def add_filter(self, filter: ImplementedC) -> Query:
    self._filters.append(filter)
    return self

  def add_callback(self, callback: Callable[[ExperimentData], bool]) -> Query:
    """
    General filter that applies the callable to all experiments and keeps only the ones with a true value
    """
    self._callbacks.append(callback)
    return self

  def clone(self) -> Query:
    return copy.deepcopy(self)

  def apply(self, db: Database) -> Database:
    """
    Applies to a database and returns a new Database
    Warning: experiments are passed by reference so modifying them might create issues
    """
    results = Database()
    for exp in db.entries:
      if all((filter_func(exp) for filter_func in self._callbacks)):
        if all((exp.apply_single_C(c) for c in self._filters)):
          results.entries.append(exp)
    return results
