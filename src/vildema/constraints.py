from __future__ import annotations

import re
from typing import Any, Callable, List, Sequence
from collections.abc import Container

_PLAIN_SEGMENT = re.compile(r"^[\w.]+$")


class C:
  """
  Constraint class
  Usage: C("filter_path") == value returns a constraint object ImplementedC that can later be used to filter data
  Used for writing filter queries in a concise simple way, and for pattern-matching a
  group/regex against a column universe via `expand`. Not an indexing type: `ExperimentData`,
  `Node` and `Database` are all subscripted with plain `C.SEP`-joined strings.
  """
  SEP = r"//"

  MISSING_SKIP = 'skip'
  MISSING_FAIL = 'fail'

  def __init__(self, name: str, missing_value: str = MISSING_FAIL):
    self.name = name
    self.filter_path = name.split(C.SEP)
    self.missing_value = missing_value

  def __eq__(self, other: object) -> ImplementedC:  # type: ignore[override]
    return ImplementedC(self.filter_path, other, lambda x, y: x == y, self.name, self.missing_value)

  def __ne__(self, other: object) -> ImplementedC:  # type: ignore[override]
    return ImplementedC(self.filter_path, other, lambda x, y: x != y, self.name, self.missing_value)

  def __lt__(self, other: object) -> ImplementedC:
    return ImplementedC(self.filter_path, other, lambda x, y: x < y, self.name, self.missing_value)

  def __le__(self, other: object) -> ImplementedC:
    return ImplementedC(self.filter_path, other, lambda x, y: x <= y, self.name, self.missing_value)

  def __gt__(self, other: object) -> ImplementedC:
    return ImplementedC(self.filter_path, other, lambda x, y: x > y, self.name, self.missing_value)

  def __ge__(self, other: object) -> ImplementedC:
    return ImplementedC(self.filter_path, other, lambda x, y: x >= y, self.name, self.missing_value)

  def is_in(self, other: Container[object]) -> ImplementedC:
    return ImplementedC(self.filter_path, other, lambda x, y: x in y, self.name, self.missing_value)

  def expand(self, columns: Sequence[str]) -> List["C"]:
    """
    Resolve this constraint's `name` against a known column universe (e.g. `Database.columns()`).

    Auto-detected mode: if every `C.SEP`-delimited segment of `name` matches `[\\w.]+`
    (letters/digits/underscore/dot), `name` is treated as a literal group-prefix path -- a column
    matches if it equals `name` exactly or starts with `name + C.SEP` (an exact leaf name is just
    the degenerate one-column case of this). Otherwise `name` is treated as a regex and matched with
    `re.search` anywhere in each column.

    Returns one concrete leaf `C` per match, in `columns` order, each carrying `self.missing_value`
    forward. Never raises: an unknown group or non-matching pattern just yields `[]`.
    """
    segments = self.name.split(C.SEP)
    if all(_PLAIN_SEGMENT.match(s) for s in segments if s):
      prefix = self.name
      matches: Callable[[str, ...], bool] = lambda col, *args: col == prefix or col.startswith(prefix + C.SEP)
    else:
      matches: Callable[[str, ...], bool] = re.compile(self.name).search
    return [C(col, self.missing_value) for col in columns if matches(col)]


class ImplementedC(C):
  """
  Auxiliary class that represents an implemented constraint, built from a constraint C
  """

  def __init__(self, filter_path: List[str], value: object, test: Callable[[Any, Any], bool], name: str,
               missing_value: str) -> None:
    super().__init__(name, missing_value)
    self.filter_path = filter_path
    self.value = value
    self.test = test
