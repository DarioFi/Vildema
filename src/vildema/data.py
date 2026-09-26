from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import List

from vildema.constraints import C, ImplementedC


# todo: the data validation is hard-coded but we should add more flexibility to it.
# the plan is to update it after seeing how it performs in real-world data loading

def _is_scalar(value: object) -> bool:
  """Whether `value` is an accepted leaf scalar (int/float/str/None)."""
  return isinstance(value, (int, float, str)) or value is None


def _validate(data: object) -> None:
  """Validate a loaded experiment: a dict of named columns whose leaves are
  scalars (int/float/str/None) or lists (of lists ...) of scalars.

  Raises `TypeError` on the first offending node. Nothing is copied or
  converted — this only checks the structure the caller passed in.
  """
  if not isinstance(data, dict):
    raise TypeError(
      f"ExperimentData must be a dict of named columns, got {type(data).__name__}."
    )
  for value in data.values():
    _validate_value(value)


def _validate_value(value: object) -> None:
  """Validate a column value: a nested dict (more named columns), a scalar leaf,
  or a list (the terminal layer) of scalars / nested lists."""
  if isinstance(value, dict):
    for v in value.values():
      _validate_value(v)
  elif isinstance(value, list):
    for item in value:
      _validate_leaf(item)
  elif not _is_scalar(value):
    raise TypeError(
      f"Unsupported value of type {type(value).__name__}: a column must be a "
      f"dict, a list, or an int/float/str/None."
    )


def _validate_leaf(item: object) -> None:
  """Validate a list element: a scalar or another list. A mapping here would be
  an unnamed, unreachable column, so it is rejected."""
  if isinstance(item, list):
    for sub in item:
      _validate_leaf(sub)
  elif not _is_scalar(item):
    raise TypeError(
      f"Lists may only hold scalars or nested lists, not {type(item).__name__}: "
      f"every column must be named."
    )


def _path_of(key: str) -> list[str]:
  """The list of nested keys for a `C.SEP`-joined subscription key."""
  return key.split(C.SEP)


_PLAIN_KEY = re.compile(r"^[\w.]+$")


def _validate_key_charset(data: object, path: str = "") -> None:
  """Raise `TypeError` if any dict key at any depth doesn't match `[\\w.]+`
  (letters/digits/underscore/dot). Independent of `_validate`/`_validate_value`
  (which check value *shape*, not key charset, and are currently disabled) --
  this only guards that a stored key can never be mistaken for a `C.expand`
  regex pattern."""
  if isinstance(data, Mapping):
    for k, v in data.items():
      if not _PLAIN_KEY.match(str(k)):
        here = f"{path}{C.SEP}{k}" if path else str(k)
        raise TypeError(
          f"Column key {k!r} at {here!r} has characters outside [\\w.]+ "
          f"(letters/digits/underscore/dot) -- rename it, or it may be "
          f"misread as a regex pattern by C.expand."
        )
      _validate_key_charset(v, f"{path}{C.SEP}{k}" if path else str(k))


def _get_path(obj: object, path: list[str]) -> object:
  """Walk `path` from `obj` level by level; `AttributeError` on a genuine miss."""
  for k in path:
    if isinstance(obj, Mapping):
      if k not in obj:
        raise AttributeError(f"Key {k} not in {obj}")
      obj = obj[k]
    else:
      obj = getattr(obj, k)
  return obj


def _set_path(root: dict, path: list[str], value: object) -> None:
  """Store `value` at `path` inside `root`, creating intermediate dicts."""
  _validate_value(value)
  *parents, leaf = path
  cursor = root
  for k in parents:
    if not _PLAIN_KEY.match(k):
      raise TypeError(
        f"Column key {k!r} has characters outside [\\w.]+ "
        f"(letters/digits/underscore/dot)."
      )
    child = cursor.get(k)
    if not isinstance(child, dict):
      child = {}
      cursor[k] = child
    cursor = child
  if not _PLAIN_KEY.match(leaf):
    raise TypeError(
      f"Column key {leaf!r} has characters outside [\\w.]+ "
      f"(letters/digits/underscore/dot)."
    )
  cursor[leaf] = value


def collect_paths(obj: object, prefix: tuple[str, ...], cols: set[str]) -> None:
  """Recurse into mappings, recording the `C.SEP`-joined path of every leaf."""
  if isinstance(obj, Mapping):
    for key, value in obj.items():
      collect_paths(value, prefix + (str(key),), cols)
  else:
    cols.add(C.SEP.join(prefix))


class Node:
  """A view over an intermediate (non-leaf) point in an `ExperimentData` tree:
  the raw nested dict at that path, by reference (mutating a `Node` mutates the
  `ExperimentData` it came from, exactly like `ExperimentData` never copies its
  own backing dict), plus enough context to support further relative
  `C`-style indexing.

  Returned wherever `ExperimentData.__getitem__` (or a `Database` access
  through it) lands on a dict instead of a scalar/list leaf -- the accessor
  this closes the gap for is code that used to bypass `C`/`__getitem__`
  entirely and reach into `entry.data.get("group", {})` by hand.
  """

  def __init__(self, data: Mapping, path: List[str]) -> None:
    self._data = data
    self._path = list(path)

  def __getitem__(self, key: str) -> object:
    """Relative access from this node -- a leaf value, or a further `Node`."""
    rel = _path_of(key)
    obj = _get_path(self._data, rel)
    return Node(obj, self._path + rel) if isinstance(obj, Mapping) else obj

  def __setitem__(self, key: str, value: object) -> None:
    """Relative write; requires the wrapped mapping to support item assignment."""
    _set_path(self._data, _path_of(key), value)

  def get(self, key: str, default: object = None) -> object:
    """`self[key]`, or `default` on a genuine miss -- the safe accessor that
    replaces `entry.data.get("group", {}).get("x")`-style bypasses."""
    try:
      return self[key]
    except (AttributeError, KeyError):
      return default

  def leaves(self) -> List[str]:
    """`C.SEP`-joined leaf paths *relative to this node*."""
    cols: set[str] = set()
    collect_paths(self._data, (), cols)
    return sorted(cols)

  def __eq__(self, other: object) -> bool:
    if isinstance(other, Node):
      return self._data == other._data
    if isinstance(other, Mapping):
      return self._data == other
    return NotImplemented

  def __repr__(self) -> str:
    return f"Node({self._data!r}, path={C.SEP.join(self._path)!r})"


class ExperimentData:
  """One experiment's data: a validated dict-of-dicts plus metadata.

  `data` is the very `dict` it was constructed from (not a copy): a mapping of
  named columns whose values are nested dicts, scalars (int/float/str/None), or
  lists (of lists ...) of scalars. Construction only *validates* this shape — it
  never rewrites it. Alongside it we keep a little metadata about where the entry
  came from: `file_origin`, `has_own_file` and a content `hash`.
  """

  def __init__(self, data: dict) -> None:
    # _validate(data)
    # todo: temporarily disabled the data validation. We need clearer specific and a type that automatically checks it when it is loaded
    _validate_key_charset(data)
    self.data: dict = data
    self.file_origin: Path | None = None
    self.has_own_file: bool = False

  def __eq__(self, other: object) -> bool:
    if isinstance(other, ExperimentData):
      return self.data == other.data
    return NotImplemented

  def __repr__(self) -> str:
    return f"ExperimentData({self.data!r})"

  def __getitem__(self, key: str) -> object:
    """Extract the value at `key` (a `C.SEP`-joined path), walking the nested
    structure level by level. A leaf resolves to its scalar/list value;
    landing on an intermediate group instead resolves to a `Node` view over
    it."""
    path = _path_of(key)
    obj = _get_path(self.data, path)
    return Node(obj, path) if isinstance(obj, Mapping) else obj

  def __setitem__(self, key: str, value: object) -> None:
    """Store `value` at the key-path of `key`, mutating this entry in place.

    Intermediate levels are created as plain nested dicts so the new column is
    reachable through `__getitem__`. `value` is validated (not copied) like any
    loaded data.
    """
    _set_path(self.data, _path_of(key), value)

  def get(self, key: str, default: object = None) -> object:
    """`self[key]`, or `default` on a genuine miss."""
    try:
      return self[key]
    except (AttributeError, KeyError):
      return default

  def apply_single_C(self, c: ImplementedC) -> bool:
    """
    Returns a boolean of whether the constraint C is satisfied by exp
    """
    try:
      obj = _get_path(self.data, c.filter_path)
    except AttributeError as e:
      if c.missing_value == ImplementedC.MISSING_SKIP:
        return False
      elif c.missing_value == ImplementedC.MISSING_FAIL:
        raise e
      else:
        raise NotImplementedError(f"Strategy {c.missing_value} not implemented")
    return c.test(obj, c.value)
