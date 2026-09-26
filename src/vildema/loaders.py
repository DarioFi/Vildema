from __future__ import annotations

import glob
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterator, Tuple, List, Literal, TYPE_CHECKING

from .data import ExperimentData

if TYPE_CHECKING:
  # Imported only for type hints; a runtime import would create a cycle because
  # `config` imports `SerializerSpec` from this module.
  from .config import OnFailedParse

# A Serializer reads a single file and returns the parsed data.
Serializer = Callable[[Path], list[ExperimentData]]
# Either the name of a registered default serializer, or a custom Serializer.
SerializerSpec = str | Serializer


# todo: once we have a good tree representation we also need to rework the parsing and perhaps move everything
#  inside the class?
#  We can do that now, but what is the best practice?

def _to_entry(obj: object) -> ExperimentData:
  """Wrap a single top-level entry, requiring it to be a mapping.

  Scalars or lists at the top level are not queriable by key-path, so we reject
  them here rather than silently storing something `exp[c]` can't traverse.
  The nested structure is kept as plain dicts/lists inside `ExperimentData`.
  """
  if not isinstance(obj, Dict):
    raise TypeError(
      f"Top-level experiment entries must be mappings, got {type(obj).__name__}"
    )
  return ExperimentData(obj)


def from_serialized_to_experiment_data(serialized_data: List[object] | Dict[object, object]) -> List[ExperimentData]:
  """
  The main problem/feature is that serialized data can be a list if multiple experiments are stacked in the same file. We do not accept top level lists as they are not really queriable.
  We assume the user is compliant and ill-infected data will return random results.
  """
  if isinstance(serialized_data, list):
    return [_to_entry(item) for item in serialized_data]
  elif isinstance(serialized_data, dict):
    return [_to_entry(serialized_data)]
  else:
    raise TypeError(f"Object {serialized_data} is weird")


def load_json(path: Path) -> List["ExperimentData"]:
  with open(path, "r") as f:
    return from_serialized_to_experiment_data(json.load(f))


def load_yaml(path: Path) -> List["ExperimentData"]:
  # yaml is not in the standard library; import lazily so it is only required
  # when a yaml file is actually loaded.
  try:
    import yaml
  except ImportError as e:
    raise ImportError(
      "Loading yaml files requires PyYAML. Install it with `pip install pyyaml`."
    ) from e
  with open(path, "r") as f:
    return from_serialized_to_experiment_data(yaml.safe_load(f))


# Registry of built-in serializers, addressable by a short string name.
# Extend it by assigning into this dict, e.g. DEFAULT_SERIALIZERS["csv"] = load_csv
DEFAULT_SERIALIZERS: Dict[str, Serializer] = {
  "json": load_json,
  "yaml": load_yaml,
  "yml": load_yaml,
}


def resolve_serializer(serializer: "str | Serializer") -> Serializer:
  """
  Turns a serializer spec into a callable.
  A string is looked up in DEFAULT_SERIALIZERS; a callable is returned as-is.
  """
  if callable(serializer):
    return serializer
  if isinstance(serializer, str):
    try:
      return DEFAULT_SERIALIZERS[serializer]
    except KeyError:
      raise ValueError(
        f"Unknown serializer {serializer!r}. "
        f"Available: {sorted(DEFAULT_SERIALIZERS)}, or pass a callable."
      )
  raise TypeError(f"Serializer must be a string or callable, got {type(serializer).__name__}.")


def iter_matching_files(pattern: str) -> Iterator[Path]:
  """
  Yields every file matching a glob pattern (e.g. `runs/**/*.yaml`).
  Supports recursive `**` matching and skips directories.
  """
  for match in sorted(glob.glob(pattern, recursive=True)):
    path = Path(match)
    if path.is_file():
      yield path


def load_pattern(pattern: str, serializer: "str | Serializer", on_failed_parse: OnFailedParse = "raise") -> Iterator[
  Tuple[Path, Any]]:
  """
  Loads every file matching `pattern` using `serializer`, yielding (path, data) pairs, failing according to the pattern strategy.
  """

  fn = resolve_serializer(serializer)
  for path in iter_matching_files(pattern):
    match on_failed_parse:

      case "ignore" | "warn":
        try:
          yield path, fn(path)
        except Exception as e:
          if on_failed_parse == "warn":
            print(f"Warning: failed to parse file {path}\n{e}")

      case "raise":
        yield path, fn(path)

      case _:
        raise AttributeError("Unknown failed parse strategy")
