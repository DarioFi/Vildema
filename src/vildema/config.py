"""Configuration for vildema.

The config holds the *stable* pre-run settings (how to parse, how to handle
failures) that are set once for a codebase and rarely change. Register them once
with `set_default_config(...)` and every `Database()` picks them up when no
explicit config is passed to the constructor.

The *paths* (which folders/files to read) are deliberately not part of the
config: they change every call and are passed straight to
`Database().load_from_disk(paths=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Tuple

from .loaders import SerializerSpec

OnFailedParse = Literal["ignore", "warn", "raise"]

# A single (glob pattern, serializer) pair, e.g. ("runs/**/*.json", "json").
PathPattern = Tuple[str, SerializerSpec]


@dataclass(frozen=True)
class VildemaConfig:
  """Immutable configuration for a `Database`.

  Frozen so it can be safely shared as a module-level default without one call
  site mutating another's settings. Build modified copies with `with_overrides`.
  """

  # What to do when a file fails to parse.
  on_failed_parse: OnFailedParse = "raise"

  def with_overrides(self, **overrides: Any) -> "VildemaConfig":
    """Return a copy with the given fields replaced. Field names are validated."""
    return replace(self, **overrides)

  valid_leaf_type: Tuple[type, ...] = (int, float, str, type(None))  # types allowed in leaf values

# Library defaults; replaced wholesale by the user's stable settings.
_default_config: VildemaConfig = VildemaConfig()


def set_default_config(config: VildemaConfig) -> None:
  """Set the config used by `Database()` when none is passed to the constructor.

  Call this once where the package is imported in your codebase.
  """
  global _default_config
  _default_config = config


def get_default_config() -> VildemaConfig:
  """Return the current default config."""
  return _default_config


def resolve_config(config: VildemaConfig | None) -> VildemaConfig:
  """Resolve the effective config: the explicit `config` if given, otherwise the
  registered default."""
  return config if config is not None else get_default_config()
