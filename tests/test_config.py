"""Unit tests for vildema.config (VildemaConfig and resolution)."""
import unittest

from vildema import config
from vildema.config import (
  VildemaConfig,
  get_default_config,
  resolve_config,
  set_default_config,
)


class VildemaConfigTest(unittest.TestCase):
  def test_defaults(self):
    cfg = VildemaConfig()
    self.assertEqual(cfg.on_failed_parse, "raise")

  def test_is_frozen(self):
    cfg = VildemaConfig()
    with self.assertRaises(Exception):
      cfg.on_failed_parse = "raise"  # type: ignore[misc]

  def test_with_overrides_returns_copy(self):
    cfg = VildemaConfig()
    new = cfg.with_overrides(on_failed_parse="ignore")
    self.assertEqual(new.on_failed_parse, "ignore")
    # Original is untouched.
    self.assertEqual(cfg.on_failed_parse, "raise")

  def test_with_overrides_validates_field_names(self):
    with self.assertRaises(TypeError):
      VildemaConfig().with_overrides(bogus_field=1)


class DefaultConfigTest(unittest.TestCase):
  def setUp(self):
    self._saved = get_default_config()

  def tearDown(self):
    set_default_config(self._saved)

  def test_set_and_get_roundtrip(self):
    cfg = VildemaConfig(on_failed_parse="raise")
    set_default_config(cfg)
    self.assertIs(get_default_config(), cfg)

  def test_module_default_is_used_by_resolve(self):
    cfg = VildemaConfig(on_failed_parse="warn")
    set_default_config(cfg)
    self.assertIs(resolve_config(None), cfg)


class ResolveConfigTest(unittest.TestCase):
  def setUp(self):
    self._saved = get_default_config()
    set_default_config(VildemaConfig())

  def tearDown(self):
    set_default_config(self._saved)

  def test_none_falls_back_to_default(self):
    self.assertEqual(resolve_config(None), get_default_config())

  def test_explicit_config_takes_precedence_over_default(self):
    explicit = VildemaConfig(on_failed_parse="warn")
    self.assertIs(resolve_config(explicit), explicit)


if __name__ == "__main__":
  unittest.main()
