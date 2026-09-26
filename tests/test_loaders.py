"""Unit tests for the individual pieces of vildema.loaders."""
import json
import tempfile
import unittest
from pathlib import Path

from vildema import loaders
from vildema.data import ExperimentData
from vildema.loaders import (
  DEFAULT_SERIALIZERS,
  iter_matching_files,
  load_json,
  load_yaml,
  resolve_serializer,
)


class ResolveSerializerTest(unittest.TestCase):
  def test_known_names(self):
    self.assertIs(resolve_serializer("json"), load_json)
    self.assertIs(resolve_serializer("yaml"), load_yaml)
    self.assertIs(resolve_serializer("yml"), load_yaml)

  def test_callable_passthrough(self):
    custom = lambda path: "data"
    self.assertIs(resolve_serializer(custom), custom)

  def test_unknown_name_raises_value_error(self):
    with self.assertRaises(ValueError) as ctx:
      resolve_serializer("nope")
    # The error should be actionable: name the offender and list the options.
    self.assertIn("nope", str(ctx.exception))
    self.assertIn("json", str(ctx.exception))

  def test_wrong_type_raises_type_error(self):
    with self.assertRaises(TypeError):
      resolve_serializer(123)


class LoadJsonTest(unittest.TestCase):
  def test_loads_into_nested_dicts(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "x.json"
      path.write_text(json.dumps({"a": 1, "b": [1, 2]}))
      data = load_json(path)
      self.assertEqual(data, [ExperimentData({"a": 1, "b": [1, 2]})])
      self.assertEqual(data[0].data, {"a": 1, "b": [1, 2]})


class LoadYamlTest(unittest.TestCase):
  def test_yaml_behaviour(self):
    try:
      import yaml  # noqa: F401
    except ImportError:
      # Without PyYAML, load_yaml must fail loudly with a helpful message.
      with self.assertRaises(ImportError) as ctx:
        load_yaml(Path("whatever.yaml"))
      self.assertIn("PyYAML", str(ctx.exception))
      return
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "x.yaml"
      path.write_text("a: 1\nb:\n  - 1\n  - 2\n")
      data = load_yaml(path)
      self.assertEqual(data, [ExperimentData({"a": 1, "b": [1, 2]})])


class IterMatchingFilesTest(unittest.TestCase):
  def test_recursive_sorted_and_skips_dirs(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      (root / "sub").mkdir()
      (root / "b.json").write_text("{}")
      (root / "a.json").write_text("{}")
      (root / "sub" / "c.json").write_text("{}")
      (root / "sub" / "ignored.txt").write_text("nope")

      matches = list(iter_matching_files(str(root / "**" / "*.json")))

      self.assertEqual(matches, sorted(matches))  # deterministic ordering
      names = {p.name for p in matches}
      self.assertEqual(names, {"a.json", "b.json", "c.json"})
      self.assertTrue(all(p.is_file() for p in matches))

  def test_no_match_returns_empty(self):
    with tempfile.TemporaryDirectory() as tmp:
      self.assertEqual(list(iter_matching_files(str(Path(tmp) / "*.json"))), [])


class RegistryExtensibilityTest(unittest.TestCase):
  def test_can_register_new_serializer(self):
    original = dict(DEFAULT_SERIALIZERS)
    try:
      DEFAULT_SERIALIZERS["txt"] = lambda path: Path(path).read_text()
      self.assertIs(resolve_serializer("txt"), DEFAULT_SERIALIZERS["txt"])
    finally:
      DEFAULT_SERIALIZERS.clear()
      DEFAULT_SERIALIZERS.update(original)


if __name__ == "__main__":
  unittest.main()
