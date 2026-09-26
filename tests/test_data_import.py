"""Integration tests for importing data files end-to-end via load_pattern."""
import json
import tempfile
import unittest
from pathlib import Path

from vildema.data import ExperimentData
from vildema.loaders import load_pattern


class LoadPatternIntegrationTest(unittest.TestCase):
  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.root = Path(self._tmp.name)

  def tearDown(self):
    self._tmp.cleanup()

  def _write_json(self, relative, obj):
    path = self.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path

  def test_imports_nested_json_with_string_serializer(self):
    self._write_json("run1/result.json", {"acc": 1})
    self._write_json("run2/result.json", {"acc": 2})
    self._write_json("run2/nested/result.json", {"acc": 3})

    loaded = dict(load_pattern(str(self.root / "**" / "*.json"), "json"))

    self.assertEqual(len(loaded), 3)
    # Each file holds a single experiment, so each value is a one-element list.
    accs = [entry for entries in loaded.values() for entry in entries]
    self.assertCountEqual(accs, [ExperimentData({"acc": 1}),
                                 ExperimentData({"acc": 2}),
                                 ExperimentData({"acc": 3})])
    # Keys are real Paths pointing at existing files.
    self.assertTrue(all(p.is_file() for p in loaded))

  def test_only_matching_extension_is_imported(self):
    self._write_json("keep.json", {"k": 1})
    (self.root / "skip.yaml").write_text("k: 1\n")
    (self.root / "skip.txt").write_text("nope")

    loaded = list(load_pattern(str(self.root / "*.json"), "json"))

    self.assertEqual(len(loaded), 1)
    self.assertEqual(loaded[0][0].name, "keep.json")

  def test_custom_callable_serializer(self):
    (self.root / "a.log").write_text("hello")
    (self.root / "b.log").write_text("world")

    loaded = dict(load_pattern(str(self.root / "*.log"),
                               lambda path: path.read_text().upper()))

    self.assertEqual(set(loaded.values()), {"HELLO", "WORLD"})

  def test_empty_pattern_yields_nothing(self):
    self.assertEqual(list(load_pattern(str(self.root / "*.json"), "json")), [])

  def test_unknown_serializer_name_raises(self):
    self._write_json("x.json", {"a": 1})
    with self.assertRaises(ValueError):
      list(load_pattern(str(self.root / "*.json"), "bogus"))


if __name__ == "__main__":
  unittest.main()
