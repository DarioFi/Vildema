"""Unit tests for the ExperimentData container."""
import unittest

from vildema.constraints import C
from vildema.data import ExperimentData, Node


class ExperimentDataTest(unittest.TestCase):
  def test_data_access(self):
    exp = ExperimentData({"a": 1, "b": 2})
    self.assertEqual(exp.data, {"a": 1, "b": 2})
    self.assertEqual(exp["a"], 1)

  def test_nested_dicts_stored_as_given(self):
    # Validation only: nested dicts are kept exactly as passed in.
    exp = ExperimentData({"cfg": {"lr": 0.1}})
    self.assertEqual(exp.data, {"cfg": {"lr": 0.1}})
    self.assertIsInstance(exp.data["cfg"], dict)

  def test_does_not_copy_backing_dict(self):
    # Validation only: the entry wraps the very dict it was given (no copy).
    source = {"a": 1, "nested": {"b": 2}}
    exp = ExperimentData(source)
    self.assertIs(exp.data, source)

  def test_is_mutable(self):
    exp = ExperimentData({"a": 1})
    exp.data["a"] = 2
    exp.data["c"] = 3
    self.assertEqual(exp.data, {"a": 2, "c": 3})

  def test_metadata_defaults(self):
    exp = ExperimentData({"a": 1})
    self.assertIsNone(exp.file_origin)
    self.assertFalse(exp.has_own_file)

  def test_setitem_sets_in_place(self):
    exp = ExperimentData({"a": 1})
    exp["a"] = 2
    self.assertEqual(exp.data, {"a": 2})

  def test_setitem_creates_nested_dicts(self):
    exp = ExperimentData({"a": 1})
    exp[f"derived{C.SEP}score"] = 0.5
    self.assertEqual(exp.data, {"a": 1, "derived": {"score": 0.5}})
    self.assertEqual(exp[f"derived{C.SEP}score"], 0.5)

  def test_setitem_keeps_metadata(self):
    exp = ExperimentData({"a": 1})
    exp.has_own_file = True
    exp["b"] = 2
    self.assertTrue(exp.has_own_file)

  def test_setitem_validates_value(self):
    # A mapping inside a list is rejected, just like at construction.
    exp = ExperimentData({"a": 1})
    with self.assertRaises(TypeError):
      exp["bad"] = [{"x": 1}]

  def test_equality_compares_data(self):
    self.assertEqual(ExperimentData({"a": 1, "b": 2}),
                     ExperimentData({"b": 2, "a": 1}))
    self.assertNotEqual(ExperimentData({"a": 1}), ExperimentData({"a": 2}))
    # No longer a mapping, so not equal to a plain dict.
    self.assertNotEqual(ExperimentData({"a": 1}), {"a": 1})

  def test_lists_of_scalars_are_allowed_as_leaves(self):
    exp = ExperimentData({"losses": [1, 2, 3], "grid": [[1, 2], [3, 4]]})
    self.assertEqual(exp.data, {"losses": [1, 2, 3], "grid": [[1, 2], [3, 4]]})

  def test_mappings_inside_lists_are_accepted_at_construction(self):
    # Construction-time validation is currently disabled (see the comment on
    # `ExperimentData.__init__`), so shapes that `__setitem__` would reject are
    # still stored as given.
    for bad in ({"a": [{"b": 1}]}, {"a": [[{"b": 1}]]}, {"a": ({"b": 1},)}):
      exp = ExperimentData(bad)
      self.assertEqual(exp.data, bad)

  def test_non_mapping_top_level_is_accepted_at_construction(self):
    # Same as above: construction no longer validates the top-level shape.
    exp = ExperimentData([1, 2, 3])  # type: ignore[arg-type]
    self.assertEqual(exp.data, [1, 2, 3])

  def test_is_unhashable(self):
    exp = ExperimentData({"a": 1})
    with self.assertRaises(TypeError):
      hash(exp)

  def test_repr_roundtrips_via_eval(self):
    exp = ExperimentData({"a": 1})
    self.assertEqual(eval(repr(exp)), exp)

  def test_empty(self):
    exp = ExperimentData({})
    self.assertEqual(exp.data, {})


class NodeAccessTest(unittest.TestCase):
  def test_group_access_returns_node(self):
    exp = ExperimentData({"cfg": {"lr": 0.1}})
    self.assertIsInstance(exp["cfg"], Node)

  def test_node_relative_leaf_access(self):
    exp = ExperimentData({"cfg": {"lr": 0.1}})
    self.assertEqual(exp["cfg"]["lr"], 0.1)

  def test_node_relative_group_access_returns_nested_node(self):
    exp = ExperimentData({"a": {"b": {"c": 1}}})
    self.assertIsInstance(exp["a"]["b"], Node)
    self.assertEqual(exp["a"]["b"]["c"], 1)

  def test_node_get_returns_default_on_missing_key(self):
    exp = ExperimentData({"cfg": {"lr": 0.1}})
    self.assertIsNone(exp["cfg"].get("missing"))
    self.assertEqual(exp["cfg"].get("missing", "fallback"), "fallback")
    self.assertEqual(exp["cfg"].get("lr"), 0.1)

  def test_node_leaves_lists_relative_leaf_paths(self):
    exp = ExperimentData({"cfg": {"lr": 0.1, "opt": {"eps": 1e-8}}})
    self.assertEqual(exp["cfg"].leaves(), sorted(["lr", f"opt{C.SEP}eps"]))

  def test_node_equality_with_plain_dict(self):
    exp = ExperimentData({"cfg": {"lr": 0.1}})
    self.assertEqual(exp["cfg"], {"lr": 0.1})

  def test_node_setitem_mutates_backing_experiment_data(self):
    exp = ExperimentData({"cfg": {"lr": 0.1}})
    exp["cfg"]["eps"] = 1e-8
    self.assertEqual(exp.data, {"cfg": {"lr": 0.1, "eps": 1e-8}})

  def test_experiment_data_get_returns_default_on_missing_key(self):
    exp = ExperimentData({"acc": 0.9})
    self.assertIsNone(exp.get("missing"))
    self.assertEqual(exp.get("missing", "fallback"), "fallback")
    self.assertEqual(exp.get("acc"), 0.9)


class KeyCharsetValidationTest(unittest.TestCase):
  def test_rejects_bad_key_at_construction(self):
    with self.assertRaises(TypeError):
      ExperimentData({"a-b": 1})

  def test_rejects_bad_nested_key_at_construction(self):
    with self.assertRaises(TypeError):
      ExperimentData({"cfg": {"a-b": 1}})

  def test_accepts_dotted_keys(self):
    exp = ExperimentData({"eval_mmd_sq_t0.05": 1})
    self.assertEqual(exp["eval_mmd_sq_t0.05"], 1)

  def test_rejects_bad_key_on_setitem(self):
    exp = ExperimentData({"a": 1})
    with self.assertRaises(TypeError):
      exp["bad-name"] = 2


if __name__ == "__main__":
  unittest.main()
