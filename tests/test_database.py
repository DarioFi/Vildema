"""Unit tests for the Database, constraint extraction, and Query engine."""
import json
import tempfile
import unittest
from pathlib import Path

from vildema.config import VildemaConfig
from vildema.constraints import C, ImplementedC
from vildema.data import ExperimentData, Node
from vildema.database import Database, Query


class GetItemTest(unittest.TestCase):
  def test_top_level_key(self):
    exp = ExperimentData({"acc": 0.9})
    self.assertEqual(exp["acc"], 0.9)

  def test_nested_key_path(self):
    exp = ExperimentData({"cfg": {"lr": 0.1}})
    self.assertEqual(exp[f"cfg{C.SEP}lr"], 0.1)

  def test_missing_key_raises_attribute_error(self):
    exp = ExperimentData({"acc": 0.9})
    with self.assertRaises(AttributeError):
      exp["nope"]

  def test_missing_nested_key_raises_attribute_error(self):
    exp = ExperimentData({"cfg": {"lr": 0.1}})
    with self.assertRaises(AttributeError):
      exp[f"cfg{C.SEP}missing"]


class ApplySingleCTest(unittest.TestCase):
  def test_satisfied_and_unsatisfied(self):
    exp = ExperimentData({"acc": 0.9})
    self.assertTrue(exp.apply_single_C(C("acc") == 0.9))
    self.assertFalse(exp.apply_single_C(C("acc") == 0.1))

  def test_missing_skip_returns_false(self):
    exp = ExperimentData({"acc": 0.9})
    self.assertFalse(exp.apply_single_C(C("missing", C.MISSING_SKIP) == 1))

  def test_missing_fail_reraises(self):
    exp = ExperimentData({"acc": 0.9})
    with self.assertRaises(AttributeError):
      exp.apply_single_C(C("missing", C.MISSING_FAIL) == 1)

  def test_unknown_missing_strategy_raises_not_implemented(self):
    exp = ExperimentData({"acc": 0.9})
    bad = ImplementedC(["missing"], 1, lambda x, y: x == y, "missing", "weird")
    with self.assertRaises(NotImplementedError):
      exp.apply_single_C(bad)


class QueryTest(unittest.TestCase):
  def setUp(self):
    self.db = Database()
    self.db.entries = [
      ExperimentData({"name": "a", "acc": 0.9}),
      ExperimentData({"name": "b", "acc": 0.5}),
      ExperimentData({"name": "c"}),  # no acc
    ]

  def test_add_filter_returns_self_for_chaining(self):
    q = Query()
    self.assertIs(q.add_filter(C("acc") == 0.9), q)
    self.assertIs(q.add_callback(lambda e: True), q)

  def test_apply_returns_new_database(self):
    q = Query().add_filter(C("name", C.MISSING_SKIP) == "a")
    result = q.apply(self.db)
    self.assertIsInstance(result, Database)
    self.assertIsNot(result, self.db)
    self.assertEqual([e.data["name"] for e in result.entries], ["a"])

  def test_multiple_filters_are_anded(self):
    q = (Query()
         .add_filter(C("acc", C.MISSING_SKIP) >= 0.6)
         .add_filter(C("name", C.MISSING_SKIP) == "a"))
    result = q.apply(self.db)
    self.assertEqual([e.data["name"] for e in result.entries], ["a"])

  def test_missing_skip_excludes_entry(self):
    q = Query().add_filter(C("acc", C.MISSING_SKIP) > 0)
    result = q.apply(self.db)
    # Entry "c" has no acc and is skipped.
    self.assertEqual({e.data["name"] for e in result.entries}, {"a", "b"})

  def test_callback_filter(self):
    q = Query().add_callback(lambda e: e.data.get("name") in {"a", "c"})
    result = q.apply(self.db)
    self.assertEqual({e.data["name"] for e in result.entries}, {"a", "c"})

  def test_empty_query_keeps_everything(self):
    self.assertEqual(len(Query().apply(self.db).entries), 3)

  def test_clone_is_independent(self):
    q = Query().add_filter(C("acc", C.MISSING_SKIP) >= 0.6)
    clone = q.clone()
    clone.add_filter(C("name", C.MISSING_SKIP) == "zzz")
    # Mutating the clone must not affect the original's results.
    self.assertEqual(len(q.apply(self.db).entries), 1)
    self.assertEqual(len(clone.apply(self.db).entries), 0)

  def test_entries_passed_by_reference(self):
    q = Query().add_filter(C("name", C.MISSING_SKIP) == "a")
    result = q.apply(self.db)
    self.assertIs(result.entries[0], self.db.entries[0])


class ColumnsTest(unittest.TestCase):
  def test_empty_database(self):
    self.assertEqual(Database().columns(), [])

  def test_flat_keys_sorted_and_deduplicated(self):
    db = Database()
    db.entries = [
      ExperimentData({"acc": 0.9, "name": "a"}),
      ExperimentData({"acc": 0.5, "loss": 1.0}),
    ]
    self.assertEqual(db.columns(), ["acc", "loss", "name"])

  def test_nested_keys_use_separator(self):
    db = Database()
    db.entries = [
      ExperimentData({"cfg": {"lr": 0.1, "epochs": 10}, "acc": 0.9}),
    ]
    self.assertEqual(db.columns(), ["acc", f"cfg{C.SEP}epochs", f"cfg{C.SEP}lr"])

  def test_columns_feed_back_into_getitem(self):
    db = Database()
    db.entries = [ExperimentData({"cfg": {"lr": 0.1}})]
    col = db.columns()[0]
    self.assertEqual(db.entries[0][col], 0.1)

  def test_non_mapping_value_is_a_leaf(self):
    db = Database()
    db.entries = [ExperimentData({"loss": [1, 2, 3]})]
    self.assertEqual(db.columns(), ["loss"])


class DatabaseGetItemTest(unittest.TestCase):
  def setUp(self):
    self.db = Database()
    self.db.entries = [
      ExperimentData({"name": "a", "cfg": {"lr": 0.1}}),
      ExperimentData({"name": "b", "cfg": {"lr": 0.2}}),
    ]

  def test_exact_leaf_returns_list_of_values(self):
    self.assertEqual(self.db["name"], ["a", "b"])

  def test_group_path_returns_list_of_nodes(self):
    result = self.db["cfg"]
    self.assertTrue(all(isinstance(n, Node) for n in result))
    self.assertEqual([n["lr"] for n in result], [0.1, 0.2])

  def test_missing_key_raises_attribute_error(self):
    with self.assertRaises(AttributeError):
      self.db["nope"]


class SelectTest(unittest.TestCase):
  def setUp(self):
    self.db = Database()
    self.db.entries = [
      ExperimentData({"name": "a", "metrics": {"acc": 0.9, "loss": 0.1}, "params": {"lr": 0.1}}),
      ExperimentData({"name": "b", "metrics": {"acc": 0.5, "loss": 0.4}, "params": {"lr": 0.2}}),
    ]

  def test_group_prefix_returns_dict_of_per_leaf_lists(self):
    view = self.db.select("metrics")
    self.assertEqual(view.columns(), [f"metrics{C.SEP}acc", f"metrics{C.SEP}loss"])
    self.assertEqual({c: view[c] for c in view.columns()},
                      {f"metrics{C.SEP}acc": [0.9, 0.5], f"metrics{C.SEP}loss": [0.1, 0.4]})

  def test_regex_returns_dict_of_per_leaf_lists(self):
    view = self.db.select(".*loss")
    self.assertEqual(view.columns(), [f"metrics{C.SEP}loss"])

  def test_no_match_returns_empty_columns(self):
    view = self.db.select("nope")
    self.assertEqual(view.columns(), [])
    self.assertEqual(len(view.entries), 2)

  def test_multiple_patterns_are_unioned(self):
    view = self.db.select("params", f"metrics{C.SEP}acc")
    self.assertEqual(view.columns(), [f"metrics{C.SEP}acc", f"params{C.SEP}lr"])

  def test_entries_are_shared_not_copied(self):
    view = self.db.select("metrics")
    self.assertIs(view.entries, self.db.entries)


class DatabaseLoadTest(unittest.TestCase):
  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.root = Path(self._tmp.name)

  def tearDown(self):
    self._tmp.cleanup()

  def _write_json(self, relative, obj):
    path = self.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))

  def test_load_from_disk_collects_entries(self):
    self._write_json("a.json", {"acc": 1})
    self._write_json("b.json", [{"acc": 2}, {"acc": 3}])
    db = Database()
    db.load_from_disk(paths=[(str(self.root / "*.json"), "json")])
    self.assertEqual({e.data["acc"] for e in db.entries}, {1, 2, 3})

  def test_load_from_disk_empty(self):
    db = Database()
    db.load_from_disk(paths=[(str(self.root / "*.json"), "json")])
    self.assertEqual(db.entries, [])

  def test_load_then_query_end_to_end(self):
    self._write_json("a.json", {"name": "keep", "acc": 0.9})
    self._write_json("b.json", {"name": "drop", "acc": 0.1})
    db = Database()
    db.load_from_disk(paths=[(str(self.root / "*.json"), "json")])
    result = Query().add_filter(C("acc", C.MISSING_SKIP) > 0.5).apply(db)
    self.assertEqual([e.data["name"] for e in result.entries], ["keep"])

  def _write_broken_json(self, relative):
    path = self.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json")

  def test_config_on_failed_parse_raise_propagates(self):
    self._write_broken_json("bad.json")
    db = Database(VildemaConfig(on_failed_parse="raise"))
    with self.assertRaises(Exception):
      db.load_from_disk(paths=[(str(self.root / "*.json"), "json")])

  def test_config_on_failed_parse_ignore_skips_bad_files(self):
    self._write_broken_json("bad.json")
    self._write_json("good.json", {"acc": 1})
    db = Database(VildemaConfig(on_failed_parse="ignore"))
    db.load_from_disk(paths=[(str(self.root / "*.json"), "json")])
    self.assertEqual({e.data["acc"] for e in db.entries}, {1})


if __name__ == "__main__":
  unittest.main()
