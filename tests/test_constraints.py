"""Unit tests for the constraint builder (C / ImplementedC)."""
import unittest

from vildema.constraints import C, ImplementedC


class CBuilderTest(unittest.TestCase):
  def test_simple_path_split(self):
    c = C("name")
    self.assertEqual(c.filter_path, ["name"])
    self.assertEqual(c.name, "name")

  def test_nested_path_split_on_separator(self):
    c = C(f"a{C.SEP}b{C.SEP}c")
    self.assertEqual(c.filter_path, ["a", "b", "c"])

  def test_default_missing_value_is_fail(self):
    self.assertEqual(C("x").missing_value, C.MISSING_FAIL)

  def test_explicit_missing_value(self):
    self.assertEqual(C("x", C.MISSING_SKIP).missing_value, "skip")


class CComparisonOperatorsTest(unittest.TestCase):
  def test_operators_build_implemented_c(self):
    for impl in (C("x") == 1, C("x") != 1, C("x") < 1,
                 C("x") <= 1, C("x") > 1, C("x") >= 1, C("x").is_in([1])):
      self.assertIsInstance(impl, ImplementedC)

  def test_eq_test(self):
    impl = C("x") == 5
    self.assertEqual(impl.value, 5)
    self.assertTrue(impl.test(5, impl.value))
    self.assertFalse(impl.test(6, impl.value))

  def test_ne_test(self):
    impl = C("x") != 5
    self.assertTrue(impl.test(6, impl.value))
    self.assertFalse(impl.test(5, impl.value))

  def test_lt_le_gt_ge_tests(self):
    self.assertTrue((C("x") < 5).test(4, 5))
    self.assertFalse((C("x") < 5).test(5, 5))
    self.assertTrue((C("x") <= 5).test(5, 5))
    self.assertTrue((C("x") > 5).test(6, 5))
    self.assertFalse((C("x") > 5).test(5, 5))
    self.assertTrue((C("x") >= 5).test(5, 5))

  def test_is_in_test(self):
    impl = C("x").is_in([1, 2, 3])
    self.assertTrue(impl.test(2, impl.value))
    self.assertFalse(impl.test(9, impl.value))

  def test_metadata_propagates_to_implemented_c(self):
    impl = C(f"a{C.SEP}b", C.MISSING_SKIP) == 1
    self.assertEqual(impl.filter_path, ["a", "b"])
    self.assertEqual(impl.name, f"a{C.SEP}b")
    self.assertEqual(impl.missing_value, C.MISSING_SKIP)


class CExpandTest(unittest.TestCase):
  def test_expand_matches_exact_leaf(self):
    self.assertEqual([c.name for c in C("acc").expand(["acc", "loss"])], ["acc"])

  def test_expand_group_prefix_matches_nested_leaves(self):
    cols = ["metrics//acc", "metrics//loss", "params//lr"]
    self.assertEqual([c.name for c in C("metrics").expand(cols)],
                      ["metrics//acc", "metrics//loss"])

  def test_expand_does_not_match_unrelated_prefix(self):
    cols = ["params//lr", "parameters//other"]
    self.assertEqual([c.name for c in C("params").expand(cols)], ["params//lr"])

  def test_expand_unknown_group_returns_empty_list(self):
    self.assertEqual(C("nope").expand(["acc", "loss"]), [])

  def test_expand_propagates_missing_value_to_results(self):
    results = C("metrics", C.MISSING_SKIP).expand(["metrics//acc"])
    self.assertEqual(results[0].missing_value, C.MISSING_SKIP)

  def test_expand_regex_mode_matches_anywhere(self):
    cols = ["metrics//best_train_loss", "metrics//latest_train_loss", "metrics//acc"]
    self.assertEqual([c.name for c in C(".*loss").expand(cols)],
                      ["metrics//best_train_loss", "metrics//latest_train_loss"])

  def test_expand_regex_mode_returns_empty_list_when_no_match(self):
    self.assertEqual(C(".*loss").expand(["metrics//acc"]), [])

  def test_expand_dotted_but_plain_name_is_treated_as_literal(self):
    cols = ["metrics//eval_mmd_sq_t0.05", "metrics//eval_mmd_sq_t0.2"]
    self.assertEqual([c.name for c in C("metrics//eval_mmd_sq_t0.05").expand(cols)],
                      ["metrics//eval_mmd_sq_t0.05"])

  def test_expand_preserves_columns_order(self):
    cols = ["metrics//loss", "metrics//acc"]
    self.assertEqual([c.name for c in C("metrics").expand(cols)],
                      ["metrics//loss", "metrics//acc"])


if __name__ == "__main__":
  unittest.main()
