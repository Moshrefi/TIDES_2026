import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare', ROOT/'scripts'/'compare_eds_paper.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ComparisonTests(unittest.TestCase):
    def test_signed_and_percentage_difference(self):
        ref = compare.PAPER['TABLE1']['tides_10x']['eds'][0]
        rows = compare.comparison_rows({'dataset':'eds','methods':{'tides_10x':{'ig_nll':ref*1.1}}})
        row = next(r for r in rows if r['table']=='Table 1' and r['method']=='tides_10x' and r['metric']=='ig_nll')
        self.assertAlmostEqual(row['difference_percent'], 10)
        self.assertAlmostEqual(row['difference'], ref*.1)
        self.assertEqual(row['status'], 'measured')

    def test_missing_and_nonfinite_are_not_zero(self):
        rows = compare.comparison_rows({'methods':{'tides_10x':{'ig_nll':float('nan')}}})
        self.assertTrue(all(r['ours'] is None for r in rows))
        self.assertTrue(all(r['difference'] is None for r in rows))
        self.assertEqual({r['status'] for r in rows}, {'not run','missing/nonfinite metric'})

    def test_all_paper_rows_are_visible(self):
        rows = compare.comparison_rows({})
        self.assertEqual(len([r for r in rows if r['table']=='Table 1']), 12)
        self.assertEqual(len([r for r in rows if r['table']=='Table 2']), 18)
        self.assertEqual(len([r for r in rows if r['table']=='Table 4']), 18)

    def test_non_eds_result_rejected(self):
        with self.assertRaises(ValueError):
            compare.comparison_rows({'dataset':'dsec'})


if __name__ == '__main__':
    unittest.main()
