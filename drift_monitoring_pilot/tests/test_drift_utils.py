import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from drift_utils import MetricRow, hist_counts_for_values, histogram_to_sample, make_numeric_snip


class DriftUtilsTests(unittest.TestCase):
    def test_hist_counts_clamps_out_of_range_values(self) -> None:
        counts = hist_counts_for_values(
            np.asarray([-5.0, 0.25, 1.25, 2.0, 9.0, math.nan]),
            edges=[0.0, 1.0, 2.0],
        )

        np.testing.assert_array_equal(counts, np.asarray([2, 3]))

    def test_numeric_snip_and_histogram_sample_are_stable(self) -> None:
        rows = [MetricRow(timestamp=None, values={"metric": float(i)}) for i in range(10)]

        snip = make_numeric_snip(rows, hist_bins=5, min_samples=2)
        summary = snip["metric"]
        sample = histogram_to_sample(summary)

        self.assertEqual(summary["n"], 10)
        self.assertAlmostEqual(summary["mean"], 4.5)
        self.assertEqual(len(summary["hist"]["counts"]), 5)
        self.assertEqual(int(np.sum(summary["hist"]["counts"])), 10)
        self.assertEqual(sample.size, 10)
        self.assertTrue(np.all(np.isfinite(sample)))


if __name__ == "__main__":
    unittest.main()
