import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import TICKERS, START, END

class TestDataLoaderConfig(unittest.TestCase):
    def test_tickers_defined(self):
        self.assertEqual(TICKERS, ["TSLA", "BND", "SPY"])

    def test_date_range_defined(self):
        self.assertEqual(START, "2015-01-01")
        self.assertEqual(END, "2026-06-30")

if __name__ == "__main__":
    unittest.main()
