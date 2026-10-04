import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.supervised_umap import EXPECTED_COLUMNS, split_xy, validate_shape


class SupervisedUmapSmokeTests(unittest.TestCase):
    def test_validate_shape_and_split(self):
        data = np.arange(3 * EXPECTED_COLUMNS).reshape(3, EXPECTED_COLUMNS)
        df = pd.DataFrame(data)
        validated = validate_shape(df, Path("synthetic.csv"))
        x, y = split_xy(validated)
        self.assertEqual(x.shape, (3, EXPECTED_COLUMNS - 1))
        self.assertEqual(y.shape, (3,))

    def test_validate_shape_rejects_short_schema(self):
        df = pd.DataFrame(np.zeros((3, EXPECTED_COLUMNS - 1)))
        with self.assertRaises(ValueError):
            validate_shape(df, Path("synthetic.csv"))


if __name__ == "__main__":
    unittest.main()
