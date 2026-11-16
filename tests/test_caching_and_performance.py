import unittest

import pandas as pd

from utils.limits import ResourcePolicy

try:
    from utils.parsing import cached_load_dataframe
except ImportError:
    cached_load_dataframe = None

class TestCachingAndPerformance(unittest.TestCase):
    def test_cached_load_dataframe_exists_and_bounds_memory(self):
        self.assertIsNotNone(cached_load_dataframe, "cached_load_dataframe should be implemented in Utils.parsing")
        df = pd.DataFrame({
            "num_col": list(range(200)),
            "cat_col": ["category_" + str(i % 5) for i in range(200)]
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        loaded_df = cached_load_dataframe(csv_bytes, file_name="benchmark_test.csv")
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), 200)
        self.assertLess(loaded_df.memory_usage(deep=True).sum(), ResourcePolicy.MAX_DATAFRAME_MEMORY_BYTES)

    def test_cached_load_dataframe_cache_decorator_attributes(self):
        self.assertTrue(
            hasattr(cached_load_dataframe, "clear"),
            "cached_load_dataframe should have Streamlit caching clear method"
        )

if __name__ == "__main__":
    unittest.main()
