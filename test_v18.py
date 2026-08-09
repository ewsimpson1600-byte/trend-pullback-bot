import unittest

import develop_v18 as v18


class FrozenStructureTest(unittest.TestCase):
    def test_extension_band_is_bounded(self):
        self.assertGreater(v18.MIN_EXTENSION_ATR, 0)
        self.assertGreater(v18.MAX_EXTENSION_ATR, v18.MIN_EXTENSION_ATR)
        self.assertLessEqual(v18.MAX_FAILURE_BARS, 3)


if __name__ == "__main__":
    unittest.main()
