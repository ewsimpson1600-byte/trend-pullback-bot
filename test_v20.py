import unittest

import develop_v20 as v20


class MomentumStructureTest(unittest.TestCase):
    def test_breakout_and_volume_rules_are_material(self):
        self.assertGreaterEqual(v20.BREAKOUT_LOOKBACK, 20)
        self.assertGreater(v20.VOLUME_MULTIPLE, 1)
        self.assertGreaterEqual(v20.MIN_CLOSE_LOCATION, 0.70)


if __name__ == "__main__":
    unittest.main()
