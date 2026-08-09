import unittest

import develop_v19 as v19


class SwingStructureTest(unittest.TestCase):
    def test_longer_dated_option_and_bounded_hold(self):
        self.assertGreaterEqual(v19.OPTION_DTE, 14)
        self.assertLess(v19.MAX_HOLD_SESSIONS, v19.OPTION_DTE)
        self.assertLess(v19.TARGET_ABS_DELTA, 0.50)


if __name__ == "__main__":
    unittest.main()
