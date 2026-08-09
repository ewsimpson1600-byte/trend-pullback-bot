import unittest

import pandas as pd

import develop_v17 as v17


class OpeningRangeTest(unittest.TestCase):
    def test_uses_first_six_five_minute_bars(self):
        times = pd.date_range("2024-01-02 09:30", periods=7, freq="5min", tz="America/New_York")
        frame = pd.DataFrame({
            "datetime": times, "time": times.strftime("%H:%M"), "date": times.date,
            "high": [1, 2, 3, 4, 5, 6, 99], "low": [0, -1, -2, -3, -4, -5, -99],
        })
        result = v17.add_30_minute_range(frame)
        self.assertEqual(result.iloc[-1]["or30_high"], 6)
        self.assertEqual(result.iloc[-1]["or30_low"], -5)


if __name__ == "__main__":
    unittest.main()
