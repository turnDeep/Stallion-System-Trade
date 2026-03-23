import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stallion.bar_aggregator import _to_utc_timestamp
from stallion.features import _ensure_market_timezone


class TimezoneHardeningTests(unittest.TestCase):
    def test_to_utc_timestamp_rejects_naive_input(self) -> None:
        with self.assertRaises(ValueError):
            _to_utc_timestamp(pd.Timestamp("2026-03-24 09:30:00"))

    def test_ensure_market_timezone_preserves_timezone_aware_values(self) -> None:
        series = _ensure_market_timezone(
            pd.to_datetime(["2026-03-24 14:15:00+00:00"]),
            market_timezone="America/New_York",
        )
        self.assertEqual(str(series.dt.tz), "America/New_York")
        self.assertEqual(series.iloc[0].isoformat(), "2026-03-24T10:15:00-04:00")

    def test_ensure_market_timezone_localizes_naive_values_to_market_time(self) -> None:
        series = _ensure_market_timezone(
            pd.to_datetime(["2026-03-24 10:15:00"]),
            market_timezone="America/New_York",
        )
        self.assertEqual(str(series.dt.tz), "America/New_York")
        self.assertEqual(series.iloc[0].isoformat(), "2026-03-24T10:15:00-04:00")


if __name__ == "__main__":
    unittest.main()
