import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stallion.live_trader import _filter_latest_candidate_rows


class LiveTraderTimestampAlignmentTests(unittest.TestCase):
    def test_filter_latest_candidate_rows_matches_timezone_aware_candidates(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "BBB", "BBB"],
                "timestamp": pd.to_datetime(
                    [
                        "2026-03-23 10:10:00-04:00",
                        "2026-03-23 10:15:00-04:00",
                        "2026-03-23 10:10:00-04:00",
                        "2026-03-23 10:15:00-04:00",
                    ]
                ),
            }
        )

        latest_rows = _filter_latest_candidate_rows(
            frame,
            latest_timestamp_utc=pd.Timestamp("2026-03-23 14:15:00+00:00"),
            market_timezone="America/New_York",
        )

        self.assertEqual(set(latest_rows["symbol"]), {"AAA", "BBB"})
        self.assertEqual(len(latest_rows), 2)

    def test_filter_latest_candidate_rows_matches_timezone_naive_candidates(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "BBB", "BBB"],
                "timestamp": pd.to_datetime(
                    [
                        "2026-03-23 10:10:00",
                        "2026-03-23 10:15:00",
                        "2026-03-23 10:10:00",
                        "2026-03-23 10:15:00",
                    ]
                ),
            }
        )

        latest_rows = _filter_latest_candidate_rows(
            frame,
            latest_timestamp_utc=pd.Timestamp("2026-03-23 14:15:00+00:00"),
            market_timezone="America/New_York",
        )

        self.assertEqual(set(latest_rows["symbol"]), {"AAA", "BBB"})
        self.assertEqual(len(latest_rows), 2)


if __name__ == "__main__":
    unittest.main()
