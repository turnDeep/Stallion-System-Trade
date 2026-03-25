import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stallion.features import _daily_bar_session_dates, build_daily_feature_history, build_daily_tradeability_flags
from stallion.watchlist_model import make_watchlist_labels


class DailySessionDateAlignmentTests(unittest.TestCase):
    def test_daily_bar_session_dates_keep_utc_calendar_date(self) -> None:
        series = _daily_bar_session_dates(pd.to_datetime(["2026-03-24 00:00:00+00:00"]))
        self.assertEqual(series.iloc[0], pd.Timestamp("2026-03-24"))

    def test_tradeability_flags_use_trading_date_not_prior_new_york_evening(self) -> None:
        daily_bars = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "ts": pd.Timestamp("2026-03-24 00:00:00+00:00"),
                    "close": 10.0,
                    "volume": 2_000_000,
                }
            ]
        )
        flags = build_daily_tradeability_flags(
            daily_bars,
            min_price=5.0,
            min_daily_volume=1_000_000,
            min_dollar_volume=10_000_000,
        )
        self.assertEqual(flags.loc[0, "session_date"], pd.Timestamp("2026-03-24"))
        self.assertEqual(int(flags.loc[0, "is_eligible"]), 1)

    def test_daily_feature_history_uses_trading_date_not_prior_new_york_evening(self) -> None:
        daily_bars = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "ts": pd.Timestamp("2026-03-24 00:00:00+00:00"),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "adj_close": 100.5,
                    "volume": 2_000_000,
                }
            ]
        )
        universe = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "sector": "Tech",
                    "industry": "Software",
                }
            ]
        )

        features = build_daily_feature_history(daily_bars, universe)

        self.assertEqual(features.loc[0, "session_date"], pd.Timestamp("2026-03-24"))

    def test_watchlist_nextday_close_label_aligns_to_feature_date(self) -> None:
        daily_features = pd.DataFrame(
            [
                {"session_date": pd.Timestamp("2026-03-24"), "symbol": "AAA"},
                {"session_date": pd.Timestamp("2026-03-25"), "symbol": "AAA"},
            ]
        )
        daily_bars = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "ts": pd.Timestamp("2026-03-24 00:00:00+00:00"),
                    "adj_close": 100.0,
                    "close": 100.0,
                },
                {
                    "symbol": "AAA",
                    "ts": pd.Timestamp("2026-03-25 00:00:00+00:00"),
                    "adj_close": 110.0,
                    "close": 110.0,
                },
            ]
        )
        intraday_labeled = pd.DataFrame(
            [
                {
                    "session_date": pd.Timestamp("2026-03-25"),
                    "symbol": "AAA",
                    "net_return_stress_exec": -0.01,
                }
            ]
        )

        labels = make_watchlist_labels(daily_features, daily_bars, intraday_labeled)
        label_row = labels.loc[
            (labels["feature_date"].eq(pd.Timestamp("2026-03-24"))) & (labels["symbol"].eq("AAA"))
        ].iloc[0]

        self.assertEqual(int(label_row["label_watchlist_nextday_close_up"]), 1)


if __name__ == "__main__":
    unittest.main()
