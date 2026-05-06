from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from core.live_trader import _entry_cash_capacity, _entry_cash_cost, _patch_today_exit_ohlc
from core.modeling import STAGE2_RETIRED_MESSAGE, Stage2RetiredError, fit_hist_gbm
from core.strategy import StandardSystemSpec
from signals.zigzag_breakout_engine import _compute_zigzag_setup_daily


class AlgorithmFixTests(unittest.TestCase):
    def test_zigzag_high_pivot_waits_for_order_confirmation(self) -> None:
        highs = [1.0, 2.0, 6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.5, 2.0, 1.0]
        base = pd.DataFrame(
            {
                "symbol": ["AAA"] * len(highs),
                "date": pd.date_range("2024-01-01", periods=len(highs), freq="D"),
                "open": [2.0] * len(highs),
                "high": highs,
                "low": [0.9, 0.8, 1.0, 0.8, 0.5, 0.8, 1.0, 0.8, 0.4, 0.8, 1.0, 0.8, 0.9],
                "close": [2.0] * len(highs),
                "volume": [1000.0] * len(highs),
                "vol20": [1000.0] * len(highs),
            }
        )

        scored = _compute_zigzag_setup_daily(
            base,
            zigzag_order=2,
            pivot_confirm_bars=1,
            line_min_gap=1,
            line_max_gap=10,
            max_line_slope_pct_per_bar=1.0,
        )

        self.assertFalse(bool(scored.loc[7, "zigzag_line_valid"]))
        self.assertTrue(bool(scored.loc[8, "zigzag_line_valid"]))

    def test_stage2_training_is_explicitly_retired(self) -> None:
        frame = pd.DataFrame({"label_stress_exec": [0, 1, 1]})

        with self.assertRaises(Stage2RetiredError) as ctx:
            fit_hist_gbm(frame, StandardSystemSpec())

        self.assertIn(STAGE2_RETIRED_MESSAGE, str(ctx.exception))

    def test_live_entry_cash_uses_commission_and_slippage(self) -> None:
        settings = SimpleNamespace(
            costs=SimpleNamespace(commission_rate_one_way=0.002, slippage_bps_per_side=5.0)
        )

        self.assertAlmostEqual(_entry_cash_capacity(settings, 1002.5), 1000.0)
        self.assertAlmostEqual(_entry_cash_cost(settings, 10, 100.0), 1002.5)

    def test_eod_exit_patch_includes_intraday_high_and_quote_close(self) -> None:
        daily = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "ts": [pd.Timestamp("2024-01-05 00:00:00Z")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000],
            }
        )
        intraday = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "ts": [
                    pd.Timestamp("2024-01-05 14:30:00Z"),
                    pd.Timestamp("2024-01-05 20:55:00Z"),
                ],
                "open": [100.0, 110.0],
                "high": [112.0, 121.0],
                "low": [98.0, 109.0],
                "close": [111.0, 120.0],
                "volume": [100, 200],
            }
        )
        quotes = pd.DataFrame({"symbol": ["AAA"], "price": [119.0]})

        patched = _patch_today_exit_ohlc(
            daily,
            session_date=pd.Timestamp("2024-01-05"),
            quote_frame=quotes,
            intraday_bars=intraday,
        )

        self.assertEqual(float(patched.loc[0, "close"]), 119.0)
        self.assertEqual(float(patched.loc[0, "high"]), 121.0)
        self.assertEqual(float(patched.loc[0, "low"]), 98.0)


if __name__ == "__main__":
    unittest.main()
