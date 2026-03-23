import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stallion.broker import WebullBroker


class WebullBrokerBuyingPowerTests(unittest.TestCase):
    def test_get_account_buying_power_prefers_positive_usd_asset_row(self) -> None:
        broker = WebullBroker.__new__(WebullBroker)
        broker.get_account_balance_raw = lambda: {
            "account_currency_assets": [
                {"currency": "JPY", "cash_balance": "0", "buying_power": "0", "unrealized_profit_loss": "0"},
                {"currency": "USD", "cash_balance": "904.37", "buying_power": "904.37", "unrealized_profit_loss": "0.00"},
            ],
            "total_asset_currency": "143334",
            "total_cash_balance": "143334",
            "total_unrealized_profit_loss": "0",
        }

        self.assertAlmostEqual(WebullBroker.get_account_buying_power(broker), 904.37)

    def test_get_account_buying_power_prefers_top_level_direct_value(self) -> None:
        broker = WebullBroker.__new__(WebullBroker)
        broker.get_account_balance_raw = lambda: {
            "buying_power": "123.45",
            "account_currency_assets": [
                {"currency": "JPY", "cash_balance": "0", "buying_power": "0"},
                {"currency": "USD", "cash_balance": "904.37", "buying_power": "904.37"},
            ],
        }

        self.assertAlmostEqual(WebullBroker.get_account_buying_power(broker), 123.45)


if __name__ == "__main__":
    unittest.main()
