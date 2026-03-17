from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd
from webullsdkcore.client import ApiClient
from webullsdkcore.common.region import Region
from webullsdktrade.api import API

from .config import Settings


LOGGER = logging.getLogger(__name__)


def _safe_json(response) -> Any:
    try:
        return response.json()
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _find_nested(payload: Any, candidate_keys: tuple[str, ...]) -> Any:
    if isinstance(payload, dict):
        lowered = {str(key).lower(): value for key, value in payload.items()}
        for key in candidate_keys:
            if key.lower() in lowered:
                return lowered[key.lower()]
        for value in payload.values():
            found = _find_nested(value, candidate_keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_nested(item, candidate_keys)
            if found is not None:
                return found
    return None


@dataclass(frozen=True)
class BrokerProbe:
    region: str
    account_list_ok: bool
    balance_ok: bool
    positions_ok: bool
    account_count: int


class WebullBroker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.region = Region.JP.value
        self._api = API(
            ApiClient(
                settings.credentials.webull_app_key,
                settings.credentials.webull_app_secret,
                self.region,
            )
        )

    @property
    def api(self) -> API:
        return self._api

    @property
    def account_id(self) -> str:
        account_id = (self.settings.credentials.webull_account_id or "").strip()
        if not account_id:
            raise ValueError("WEBULL_ACCOUNT_ID is required.")
        return account_id

    def probe(self) -> BrokerProbe:
        account_list_ok = False
        balance_ok = False
        positions_ok = False
        account_count = 0

        try:
            response = self.api.account_v2.get_account_list()
            payload = _safe_json(response)
            account_count = len(payload) if isinstance(payload, list) else 0
            account_list_ok = getattr(response, "status_code", None) == 200
        except Exception:
            LOGGER.exception("Webull account list probe failed")

        try:
            response = self.api.account_v2.get_account_balance(self.account_id)
            balance_ok = getattr(response, "status_code", None) == 200 and isinstance(_safe_json(response), dict)
        except Exception:
            LOGGER.exception("Webull balance probe failed")

        try:
            response = self.api.account_v2.get_account_position(self.account_id)
            payload = _safe_json(response)
            positions_ok = getattr(response, "status_code", None) == 200 and isinstance(payload, list)
        except Exception:
            LOGGER.exception("Webull positions probe failed")

        return BrokerProbe(
            region=self.region,
            account_list_ok=account_list_ok,
            balance_ok=balance_ok,
            positions_ok=positions_ok,
            account_count=account_count,
        )

    def get_account_list(self) -> list[dict[str, Any]]:
        response = self.api.account_v2.get_account_list()
        payload = _safe_json(response)
        return payload if isinstance(payload, list) else []

    def get_account_balance_raw(self) -> dict[str, Any]:
        response = self.api.account_v2.get_account_balance(self.account_id)
        payload = _safe_json(response)
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected account balance payload.")
        return payload

    def get_account_buying_power(self) -> float:
        payload = self.get_account_balance_raw()
        direct = _as_float(payload.get("buying_power"))
        if direct is not None and direct > 0:
            return direct

        nested_direct = _as_float(
            _find_nested(
                payload,
                (
                    "buying_power",
                    "cash_buying_power",
                    "daytrade_buying_power",
                    "overnight_buying_power",
                    "available_buying_power",
                ),
            )
        )
        if nested_direct is not None and nested_direct > 0:
            return nested_direct

        asset_rows = payload.get("account_currency_assets") or []
        if isinstance(asset_rows, list):
            for row in asset_rows:
                if not isinstance(row, dict):
                    continue
                buying_power = _as_float(row.get("buying_power"))
                if buying_power is not None and buying_power > 0:
                    return buying_power
                cash_balance = _as_float(row.get("cash_balance"))
                if cash_balance is not None and cash_balance > 0:
                    return cash_balance

        cash = _as_float(payload.get("total_cash_balance"))
        if cash is not None and cash > 0:
            return cash

        raise RuntimeError("Could not derive buying power from Webull balance payload.")

    def get_account_equity(self) -> float:
        payload = self.get_account_balance_raw()
        direct = _as_float(payload.get("total_asset_currency"))
        if direct is not None and direct > 0:
            return direct

        cash = _as_float(payload.get("total_cash_balance")) or 0.0
        upl = _as_float(payload.get("total_unrealized_profit_loss")) or 0.0
        fallback = cash + upl
        if fallback > 0:
            return fallback

        raise RuntimeError("Could not derive account equity from Webull balance payload.")

    def get_positions_df(self) -> pd.DataFrame:
        response = self.api.account_v2.get_account_position(self.account_id)
        payload = _safe_json(response)
        if not isinstance(payload, list) or not payload:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "quantity",
                    "available_quantity",
                    "avg_price",
                    "market_value",
                    "payload_json",
                ]
            )

        rows: list[dict[str, Any]] = []
        for item in payload:
            symbol = _find_nested(item, ("symbol", "ticker", "ticker_symbol", "display_symbol", "instrument_symbol"))
            quantity = _find_nested(item, ("quantity", "position", "holding_quantity", "qty", "filled_quantity"))
            available_quantity = _find_nested(item, ("available_quantity", "sellable_quantity", "available_qty"))
            avg_price = _find_nested(item, ("avg_price", "average_cost", "cost_price", "average_price"))
            market_value = _find_nested(item, ("market_value", "position_value", "market_val"))
            if symbol is None:
                continue
            rows.append(
                {
                    "symbol": str(symbol).upper(),
                    "quantity": _as_int(quantity) or 0,
                    "available_quantity": _as_int(available_quantity),
                    "avg_price": _as_float(avg_price),
                    "market_value": _as_float(market_value),
                    "payload_json": json.dumps(item, ensure_ascii=True, default=str),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "quantity",
                    "available_quantity",
                    "avg_price",
                    "market_value",
                    "payload_json",
                ]
            )
        return frame.groupby("symbol", as_index=False).last()

    def get_order_history_df(self, *, lookback_days: int = 7, page_size: int = 100) -> pd.DataFrame:
        start = str(date.today() - timedelta(days=lookback_days))
        end = str(date.today())
        response = self.api.order_v2.get_order_history_request(
            self.account_id,
            page_size=page_size,
            start_date=start,
            end_date=end,
        )
        payload = _safe_json(response)
        if not isinstance(payload, list) or not payload:
            return pd.DataFrame(
                columns=[
                    "client_order_id",
                    "order_id",
                    "symbol",
                    "side",
                    "status",
                    "quantity",
                    "filled_quantity",
                    "place_time_at",
                    "filled_time_at",
                    "payload_json",
                ]
            )
        rows: list[dict[str, Any]] = []
        for item in payload:
            symbol = item.get("symbol")
            if symbol is None:
                symbol = _find_nested(item.get("items"), ("symbol", "ticker", "display_symbol", "instrument_symbol"))
            rows.append(
                {
                    "client_order_id": item.get("client_order_id"),
                    "order_id": item.get("order_id"),
                    "trade_id": item.get("trade_id"),
                    "symbol": str(symbol).upper() if symbol else None,
                    "side": item.get("side"),
                    "status": item.get("status"),
                    "quantity": _as_int(item.get("quantity")) or 0,
                    "filled_quantity": _as_int(item.get("filled_quantity")) or 0,
                    "place_time_at": item.get("place_time_at"),
                    "filled_time_at": item.get("filled_time_at"),
                    "payload_json": json.dumps(item, ensure_ascii=True, default=str),
                }
            )
        return pd.DataFrame(rows)

    def place_market_order(self, *, symbol: str, side: str, quantity: int) -> dict[str, Any]:
        client_order_id = uuid.uuid4().hex
        new_order = {
            "client_order_id": client_order_id,
            "symbol": str(symbol).upper(),
            "instrument_type": "EQUITY",
            "market": "US",
            "order_type": "MARKET",
            "quantity": str(int(quantity)),
            "support_trading_session": "N",
            "side": side.upper(),
            "time_in_force": "DAY",
            "entrust_type": "QTY",
            "account_tax_type": "SPECIFIC",
        }
        response = self.api.order_v2.place_order(account_id=self.account_id, new_orders=new_order)
        payload = _safe_json(response)
        return {
            "client_order_id": client_order_id,
            "status_code": getattr(response, "status_code", None),
            "response_json": payload if payload is not None else str(response),
            "request_symbol": str(symbol).upper(),
            "request_side": side.upper(),
            "request_quantity": int(quantity),
        }

    def cancel_order(self, *, client_order_id: str) -> dict[str, Any]:
        response = self.api.order_v2.cancel_order_v2(self.account_id, client_order_id)
        payload = _safe_json(response)
        return {
            "client_order_id": client_order_id,
            "status_code": getattr(response, "status_code", None),
            "response_json": payload if payload is not None else str(response),
        }
