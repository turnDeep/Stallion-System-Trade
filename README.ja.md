# Auto-Swing-Trade-Bot

[English](./README.md) | 日本語

Auto-Swing-Trade-Bot は、米国株向けの Qullamaggie 型 breakout スイングトレードシステムです。

- ユニバース: FMP から取得する時価総額上位 3000 銘柄
- シグナル: 日足 setup と 5 分足 breakout 確認
- 保有スタイル: オーバーナイト前提のスイング保有
- 保存方式: SQLite + Parquet snapshot + 日次 archive
- 実行モード: Webull を使う LIVE と、自動フォールバックの DEMO

## 基本フロー

1. nightly breakout pipeline でユニバース、日足、5分足、signal report、翌営業日の shortlist を更新します。
2. live trader が market hours 中に shortlist を監視し、breakout 検出、エントリー、保有ポジション管理を行います。
3. scheduler が nightly 実行と live 実行を自動で回します。

## 主なコマンド

Nightly breakout pipeline:

```bash
python nightly_breakout_pipeline.py
```

Live trader:

```bash
python breakout_live_trader.py
```

Scheduler:

```bash
python master_scheduler.py
```

Backtester:

```bash
python backtester.py
```

Docker:

```bash
docker compose up -d --build
```

## 主なファイル

| ファイル | 役割 |
|---|---|
| `nightly_breakout_pipeline.py` | nightly pipeline のトップレベル入口 |
| `breakout_live_trader.py` | live trader のトップレベル入口 |
| `master_scheduler.py` | scheduler と bootstrap ループ |
| `breakout_signal_engine.py` | 日足 breakout score 計算エンジン |
| `breakout_signal_report.py` | setup / breakout report と golden-rule filter |
| `qullamaggie_breakout_backtest.py` | スイング用 backtest / exit engine |
| `stallion/nightly_pipeline.py` | nightly refresh、repair fetch、signal report 生成 |
| `stallion/live_trader.py` | live polling、entry、hard stop、引け後 exit 判定 |
| `stallion/breakout_bridge.py` | signal / sizing / exit / backtest の橋渡し |
| `stallion/storage.py` | SQLite + Parquet の運用ストア |
| `stallion/fmp.py` | FMP ユニバース取得と yfinance bar 取得 |

## 補足

- 旧来の 60 日 ML pipeline 名、threshold classifier、same-day flatten 前提の構成は使っていません。
- Yahoo の 5 分足には取得制約があるため、この repo ではローカル履歴を継続保存しつつ監査します。
- LIVE 用の認証情報が不足している場合は、自動的に DEMO モードで起動します。
