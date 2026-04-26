# Auto-Swing-Trade-Bot

[English](./README.md) | 日本語

Auto-Swing-Trade-Bot は、米国株向けの Qullamaggie 型ブレイクアウト・スイングトレードシステムです。

- ユニバース: FMP から取得する時価総額上位3000銘柄
- シグナル: 日足セットアップ + 5分足ブレイクアウト確認
- 運用スタイル: 翌日以降も持ち越すスイング保有
- 保存方式: SQLite + Parquet snapshot + 日次 archive
- 実行モード: Webull LIVE、または認証情報がない場合の DEMO 自動フォールバック

## 基本フロー

1. nightly pipeline でユニバース、日足、5分足、signal report、翌営業日の shortlist を更新します。
2. live trader が市場時間中に shortlist を監視し、breakout 検知、entry、hard stop、保有ポジション管理を行います。
3. scheduler が nightly pipeline と market-hours live trader を自動実行します。

## 主なコマンド

Nightly breakout pipeline:

```bash
python scripts/nightly_pipeline.py
```

Live trader:

```bash
python scripts/live_trader.py
```

Scheduler:

```bash
python scripts/scheduler.py
```

Backtester:

```bash
python scripts/backtest.py
```

Tax reserve manager:

```bash
python scripts/manage_tax_reserve.py show
```

Docker:

```bash
docker compose up -d --build
```

## ディレクトリ構成

| Directory | Role |
|---|---|
| `core/` | 本番運用の中核。storage、broker、pipeline、live trader、watchdog |
| `signals/` | standard breakout、zigzag breakout、entry lane、industry priority のシグナルエンジン |
| `backtesting/` | 再利用するスイングバックテストとポートフォリオ検証コード |
| `research/` | キャリブレーションと探索的分析スクリプト |
| `scripts/` | 運用者、Docker、scheduler が使うCLI入口 |
| `configs/` | キャリブレーション済み戦略パラメータ |

## Notes

- ルート直下の薄いラッパーは削除し、`scripts/` のコマンドへ統一しました。
- 旧 `backtester.py` 入口は `scripts/backtest.py` へ移動し、実装本体は `core/backtester.py` に残しています。
- 現行システムで使われておらず、存在しない `run_backtest` export を参照していた旧 `optimizer.py` は削除しました。
- 旧60日ML pipeline名、threshold classifier、same-day flatten 前提の構成は使っていません。
- Yahooの5分足には取得制約があるため、ローカル履歴を継続保存しながら監査します。
- LIVE認証情報が不足している場合は、自動的にDEMOモードで起動します。