# Auto-Swing-Trade-Bot

[English](./README.md) | 日本語

Auto-Swing-Trade-Bot は、米国株向けの完全自動化された **Qullamaggie 型ブレイクアウト・スイングトレードシステム** です。毎日の銘柄スクリーニングから日中の自動発注・ポジション管理まで、すべてを自律的に行います。

## 🌟 システムの概要（System Overview）

このシステムは、日足の保ち合い（コンソリデーション）から上にブレイクアウトする瞬間を捉え、数日から数週間にわたる大きなトレンドに乗る（スイングトレード）ことを目的としています。

- **対象ユニバース:** Financial Modeling Prep (FMP) から取得する、米国株の時価総額上位3000銘柄。
- **トレード戦略 (Qullamaggie スタイル):** 強いモメンタムを持つ銘柄に焦点を当てます。日足のセットアップ（フラッグなど）を評価し、日中（5分足）の明確なブレイクアウトでエントリーを確認します。
- **高度なシグナルエンジン:** 標準的なブレイクアウトに加え、ジグザグ（Zigzag）ブレイクアウト、エントリーレーン分析、セクター/産業ごとの優先度（Industry priority）など、複数のロジックを統合しています。
- **ポートフォリオ管理:** 翌日以降も持ち越すスイング保有を前提としています。トレイリングストップ等の動的エグジット戦略や、利益を確保するための「納税用リザーブ管理機能」を備えています。
- **実行環境:** Docker上で稼働します。Webull証券APIを通じたリアルトレード（LIVE）に対応し、認証情報がない場合は自動的に「DEMOモード」で動作します。
- **データ保存方式:** APIの取得制限を回避するため、SQLite、Parquet スナップショット、および日次アーカイブファイルを組み合わせた堅牢なローカルデータ基盤を使用しています。

## ⚙️ 基本ワークフロー

1. **Nightly Pipeline (夜間処理):** 市場閉場後に実行されます。ユニバース、日足・5分足データを更新し、シグナルを計算して、翌営業日に監視すべき「ショートリスト（候補銘柄）」を作成します。
2. **Live Trader (自動取引):** 市場時間中に実行されます。ショートリストの銘柄をリアルタイム監視し、ブレイクアウト検知時のエントリー、ハードストップ（損切り）の設定、および保有ポジションの管理を行います。
3. **Scheduler (スケジューラ):** 上記の夜間処理と市場時間中のLive Traderを自動的にオーケストレーション（スケジュール管理）します。

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