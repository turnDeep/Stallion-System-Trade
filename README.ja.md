# Auto-Swing-Trade-Bot

[English](./README.md) | 日本語

Auto-Swing-Trade-Bot は、米国株向けの完全自動化された **Qullamaggie 型ブレイクアウト・スイングトレードシステム** です。毎日の銘柄スクリーニングから日中の自動発注・ポジション管理まで、すべてを自律的に行います。

## 🌟 システムの概要（System Overview）

このシステムは、日足の保ち合い（コンソリデーション）から上にブレイクアウトする瞬間を捉え、数日から数週間にわたる大きなトレンドに乗る（スイングトレード）ことを目的としています。

- **対象ユニバース:** Financial Modeling Prep (FMP) から取得する、米国株の時価総額上位3000銘柄。
- **トレード戦略 (Base Compact):** 強いモメンタムと構造的なブレイクアウトを持つ銘柄に焦点を当てます。高い出来高インパクト（`cum_vol_ratio >= 1.5`）と、日中の明確な上抜け（`trigger_close >= pivot * 1.01`）を条件にエントリーします。広範なバックテストの結果、本物のブレイクアウト銘柄は自然と相対的強さ（RS）が高くなることが判明したため、`leader_score >= 60` は選別用ではなく、異常値を弾く「安全装置（フェイルセーフ）」としてのみ機能しています。
- **ポートフォリオ管理:** マルチバガー（大化け銘柄）を積極的に狙うため、3枠（各33.3%）に極度に集中投資するアーキテクチャを採用しています。資金を守るための「納税用リザーブ管理機能」も備えています。
- **動的エグジット戦略:** リーダー銘柄のライフサイクルに合わせた多段階のランナー管理システム：
  - +20%到達時に部分利確を行い、初期利益を確保します。
  - 残りのコアポジションは、段階的な移動平均線（10, 21, 50 DMA）を用いてトレイリングストップを行います。
  - 評価益が+200%に到達した「スーパーウィナー」は、トレンド構造が崩れるまで利益を伸ばす「前日安値ストップ」へ昇格します。
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
python scripts/manage_tax_reserve.py archive --year 2024
python scripts/manage_tax_reserve.py history
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

- 現在のシステムは、頑健な「Base Compact」戦略構成に完全移行しています。過剰最適化（カーブフィッティング）の原因となる複雑な業種別キャップやA+銘柄の入れ替えロジックは無効化され、純粋な「出来高インパクト（`cum_vol_ratio_at_trigger`）順」のランキングを採用しています。
- ルート直下の薄いラッパーは削除し、`scripts/` のコマンドへ統一しました。
- 旧 `backtester.py` 入口は `scripts/backtest.py` へ移動し、実装本体は `core/backtester.py` に残しています。
- 現行システムで使われておらず、存在しない `run_backtest` export を参照していた旧 `optimizer.py` は削除しました。
- 旧60日ML pipeline名、threshold classifier、same-day flatten 前提の構成は使っていません。
- Yahooの5分足には取得制約があるため、ローカル履歴を継続保存しながら監査します。
- LIVE認証情報が不足している場合は、自動的にDEMOモードで起動します。