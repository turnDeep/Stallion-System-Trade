# Stallion-System-Trade

Language / 言語: [English](./README.md) | **日本語**

Stallion-System-Trade は、現在 **標準のラッセル3000向けイントラデイ売買ロジックを実運用するための live-trading scaffold** として構成されています。

- 前夜に **時価総額上位3000銘柄** を更新
- **252営業日以上** の split-adjusted 日足から日次コンテキストを計算
- 場中は **5分足** を監視
- **15分足コンテキストは 5分足から導出**
- **16特徴量の `hist_gbm_extended 5m_start` ロジック** でスコアリング
- **next-bar open でエントリー**, **当日引けでクローズ**, **1セッション最大4ポジション**
- データ保存は pickle ではなく **SQLite + Parquet**

## 標準ロジック

本番システムは、現在の研究コードベースで使っている標準ロジックに合わせています。

### 売買時間帯

- シグナル判定ウィンドウ: **米国市場オープン後 5〜90分**
- 最速の実約定タイミング: **次の5分足の始値**
- オーバーナイトなし
- 同一銘柄は1日1回まで
- 同時保有上限: **4**

### モデル

- モデル: `HistGradientBoostingClassifier`
- 特徴量数: **16**
- 学習ごとの閾値:

```text
threshold = max(0.55, 90th percentile of train_scores)
```

### エントリー選抜

このシステムは、後から振り返って「その日の score 上位4銘柄」を選ぶ方式ではありません。

動き方は以下です。

1. 場中に全候補をリアルタイムでスコアリング
2. `score >= threshold` の銘柄だけ残す
3. **timestamp 順** に処理する
4. 同じ timestamp に複数銘柄がある場合は **score 降順**
5. **4ポジション** 埋まるまで約定する

### エグジット

- 運用上の前提: **当日引けクローズ**
- バックテスト / live 会計前提:
  - 手数料: `片道 0.2%`
  - スリッページ: `片道 5 bps`
  - スプレッド: `往復 5 bps`

## 16個の live 特徴量

実運用で使う特徴量セットは以下です。

1. `daily_buy_pressure_prev`
2. `prev_day_adr_pct`
3. `industry_buy_pressure_prev`
4. `EMA_8_15`
5. `distance_to_prev_day_high`
6. `close_vs_vwap_15`
7. `sector_buy_pressure_prev`
8. `daily_rrs_prev`
9. `daily_rs_score_prev`
10. `distance_to_avwap_63_prev`
11. `volume_spike_5m`
12. `industry_rs_prev`
13. `same_slot_avg_vol_20d`
14. `rs_x_intraday_rvol`
15. `intraday_range_expansion_vs_atr`
16. `prev_day_close_vs_sma50`

## 必要データ

### 前夜に全ユニバースで必要なデータ

- 時価総額上位3000銘柄
- `symbol`, `exchange`, `sector`, `industry`, `market_cap`
- split-adjusted 日足 OHLCV
- `SPY` の日足 OHLCV

### 場中に必要なデータ

- 監視 shortlist の当日5分足 OHLCV
- 5分足から導出した15分足コンテキスト

### ローカルに保持しておく履歴

- 日足: **300〜400営業日**
- 5分足: **20〜40営業日**
- 同時刻出来高履歴: **20営業日**

## ストレージ構成

このプロジェクトは次の形で保存します。

- **SQLite**
  - ユニバース metadata
  - 日足バー
  - 5分足バー
  - 日次特徴量
  - shortlist
  - model registry
  - live signals / fills
- **Parquet**
  - raw daily snapshots
  - raw intraday snapshots
  - daily feature snapshots
  - nightly shortlist snapshots

重要な点:

- pickle は本番の主保存形式ではありません
- Parquet は再現可能な snapshot と研究再利用向けです
- SQLite は運用時の operational store です

## 主なファイル

| File | Role |
|---|---|
| `ml_pipeline_60d.py` | nightly pipeline の入口 |
| `webull_live_trader.py` | live execution の入口 |
| `backtester.py` | event-driven backtest の入口 |
| `master_scheduler.py` | 日次 scheduler |
| `stallion/config.py` | 実行設定とパス管理 |
| `stallion/storage.py` | SQLite + Parquet 保存 |
| `stallion/features.py` | 日足 / イントラデイ特徴量生成 |
| `stallion/modeling.py` | HistGBM 学習 / スコアリング / 閾値管理 |
| `stallion/nightly_pipeline.py` | ユニバース更新、特徴量生成、モデル学習、shortlist 作成 |
| `stallion/live_trader.py` | リアルタイム監視、スコアリング、選抜、注文執行 |

## 実行フロー

### Nightly pipeline

```bash
python ml_pipeline_60d.py
```

やること:

1. FMP から top-3000 universe を更新
2. 日足と5分足のローカル履歴を更新
3. 日次特徴量履歴を生成
4. イントラデイ学習 panel を生成
5. HistGBM モデルを学習
6. モデル artifact と threshold を保存
7. 翌営業日の shortlist を作成

### Live trader

```bash
python webull_live_trader.py
```

やること:

1. 保存済みモデルと nightly shortlist をロード
2. 監視銘柄の FMP batch quote を取得
3. operational 5分足ストアへ集約
4. 現在時点の intraday 特徴量を再構築
5. 候補銘柄をスコアリング
6. 最大4銘柄までリアルタイムで選抜
7. Webull に注文をルーティング

### Scheduler

```bash
python master_scheduler.py
```

デフォルトのスケジュール:

- `17:00 America/New_York`: nightly pipeline
- `09:25 America/New_York`: live trader bootstrap

## 環境変数

`.env.example` を元に `.env` を作成してください。

```env
FMP_API_KEY=
WEBULL_APP_KEY=
WEBULL_APP_SECRET=
WEBULL_ACCOUNT_ID=
```

## インストール

```bash
git clone https://github.com/turnDeep/Stallion-System-Trade.git
cd Stallion-System-Trade
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Docker

```bash
docker compose up -d
```

コンテナの entrypoint は引き続き `master_scheduler.py` です。

## Notes

- このリポジトリは、旧 ORB Top-10 システムではなく **標準 daytrade live architecture** 向けに再構成されています。
- 現在の live engine は、意図的にシンプルで追いやすい形にしています。
- より厳しい地合いフィルター、暴落日ブロック、websocket collector などは、この土台の上に追加できます。
