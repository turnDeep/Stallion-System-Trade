# Stallion-System-Trade

[English](./README.md) | 日本語

Stallion-System-Trade は、米国個別株向けの 2 段階 intraday 売買システムです。

- 第1段階: 前日引け後に 7 つの日足特徴量で翌営業日の上位 100 銘柄を選ぶ watchlist モデル
- 第2段階: shortlist に対して 16 特徴量の intraday モデルで 5 分足ごとにスコアリングし、最大 4 銘柄まで発注
- 保存形式: SQLite + Parquet
- Webull 本番認証が不足している場合は自動で DEMO モード
- Discord 通知で起動状態、注文、約定、クローズサマリーを配信

## 動作モード

`.env` の内容で自動判定します。

- `LIVE`
  - `WEBULL_APP_KEY`
  - `WEBULL_APP_SECRET`
  - `WEBULL_ACCOUNT_ID`
  がすべて揃っている場合のみ
- `DEMO`
  - 上記のいずれかが欠けている場合

Discord 通知には `[LIVE]` または `[DEMO]` が付きます。

## 売買フロー

### 第1段階: Nightly watchlist model

ユニバースと前処理:

- 時価総額上位 3000 銘柄を raw universe として保持
- 日足と 5 分足の履歴は full universe のまま保存
- `daily_features` も full universe のまま計算
- その後、stage-1 学習、最新 watchlist scoring、stage-2 学習の直前に流動性フィルタを適用
- 流動性フィルタ:
  - `min_price >= 5.0`
  - `min_daily_volume >= 1,000,000`
  - `min_dollar_volume >= 10,000,000`

引け時点 `t` で以下 7 特徴量を使い、翌営業日 `t+1` の shortlist を作ります。

1. `daily_buy_pressure_prev`
2. `daily_rs_score_prev`
3. `daily_rrs_prev`
4. `prev_day_adr_pct`
5. `industry_buy_pressure_prev`
6. `sector_buy_pressure_prev`
7. `industry_rs_prev`

- モデル: `LogisticRegression`
- `daily_rs_score`: `0.40 * ROC21 + 0.20 * ROC63 + 0.20 * ROC126 + 0.20 * ROC252`
- 出力: 翌営業日用 top 100 shortlist

### 第2段階: Intraday execution

- 対象: stage-1 shortlist のみ
- モデル: `HistGradientBoostingClassifier`
- 時間足:
  - 5分足
  - 5分足から集約した 15 分文脈
  - 前日までの daily 文脈
- シグナル時間帯: 米国市場オープン後 5〜90 分
- エントリー: 次バー始値
- エグジット: 当日引け前に全決済
- 最大保有数: 4

## 注文・スロット管理

live trader は以下を明示的に区別します。

- 空き枠
- 発注済み未約定枠
- 部分約定枠
- 完全約定済み保有枠
- 売却待ち枠
- reserved buying power

重要:

- 未約定注文が残っている間は、その枠を空き枠として扱いません
- 枠を解放するのは以下のいずれかが確認できた後だけです
  - cancel 完了
  - rejected
  - expired
  - failed
  - 売却完了かつ保有解消
- 部分約定は枠を占有したまま維持します

## 買付余力と数量計算

- 使用するのは総資産ではなく **買付余力**
- 当日の開始時点の買付余力を 4 分割してスロット予算にします
- デフォルトでは整数株で発注します
- スロット予算が 1 株価格に届かない場合は:
  - 発注しない
  - 理由をログと Discord に残す
- 成行が失敗した場合は:
  - 設定可能な marketable limit order にフォールバックします

## Discord 通知

`DISCORD_BOT_TOKEN` と `DISCORD_CHANNEL_ID` が両方ある場合、以下を送ります。

- scheduler 起動
- nightly pipeline 開始 / 完了 / 失敗
- プレマーケット通知
- 買い発注通知
- 約定通知
- cancel 要求通知
- 市場クローズ時サマリー

`DISCORD_BOT_TOKEN` だけある場合は token の疎通確認はできますが、実際に送信するには `DISCORD_CHANNEL_ID` も必要です。

## 環境変数

`.env.example` をコピーして `.env` を作成してください。

```env
FMP_API_KEY=
WEBULL_APP_KEY=
WEBULL_APP_SECRET=
WEBULL_ACCOUNT_ID=
DISCORD_BOT_TOKEN=
DISCORD_CHANNEL_ID=
```

## 主な実行コマンド

Nightly pipeline:

```bash
python ml_pipeline_60d.py
```

Live trader:

```bash
python webull_live_trader.py
```

Scheduler:

```bash
python master_scheduler.py
```

Docker:

```bash
docker compose up -d --build
```

初回デプロイ時は履歴取得と特徴量構築のため時間がかかります。以後の再起動では既存の SQLite + Parquet artifact を再利用します。

## Docker メモ

- `/app/data` と `/app/reports` は named volume です
- 必要 artifact が無いときだけ startup bootstrap を走らせます
- healthcheck は `stallion.watchdog` を使用します

## 主なファイル

| ファイル | 役割 |
|---|---|
| `ml_pipeline_60d.py` | nightly pipeline 入口 |
| `webull_live_trader.py` | live 実行入口 |
| `master_scheduler.py` | scheduler と bootstrap |
| `stallion/config.py` | 設定と DEMO/LIVE 判定 |
| `stallion/storage.py` | SQLite + Parquet 保存 |
| `stallion/watchlist_model.py` | stage-1 watchlist 学習 |
| `stallion/modeling.py` | stage-2 HistGBM 学習 / 推論 |
| `stallion/live_trader.py` | live の状態管理・発注・通知 |
| `stallion/broker.py` | Webull JP wrapper / demo broker |
| `stallion/discord_notifier.py` | 非同期 Discord 通知 |
| `stallion/slot_manager.py` | スロット状態と reserved buying power |

## 注意

本番利用前に、必ず以下を確認してください。

- Webull JP 口座の API 権限
- Discord channel 設定
- ホスト側の時刻と市場時間
- 初回 bootstrap が完了していること

不安がある場合は、まず DEMO モードで確認してください。
