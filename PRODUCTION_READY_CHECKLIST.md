# Stallion Production-Ready Checklist

最終更新: 2026-03-15

## 現在の判定

- 判定: `Conditional Go`
- 意味:
  - コードベースとしては、`docker compose up -d` 後に bootstrap / scheduler / live trader / healthcheck が回る形まで入った
  - ただし、**実口座での BUY/SELL 発注そのものはまだ試していない**
  - したがって、現時点では **production-ready 実装は完了、live-fire 検証は未完了** という扱い

## 優先度 1-8 の実装状況

| 優先度 | 項目 | 状態 | 実装内容 |
|---|---|---|---|
| 1 | Broker/API 適合性確認 | DONE | Webull JP 向け wrapper を追加し、口座一覧・残高・建玉・注文履歴の read 系 API を確認 |
| 2 | 本物の 5 分 OHLCV 生成 | DONE | FMP quote snapshot を 5 分 bin に集約する bar aggregator を追加 |
| 3 | 引けで自動 SELL | DONE | `15:58 ET` 以降に open position を市場成行でクローズする処理を追加 |
| 4 | 口座残高ベースの数量計算 | DONE | `09:30 ET` 時点の口座資産を 4 分割し、整数株で数量算出する処理を追加 |
| 5 | ポジション / 注文状態テーブル | DONE | SQLite に `live_orders`, `open_positions`, `quote_snapshots`, `heartbeats`, `alerts` などを追加 |
| 6 | 再起動後の状態復元 | DONE | quote snapshot からの 5 分バー再構築、broker order history / position からの再同期を追加 |
| 7 | 未約定・部分約定・取消 | PARTIAL | stale order cancel と status 正規化は追加済み。実口座での部分約定ケースは未実地確認 |
| 8 | watchdog / 障害監視 | DONE | heartbeat と Docker healthcheck を追加し、pipeline 中の stale 判定も調整 |

## 追加で入れた主要コンポーネント

- `stallion/broker.py`
  - Webull JP 用の broker wrapper
- `stallion/bar_aggregator.py`
  - quote snapshot から 5 分 OHLCV を生成
- `stallion/live_trader.py`
  - opening equity sizing
  - signal generation
  - BUY / SELL 発注
  - stale order cancel
  - restart reconciliation
- `stallion/storage.py`
  - SQLite / Parquet 永続化
- `stallion/watchdog.py`
  - healthcheck
- `master_scheduler.py`
  - bootstrap 判定
  - heartbeat

## 実装済みの運用フロー

1. `docker compose up -d`
2. `master_scheduler.py` が起動
3. SQLite / Parquet / model / shortlist が不足していれば bootstrap
4. 平日 `17:00 ET` に nightly pipeline 実行
5. 平日 `09:25 ET` に live trader 起動
6. live trader が FMP quote を polling
7. quote snapshot から 5 分 OHLCV を生成
8. 特徴量を更新して score 計算
9. opening equity を 4 分割して整数株の BUY
10. `15:58 ET` 以降に open position を自動 SELL
11. heartbeat / alerts / orders / positions を SQLite に保存

## まだ残るリスク

### 1. 実発注の最終確認

- BUY / SELL のコードは入っているが、**実口座への live-fire 発注テストは未実施**
- 最低限、次を小サイズで確認する必要がある
  - BUY 1 回
  - SELL 1 回
  - cancel 1 回
  - 注文履歴 / 建玉反映

### 2. 部分約定の実地確認

- コード上は status を正規化している
- ただし、Webull JP 実口座で部分約定 payload をまだ観測していない

### 3. WSL2 の 24/365 問題

- PC がスリープ・再起動・Windows Update・Docker Desktop 停止になると止まる
- `docker compose up -d` だけで真の 24/365 にはならない
- 常時稼働を求めるなら VPS か専用マシンが必要

## 実運用前の最終チェック

- `.env` が設定済み
- Docker image が build 済み
- SQLite / Parquet の書き込み権限がある
- Webull JP の API 権限が有効
- FMP quota が足りる
- マシンがスリープしない
- market close 前の SELL を小サイズで実地確認

## 推奨 next step

1. 極小サイズで live order smoke test
2. 半日だけ監視し、quote -> bar -> signal -> order -> position の整合確認
3. 引けクローズ確認
4. その後に常時稼働へ移行
